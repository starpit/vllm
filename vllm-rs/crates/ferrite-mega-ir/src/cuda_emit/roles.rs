// SPDX-License-Identifier: Apache-2.0
//! Per-`MegaNode` role-body emit.
//!
//! Each variant's emit fn pulls typed-getter values from the node
//! and assembles four [`CuBlock`](super::cu::CuBlock)s — one for
//! each warp role (loader / launcher / consumer / storer). The
//! top-level [`lower_to_cuda`](super::lower_to_cuda) walks the tape
//! and concatenates each variant's role bodies into the per-role
//! sections of the final kernel.
//!
//! Sprint 1 ships RmsNorm only. Every other variant returns a
//! [`RoleBodies::skipped`] marker; the top-level walker collects
//! the skipped variant names into the [`CuVariant::skipped_variants`]
//! diagnostics list and renders a `// SKIPPED: <variant>` comment
//! at the corresponding tape position so the structural shape of
//! the emitted `.cu` is preserved.

use crate::nodes::{MegaNode, RmsNorm};
use crate::tape::TapeBudget;

use super::cu::{CuBlock, CuExpr, CuStmt};
use super::handles::{
    gmem_act_ptr_bf16, gmem_weight_ptr_bf16, page_as_sv_bf, page_consumed_sem, page_done_sem,
    page_ready_sem, scratch_as, warp_slice_sv_bf,
};
use super::tk;

/// Four role-body chunks for a single MegaNode. Concatenated by the
/// top-level walker into the per-role sections of the final kernel.
#[derive(Debug, Default)]
pub struct RoleBodies {
    pub loader: CuBlock,
    pub launcher: CuBlock,
    pub consumer: CuBlock,
    pub storer: CuBlock,
    /// `Some(variant)` if no role-body emit exists yet for this
    /// `MegaNode`; the four blocks are empty when this is set.
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

/// Per-`MegaNode` dispatch. The scaffolding is fixed (always four
/// roles, always traversed in tape order); each match arm calls a
/// per-variant `emit_*` fn. Phase 1: RmsNorm is the only variant
/// with a real body.
pub fn emit_role_bodies(node: &MegaNode, budget: TapeBudget) -> RoleBodies {
    match node {
        MegaNode::RmsNorm(n) => emit_rms_norm(n, budget),
        MegaNode::FusedQkvRopeCache(_) => RoleBodies::skipped("FusedQkvRopeCache"),
        MegaNode::Add(_) => RoleBodies::skipped("Add"),
        MegaNode::FusedAddRmsNorm(_) => RoleBodies::skipped("FusedAddRmsNorm"),
        MegaNode::FusedGateUpActivateMul(_) => RoleBodies::skipped("FusedGateUpActivateMul"),
        MegaNode::Embed(_) => RoleBodies::skipped("Embed"),
        MegaNode::ScalarMul(_) => RoleBodies::skipped("ScalarMul"),
        MegaNode::TanhSoftCap(_) => RoleBodies::skipped("TanhSoftCap"),
        MegaNode::ScalarOffsetRmsNorm(_) => RoleBodies::skipped("ScalarOffsetRmsNorm"),
        MegaNode::Gemm(_) => RoleBodies::skipped("Gemm"),
        MegaNode::FusedCublasGemmAdd(_) => RoleBodies::skipped("FusedCublasGemmAdd"),
        MegaNode::CutlassFusedNormGemm(_) => RoleBodies::skipped("CutlassFusedNormGemm"),
        MegaNode::AttentionViaCache(_) => RoleBodies::skipped("AttentionViaCache"),
        MegaNode::SpliceMmEmbeds(_) => RoleBodies::skipped("SpliceMmEmbeds"),
        MegaNode::BarrierSignal(_) => RoleBodies::skipped("BarrierSignal"),
        MegaNode::BarrierWait(_) => RoleBodies::skipped("BarrierWait"),
    }
}

// ============================================================
// RmsNorm — in-place, 2 pages (in_page + weight_page).
// ============================================================
//
// Page lifecycle (per `MEGA_IR_PLAN.md` §1#2 + the substrate
// semaphores in `ferrite_substrate.cuh`):
//
//   in_page:      Empty → Filled (loader TMA)
//                       → Produced (consumer rms_norm + warp::store)
//                       → Empty (storer TMA + arrive page_consumed)
//   weight_page:  Empty → Filled (loader TMA)
//                       → Empty (consumer arrive page_consumed)
//
// Reference for the role bodies' shape: TK upstream
// `third_party/thunderkittens/tests/vm/llama_official/utils.cuh`'s
// `rms_norm` helper + `rms_matvec_rope_append.cu` for the loader /
// consumer / storer page-handoff sequencing. ferrite splices the
// per-warp slice carving + the `kittens::wait` / `kittens::arrive`
// pairs around the TK helper call.

fn emit_rms_norm(n: &RmsNorm, budget: TapeBudget) -> RoleBodies {
    let in_page = n.in_page();
    let weight_page = n.weight_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    // Loader phase mirrors the storer's: both are the "inverse" of
    // the consumer phase relative to the cumulative arrives the IR
    // has already discharged. The substrate-proof typed primitive
    // [`MbarrierPhase`] enforces the parity at IR construction; we
    // just splice the two values the IR carries.
    let layer = n.layer().raw();
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let in_act_slot = n.in_act_slot().raw();
    let out_act_slot = n.out_act_slot().raw();
    let weight_accessor_idx = n.weight_accessor_idx().raw();
    let bar_reduce = n.consumer_bar_reduce().raw();
    let _bar_publish = n.consumer_bar_publish().raw();
    let eps_value = n.eps().raw();

    let ncw = budget.num_consumer_warps;
    debug_assert!(
        ncw > 0 && hidden_dim % ncw == 0,
        "emit_rms_norm: HIDDEN_DIM ({hidden_dim}) must be divisible by NCW ({ncw})"
    );
    let k_per_warp = hidden_dim / ncw;
    let num_layers_for_weight = budget.num_layers.max(1);

    // Substrate-page handles.
    let in_smem = page_as_sv_bf(in_page, hidden_dim);
    let weight_smem = page_as_sv_bf(weight_page, hidden_dim);
    let in_ready = page_ready_sem(in_page);
    let weight_ready = page_ready_sem(weight_page);
    let in_done = page_done_sem(in_page);
    let in_consumed = page_consumed_sem(in_page);
    let weight_consumed = page_consumed_sem(weight_page);
    let partial = scratch_as::<super::handles::F32>(n.partial_offset());
    let in_gmem = gmem_act_ptr_bf16(in_act_slot);
    let out_gmem = gmem_act_ptr_bf16(out_act_slot);
    let weight_gmem =
        gmem_weight_ptr_bf16(weight_accessor_idx, layer, num_layers_for_weight);

    let bf16_size_bytes = 2;
    let act_bytes = hidden_dim * num_tokens * bf16_size_bytes;
    let weight_bytes = hidden_dim * bf16_size_bytes;

    // ---------------- Loader ----------------
    //
    //   wait(page_consumed[in_page],     LOADER_PHASE);
    //   wait(page_consumed[weight_page], LOADER_PHASE);
    //   if (laneid() == 0) {
    //       tma::expect_bytes(page_ready[in_page], <act_bytes>);
    //       tma::load_async(in_smem, g.act_ptrs[IN_ACT_SLOT],
    //                       {0}, page_ready[in_page]);
    //       tma::expect_bytes(page_ready[weight_page], <weight_bytes>);
    //       tma::load_async(weight_smem, g.weight_ptrs[acc * NL + L],
    //                       {0}, page_ready[weight_page]);
    //   }
    let mut loader = CuBlock::new();
    let loader_phase = storer_phase; // loader waits on consumed/storer-arrived; same parity.
    loader.push(tk::wait(&in_consumed, loader_phase));
    loader.push(tk::wait(&weight_consumed, loader_phase));
    loader.push(CuStmt::new("if (kittens::laneid() == 0) {".to_string()));
    loader.push(tk::tma_expect_bytes(&in_ready, act_bytes));
    loader.push(tk::tma_load_async_bf16(
        &in_smem,
        &in_gmem,
        &CuExpr::new("{0}".to_string()),
        &in_ready,
    ));
    loader.push(tk::tma_expect_bytes(&weight_ready, weight_bytes));
    loader.push(tk::tma_load_async_bf16(
        &weight_smem,
        &weight_gmem,
        &CuExpr::new("{0}".to_string()),
        &weight_ready,
    ));
    loader.push(CuStmt::new("}".to_string()));

    // ---------------- Launcher ----------------
    //
    // RmsNorm has no launcher work today — register-budget on the
    // launcher slot is unused. Empty body; the role-dispatch in the
    // kernel scaffold still routes warpid=NCW+1 here so the warp
    // doesn't drop into a different op's launcher body.
    let launcher = CuBlock::new();

    // ---------------- Consumer ----------------
    //
    //   wait(page_ready[in_page],     CONSUMER_PHASE);
    //   wait(page_ready[weight_page], CONSUMER_PHASE);
    //   <per-warp slices>
    //   auto __rms_rv = ferrite::tk::rms_norm_vec<NCW, HIDDEN_DIM, BAR>(
    //       wt_slice, in_slice, EPS, partial);
    //   warp::store(in_slice, __rms_rv);
    //   group<NCW>::sync(BAR_PUBLISH);
    //   if (warpid() == 0) {
    //       arrive(page_done[in_page]);
    //       arrive(page_consumed[weight_page]);
    //   }
    let mut consumer = CuBlock::new();
    consumer.push(tk::wait(&in_ready, consumer_phase));
    consumer.push(tk::wait(&weight_ready, consumer_phase));
    let in_slice = warp_slice_sv_bf(&in_smem, ncw, k_per_warp);
    let weight_slice = warp_slice_sv_bf(&weight_smem, ncw, k_per_warp);
    let eps_lit = format!("{:e}f", eps_value);
    let rv = tk::rms_norm_vec(
        ncw,
        hidden_dim,
        bar_reduce,
        &weight_slice,
        &in_slice,
        &CuExpr::new(eps_lit),
        &partial,
    );
    let (let_stmt, rv_named) = tk::let_reg_col_vec("__rms_rv", rv);
    consumer.push(let_stmt);
    consumer.push(tk::warp_store_bf16(&in_slice, &rv_named));
    // Cross-warp publish barrier so all NCW slices land before warp 0
    // signals page_done.
    consumer.push(CuStmt::new(format!(
        "kittens::group<{ncw}>::sync({});",
        n.consumer_bar_publish().raw()
    )));
    consumer.push(CuStmt::new("if (kittens::warpid() == 0) {".to_string()));
    consumer.push(tk::arrive(&in_done));
    consumer.push(tk::arrive(&weight_consumed));
    consumer.push(CuStmt::new("}".to_string()));

    // ---------------- Storer ----------------
    //
    //   wait(page_done[in_page], STORER_PHASE);
    //   if (laneid() == 0) {
    //       tma::store_async(g.act_ptrs[OUT_ACT_SLOT], in_smem, {0});
    //       tma::store_async_wait();
    //       arrive(page_consumed[in_page]);
    //   }
    let mut storer = CuBlock::new();
    storer.push(tk::wait(&in_done, storer_phase));
    storer.push(CuStmt::new("if (kittens::laneid() == 0) {".to_string()));
    storer.push(tk::tma_store_async_bf16(
        &out_gmem,
        &in_smem,
        &CuExpr::new("{0}".to_string()),
    ));
    storer.push(tk::tma_store_async_wait());
    storer.push(tk::arrive(&in_consumed));
    storer.push(CuStmt::new("}".to_string()));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

