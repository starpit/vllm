// SPDX-License-Identifier: Apache-2.0
//! Rust API surface of the TK / ferrite-tk primitive set.
//!
//! Each function below mirrors a C++ entry point in `kittens` /
//! `ferrite::tk` (see `third_party/thunderkittens/include/...` and
//! `crates/ferrite-kernels/csrc/tk/ferrite_tk_helpers.cuh`). The
//! Rust function signature pins the typed handles each primitive
//! takes (dtype-checked at codegen build time); the body emits the
//! corresponding CUDA call as a [`CuStmt`](super::cu::CuStmt) or a
//! handle wrapping a [`CuExpr`](super::cu::CuExpr).
//!
//! "Calling" a TK primitive from emit code is a Rust function call.
//! Wrong dtype handles fail to typecheck. Length / template-arg
//! mismatches are formatted into the CUDA source and re-checked by
//! `nvcc`'s `static_assert`s inside the TK helpers.
//!
//! The surface starts narrow (RmsNorm Sprint 1) and grows one
//! variant at a time. Each new variant adds whichever primitives
//! its TK reference uses, no more.

use super::cu::{CuExpr, CuStmt};
use super::handles::{Bf16, GmemPtr, RegColVec, ScratchPtr, Semaphore, SmemColVec, F32};

// ============================================================
// Page-handoff semaphores — kittens::wait, kittens::arrive
// ============================================================

/// `kittens::wait(sem, phase);` — block the calling warp until the
/// semaphore's phase bit matches `phase`.
pub fn wait(sem: &Semaphore, phase: u32) -> CuStmt {
    CuStmt::new(format!(
        "kittens::wait({sem}, {phase});",
        sem = sem.expr()
    ))
}

/// `kittens::arrive(sem);` — flip the semaphore's phase bit.
pub fn arrive(sem: &Semaphore) -> CuStmt {
    CuStmt::new(format!("kittens::arrive({sem});", sem = sem.expr()))
}

// ============================================================
// TMA — tile-memory accelerator load/store
// ============================================================

/// `kittens::tma::expect_bytes(sem, bytes);` — declare the next TMA
/// transaction's byte count to the semaphore so `wait` knows how
/// much arrival traffic to expect.
pub fn tma_expect_bytes(sem: &Semaphore, bytes: u32) -> CuStmt {
    CuStmt::new(format!(
        "kittens::tma::expect_bytes({sem}, {bytes});",
        sem = sem.expr()
    ))
}

/// `kittens::tma::load_async(dst_smem, src_gmem, {indices...},
/// sem);` — asynchronously stream a tile from gmem into smem,
/// arriving on `sem` when complete.
///
/// `indices` is a single CUDA expression representing the brace-
/// initialised gmem coordinate list (e.g. `{layer, tok}`); the
/// caller is responsible for shaping it.
pub fn tma_load_async_bf16(
    dst: &SmemColVec<Bf16>,
    src: &GmemPtr<Bf16>,
    indices: &CuExpr,
    sem: &Semaphore,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::tma::load_async({dst}, {src}, {indices}, {sem});",
        dst = dst.expr(),
        src = src.expr(),
        sem = sem.expr()
    ))
}

/// `kittens::tma::store_async(dst_gmem, src_smem, {indices...});` —
/// asynchronously stream a tile from smem to gmem.
pub fn tma_store_async_bf16(
    dst: &GmemPtr<Bf16>,
    src: &SmemColVec<Bf16>,
    indices: &CuExpr,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::tma::store_async({dst}, {src}, {indices});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

/// `kittens::tma::store_async_wait();` — wait for every outstanding
/// TMA store from this warp to be globally visible.
pub fn tma_store_async_wait() -> CuStmt {
    CuStmt::new("kittens::tma::store_async_wait();".to_string())
}

// ============================================================
// Warp-level register-tile primitives — kittens::warp::*
// ============================================================

/// `kittens::warp::store(dst_smem, src_rv);` — write a register
/// vector back to a shared column vector. Lengths must match;
/// re-checked by CUDA `static_assert` inside `kittens::warp::store`.
pub fn warp_store_bf16(dst: &SmemColVec<Bf16>, src: &RegColVec<F32>) -> CuStmt {
    debug_assert_eq!(
        dst.len(),
        src.len(),
        "warp_store_bf16: dst.len() must equal src.len()"
    );
    CuStmt::new(format!(
        "kittens::warp::store({dst}, {src});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

/// Declare a CUDA local `kittens::rv_fl<LEN> <name>;` and return the
/// statement plus a handle bound to the named register vector. Used
/// before `warp::load` since TK's `warp::load(rv, sv)` mutates
/// `rv` in-place rather than returning.
pub fn decl_rv_fl(name: &str, len: u32) -> (CuStmt, RegColVec<F32>) {
    let stmt = CuStmt::new(format!("kittens::rv_fl<{len}> {name};"));
    (stmt, RegColVec::from_expr(CuExpr::new(name.to_string()), len))
}

/// `kittens::warp::load(dst_rv, src_smem);` — load a bf16 shared
/// column vector into an fp32 register vector (TK widens bf16→fp32
/// on load). Lengths must match.
pub fn warp_load_bf16_to_f32(dst: &RegColVec<F32>, src: &SmemColVec<Bf16>) -> CuStmt {
    debug_assert_eq!(
        dst.len(),
        src.len(),
        "warp_load_bf16_to_f32: dst.len() must equal src.len()"
    );
    CuStmt::new(format!(
        "kittens::warp::load({dst}, {src});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

/// `kittens::warp::add(dst_rv, lhs_rv, rhs_rv);` — elementwise
/// fp32 register-vector add. `dst` may alias `lhs` or `rhs`. All
/// three lengths must match.
pub fn warp_add_f32(
    dst: &RegColVec<F32>,
    lhs: &RegColVec<F32>,
    rhs: &RegColVec<F32>,
) -> CuStmt {
    debug_assert_eq!(
        dst.len(),
        lhs.len(),
        "warp_add_f32: dst.len() must equal lhs.len()"
    );
    debug_assert_eq!(
        lhs.len(),
        rhs.len(),
        "warp_add_f32: lhs.len() must equal rhs.len()"
    );
    CuStmt::new(format!(
        "kittens::warp::add({dst}, {lhs}, {rhs});",
        dst = dst.expr(),
        lhs = lhs.expr(),
        rhs = rhs.expr()
    ))
}

/// `kittens::warp::sync();` — single-warp sync (lane convergence).
pub fn warp_sync() -> CuStmt {
    CuStmt::new("kittens::warp::sync();".to_string())
}

/// `kittens::warp::mul(dst_rv, lhs_rv, <scalar>);` — elementwise
/// fp32 register-vector multiply by a scalar literal expression.
/// `dst` may alias `lhs`. Lengths match by construction.
pub fn warp_mul_f32_scalar(
    dst: &RegColVec<F32>,
    lhs: &RegColVec<F32>,
    scalar: &CuExpr,
) -> CuStmt {
    debug_assert_eq!(
        dst.len(),
        lhs.len(),
        "warp_mul_f32_scalar: dst.len() must equal lhs.len()"
    );
    CuStmt::new(format!(
        "kittens::warp::mul({dst}, {lhs}, {scalar});",
        dst = dst.expr(),
        lhs = lhs.expr()
    ))
}

/// `kittens::warp::add(dst_rv, lhs_rv, <scalar>);` — elementwise
/// fp32 register-vector add of a scalar (broadcast). `dst` may
/// alias `lhs`.
pub fn warp_add_f32_scalar(
    dst: &RegColVec<F32>,
    lhs: &RegColVec<F32>,
    scalar: &CuExpr,
) -> CuStmt {
    debug_assert_eq!(
        dst.len(),
        lhs.len(),
        "warp_add_f32_scalar: dst.len() must equal lhs.len()"
    );
    CuStmt::new(format!(
        "kittens::warp::add({dst}, {lhs}, {scalar});",
        dst = dst.expr(),
        lhs = lhs.expr()
    ))
}

// ============================================================
// ferrite::tk helpers — generalised TK primitives
// ============================================================

/// `ferrite::tk::rms_norm_vec<NCW, HIDDEN_DIM, BAR>(rms_scale_smem,
/// activations_smem, eps, partial_sums_scratch)` returning a
/// `kittens::rv_fl<HIDDEN_DIM / NCW>` register vector.
///
/// Mirrors `ferrite_tk_helpers.cuh::rms_norm_vec`. Caller passes
/// per-warp slices of the rms-scale and activation shared vectors;
/// the TK helper does the warp-local sum-of-squares, the
/// CONSUMER-warp-group sync on `BAR`, the global reduction, the
/// rsqrt scale, and the in-register multiply by the rms-scale
/// slice.
///
/// `bar` must be in `1..=15` (bar 0 is `__syncthreads`); the IR
/// carries it as a `BarSyncId` typed primitive whose `new()`
/// discharges the bound. The caller passes the runtime value here.
pub fn rms_norm_vec(
    ncw: u32,
    hidden_dim: u32,
    bar: u32,
    rms_scale_warp_slice: &SmemColVec<Bf16>,
    activations_warp_slice: &SmemColVec<Bf16>,
    eps_expr: &CuExpr,
    partial_sums_scratch: &ScratchPtr<F32>,
) -> RegColVec<F32> {
    debug_assert!(
        ncw > 0 && rms_scale_warp_slice.len() == hidden_dim / ncw,
        "rms_norm_vec: scale slice length ({}) must equal HIDDEN_DIM ({hidden_dim}) / NCW ({ncw})",
        rms_scale_warp_slice.len()
    );
    debug_assert_eq!(
        rms_scale_warp_slice.len(),
        activations_warp_slice.len(),
        "rms_norm_vec: scale and activation slice lengths must match"
    );
    debug_assert!(
        (1..=15).contains(&bar),
        "rms_norm_vec: bar ({bar}) must be in 1..=15"
    );
    let k_per_warp = rms_scale_warp_slice.len();
    RegColVec::from_expr(
        CuExpr::new(format!(
            "ferrite::tk::rms_norm_vec<{ncw}, {hidden_dim}, {bar}>({scale}, {act}, {eps}, {partial})",
            scale = rms_scale_warp_slice.expr(),
            act = activations_warp_slice.expr(),
            eps = eps_expr,
            partial = partial_sums_scratch.expr()
        )),
        k_per_warp,
    )
}

/// `ferrite::tk::rms_norm_scale_from_rv<NCW, HIDDEN_DIM, BAR>(rv,
/// eps, partial_sums_scratch)` returning the scalar `rsqrt(mean(x^2)
/// + eps)` replicated across all lanes of the warp.
///
/// Mirrors `ferrite_tk_helpers.cuh::rms_norm_scale_from_rv`. Used
/// by `FusedAddRmsNorm` (where the rv comes from a residual add in
/// registers, not from a shared load — `rms_norm_vec` is the wrong
/// shape because it does its own `warp::load`).
///
/// Returns a CUDA expression of fp32 type — typically bound to a
/// local with `auto scale = ...;` then passed as the scalar arg to
/// `warp::mul`.
pub fn rms_norm_scale_from_rv(
    ncw: u32,
    hidden_dim: u32,
    bar: u32,
    rv: &RegColVec<F32>,
    eps_expr: &CuExpr,
    partial_sums_scratch: &ScratchPtr<F32>,
) -> CuExpr {
    debug_assert!(
        (1..=15).contains(&bar),
        "rms_norm_scale_from_rv: bar ({bar}) must be in 1..=15"
    );
    debug_assert!(
        ncw > 0 && rv.len() == hidden_dim / ncw,
        "rms_norm_scale_from_rv: rv length ({}) must equal HIDDEN_DIM ({hidden_dim}) / NCW ({ncw})",
        rv.len()
    );
    CuExpr::new(format!(
        "ferrite::tk::rms_norm_scale_from_rv<{ncw}, {hidden_dim}, {bar}>({rv}, {eps}, {partial})",
        rv = rv.expr(),
        eps = eps_expr,
        partial = partial_sums_scratch.expr()
    ))
}

/// `kittens::warp::mul(dst_rv, lhs_rv, rhs_rv);` — elementwise
/// fp32 register-vector multiply. `dst` may alias `lhs` or `rhs`.
pub fn warp_mul_f32(
    dst: &RegColVec<F32>,
    lhs: &RegColVec<F32>,
    rhs: &RegColVec<F32>,
) -> CuStmt {
    debug_assert_eq!(
        dst.len(),
        lhs.len(),
        "warp_mul_f32: dst.len() must equal lhs.len()"
    );
    debug_assert_eq!(
        lhs.len(),
        rhs.len(),
        "warp_mul_f32: lhs.len() must equal rhs.len()"
    );
    CuStmt::new(format!(
        "kittens::warp::mul({dst}, {lhs}, {rhs});",
        dst = dst.expr(),
        lhs = lhs.expr(),
        rhs = rhs.expr()
    ))
}

/// `ferrite::tk::tanh_softcap_vec(rv, cap);` — in-place per-lane
/// `x = tanhf(x / cap) * cap`. Mirrors
/// `ferrite_tk_helpers.cuh::tanh_softcap_vec`. No cross-warp
/// coordination — every lane operates on its own register slots.
pub fn tanh_softcap_vec(rv: &RegColVec<F32>, cap: &CuExpr) -> CuStmt {
    CuStmt::new(format!(
        "ferrite::tk::tanh_softcap_vec({rv}, {cap});",
        rv = rv.expr()
    ))
}

/// `ferrite::barrier_signal(<slot_ptr>, <count>);` — atomicAdd
/// a gmem cross-CTA counter to flag completion of a prior op.
/// One thread per CTA; mirrors `ferrite_barrier.cuh::barrier_signal`.
pub fn barrier_signal(slot_ptr: &CuExpr, count: u32) -> CuStmt {
    CuStmt::new(format!("ferrite::barrier_signal({slot_ptr}, {count});"))
}

/// `ferrite::barrier_wait(<slot_ptr>, <expected>);` — spin-load
/// the gmem counter until it reaches `expected`. One thread per
/// CTA; mirrors `ferrite_barrier.cuh::barrier_wait`.
pub fn barrier_wait(slot_ptr: &CuExpr, expected: u32) -> CuStmt {
    CuStmt::new(format!(
        "ferrite::barrier_wait({slot_ptr}, {expected});"
    ))
}

/// Bind a returned [`RegColVec`] expression to a CUDA local
/// variable, returning the `auto <name> = <expr>;` statement plus a
/// fresh handle that refers to the bound name. Useful when an op's
/// role body wants to consume a register vector more than once
/// (compute + store).
pub fn let_reg_col_vec<T: super::handles::DtypeName>(
    name: &str,
    value: RegColVec<T>,
) -> (CuStmt, RegColVec<T>) {
    let len = value.len();
    let stmt = CuStmt::new(format!(
        "auto {name} = {expr};",
        expr = value.expr()
    ));
    (stmt, RegColVec::from_expr(CuExpr::new(name.to_string()), len))
}
