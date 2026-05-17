// SPDX-License-Identifier: Apache-2.0
//! Rust API surface mirroring TK 2.0's C++ primitive surface.
//!
//! Each function below corresponds to exactly one TK 2.0 entry
//! point in `third_party/thunderkittens/include/`. The Rust
//! signature pins the typed handles (dtype-checked at codegen
//! build time); the body emits the corresponding CUDA call as a
//! [`CuStmt`](super::cu::CuStmt) (or returns a typed handle
//! wrapping a [`CuExpr`](super::cu::CuExpr) for value-producing
//! ops like `decl_rv_fl`).
//!
//! "Calling" a TK 2.0 primitive from emit code is a Rust function
//! call with type-checked args; the emitted CUDA is the function's
//! body. Wrong dtype handles fail to typecheck. Length / shape
//! mismatches surface as TK 2.0 `static_assert` failures at
//! `nvcc` time on the pod (which is the DOD per
//! `MEGA_IR_PLAN.md` §8.0a).
//!
//! Every function comments its TK 2.0 source citation. NEVER add
//! a function here without first opening the TK 2.0 header and
//! confirming the signature.

use super::cu::{CuExpr, CuStmt};
use super::handles::{Bf16, F32, GmemPtrRaw, Rv, RvCudaName, Semaphore, Sv};

// ============================================================
// kittens::group<N>::wait / arrive / sync
//
// All sync primitives live inside `kittens::group<N>::*`. There is
// NO top-level `kittens::wait/arrive`. The `kittens::warp` alias
// is `kittens::group<1>` (group.cuh:114).
// ============================================================

/// `kittens::group<N>::wait(sem, phase);`
///
/// Source: `include/ops/group/util/sync.cuh:112`
pub fn group_wait(n: u32, sem: &Semaphore, phase: u32) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{n}>::wait({sem}, {phase});",
        sem = sem.expr()
    ))
}

/// `kittens::group<N>::arrive(sem);` — auto-laneid-gates internally.
///
/// Source: `include/ops/group/util/sync.cuh:69`
pub fn group_arrive(n: u32, sem: &Semaphore) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{n}>::arrive({sem});",
        sem = sem.expr()
    ))
}

/// `kittens::group<N>::sync(int id);` — named PTX bar.sync,
/// id ∈ 1..=15.
///
/// Source: `include/ops/group/group.cuh:33`
pub fn group_sync_named(n: u32, bar_id: u32) -> CuStmt {
    debug_assert!(
        (1..=15).contains(&bar_id),
        "group_sync_named: bar_id ({bar_id}) must be in 1..=15"
    );
    CuStmt::new(format!("kittens::group<{n}>::sync({bar_id});"))
}

// ============================================================
// kittens::group<N>::tma::* — non-tensor TMA path (raw pointers,
// byte count). Picked over tensor TMA for first cuda_emit pass
// because the existing host-side ABI passes raw `bf16**` pointer
// arrays — no `kittens::gl<...>` construction needed. See
// `CUDA_EMIT_TK20_AUDIT.md` §11 + §14 (Decision 1).
// ============================================================

/// `kittens::group<N>::tma::expect_bytes(sem, bytes);` —
/// auto-laneid-gates internally.
///
/// Source: `include/ops/group/util/tma.cuh:18`
pub fn group_tma_expect_bytes(n: u32, sem: &Semaphore, bytes: u32) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{n}>::tma::expect_bytes({sem}, {bytes});",
        sem = sem.expr()
    ))
}

/// `kittens::group<N>::tma::load_async(void* dst, void* src,
/// uint32_t size_bytes, semaphore& bar);` — auto-laneid-gates.
/// Casts the typed `Sv<Bf16>` dst and `GmemPtrRaw<Bf16>` src to
/// `void*` at the call site.
///
/// Source: `include/ops/group/util/tma.cuh:72` (group-scope
/// non-tensor variant; delegates to thread-scope at `:86` of
/// `include/ops/thread/util/tma.cuh`).
pub fn group_tma_load_async_raw(
    n: u32,
    dst: &Sv<Bf16>,
    src: &GmemPtrRaw<Bf16>,
    size_bytes: u32,
    sem: &Semaphore,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{n}>::tma::load_async(\
         reinterpret_cast<void*>(&{dst}), \
         reinterpret_cast<void*>({src}), \
         {size_bytes}, \
         {sem});",
        dst = dst.expr(),
        src = src.expr(),
        sem = sem.expr()
    ))
}

/// `kittens::group<N>::tma::store_async(void* dst, void* src,
/// uint32_t size_bytes);` — auto-laneid-gates. Stores from a
/// shared vec to a raw gmem pointer.
///
/// Source: `include/ops/group/util/tma.cuh:82`
pub fn group_tma_store_async_raw(
    n: u32,
    dst: &GmemPtrRaw<Bf16>,
    src: &Sv<Bf16>,
    size_bytes: u32,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{n}>::tma::store_async(\
         reinterpret_cast<void*>({dst}), \
         reinterpret_cast<void*>(&{src}), \
         {size_bytes});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

/// `kittens::group<N>::tma::store_async_wait();` — wait for
/// outstanding TMA stores.
///
/// Source: `include/ops/group/util/tma.cuh:46`
pub fn group_tma_store_async_wait(n: u32) -> CuStmt {
    CuStmt::new(format!("kittens::group<{n}>::tma::store_async_wait();"))
}

// ============================================================
// Group-scope shared <-> register transfer.
//
// kittens::group<NCW>::load(rv, sv) and store(sv, rv) auto-slice
// the shared vec into per-warp subvecs when NCW > 1. The
// caller's responsibility: ensure `sv.len() == rv.len() * NCW`.
// ============================================================

/// `kittens::group<N>::load(rv, sv);` — when N == 1 a direct
/// load; when N > 1 the SV is auto-sliced into
/// `subvec<RV::length>(warpid())` per-warp.
///
/// Source: `include/ops/group/memory/vec/shared_to_register.cuh:14`
/// (the `else` branch at line 84 does the auto-subvec).
pub fn group_load_sv_to_rv_bf16_to_f32(
    n: u32,
    rv: &Rv<F32>,
    sv: &Sv<Bf16>,
) -> CuStmt {
    debug_assert!(
        n > 0 && sv.len() == rv.len() * n,
        "group_load_sv_to_rv_bf16_to_f32: sv.len ({}) must equal rv.len ({}) * N ({n})",
        sv.len(),
        rv.len()
    );
    CuStmt::new(format!(
        "kittens::group<{n}>::load({rv}, {sv});",
        rv = rv.expr(),
        sv = sv.expr()
    ))
}

/// `kittens::group<N>::store(sv, rv);` — symmetric to load.
///
/// Source: `include/ops/group/memory/vec/shared_to_register.cuh:101`
pub fn group_store_rv_to_sv_f32_to_bf16(
    n: u32,
    sv: &Sv<Bf16>,
    rv: &Rv<F32>,
) -> CuStmt {
    debug_assert!(
        n > 0 && sv.len() == rv.len() * n,
        "group_store_rv_to_sv_f32_to_bf16: sv.len ({}) must equal rv.len ({}) * N ({n})",
        sv.len(),
        rv.len()
    );
    CuStmt::new(format!(
        "kittens::group<{n}>::store({sv}, {rv});",
        sv = sv.expr(),
        rv = rv.expr()
    ))
}

// ============================================================
// kittens::warp::* register-vec maps. (warp == group<1>.)
// ============================================================

/// `kittens::warp::copy(dst, src);` — register-vec dtype-converting
/// copy.
///
/// Source: `include/ops/group/register/vec/maps.cuh:176`
pub fn warp_copy_rv<T: super::handles::DtypeName + RvCudaName>(
    dst: &Rv<T>,
    src: &Rv<T>,
) -> CuStmt {
    debug_assert_eq!(dst.len(), src.len());
    CuStmt::new(format!(
        "kittens::warp::copy({dst}, {src});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

/// `kittens::warp::mul(dst, lhs, rhs);` — rv-rv elementwise mul.
///
/// Source: `include/ops/group/register/vec/maps.cuh:359` (rv-rv
/// overload via the `bin_op` template at `:35`).
pub fn warp_mul_rv_rv(
    dst: &Rv<F32>,
    lhs: &Rv<F32>,
    rhs: &Rv<F32>,
) -> CuStmt {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(lhs.len(), rhs.len());
    CuStmt::new(format!(
        "kittens::warp::mul({dst}, {lhs}, {rhs});",
        dst = dst.expr(),
        lhs = lhs.expr(),
        rhs = rhs.expr()
    ))
}

/// `kittens::warp::add(dst, lhs, rhs);` — rv-rv elementwise add.
///
/// Source: `include/ops/group/register/vec/maps.cuh:333`
pub fn warp_add_rv_rv(
    dst: &Rv<F32>,
    lhs: &Rv<F32>,
    rhs: &Rv<F32>,
) -> CuStmt {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(lhs.len(), rhs.len());
    CuStmt::new(format!(
        "kittens::warp::add({dst}, {lhs}, {rhs});",
        dst = dst.expr(),
        lhs = lhs.expr(),
        rhs = rhs.expr()
    ))
}

/// `kittens::warp::add(dst, src, scalar);` — rv-scalar broadcast add.
///
/// Source: `include/ops/group/register/vec/maps.cuh:54-55` (the
/// `bin_op(T &dst, const T &src, const typename T::dtype &param)`
/// scalar overload — `add` reaches it via the same dispatch).
pub fn warp_add_rv_scalar_f32(
    dst: &Rv<F32>,
    src: &Rv<F32>,
    scalar: &CuExpr,
) -> CuStmt {
    debug_assert_eq!(dst.len(), src.len());
    CuStmt::new(format!(
        "kittens::warp::add({dst}, {src}, {scalar});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

/// `kittens::warp::mul(dst, src, scalar);` — rv-scalar broadcast.
///
/// Source: `include/ops/group/register/vec/maps.cuh:54-55` (the
/// `bin_op(T &dst, const T &src, const typename T::dtype &param)`
/// scalar overload — `mul` reaches it via the same dispatch).
pub fn warp_mul_rv_scalar_f32(
    dst: &Rv<F32>,
    src: &Rv<F32>,
    scalar: &CuExpr,
) -> CuStmt {
    debug_assert_eq!(dst.len(), src.len());
    CuStmt::new(format!(
        "kittens::warp::mul({dst}, {src}, {scalar});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

/// `kittens::warp::sum(scalar_out, rv);` — warp-wide reduction
/// into a scalar mut-ref out arg.
///
/// Source: `include/ops/group/register/vec/reductions.cuh:129`
pub fn warp_sum_to_scalar_f32(
    scalar_out: &CuExpr,
    src: &Rv<F32>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::sum({scalar_out}, {src});",
        src = src.expr()
    ))
}

/// Cross-warp fp32 sum reduction via the canonical TK 2.0 idiom:
/// per-warp partial sum already in `partial_in_name` (lane 0 has the
/// authoritative value); lane 0 of each warp writes to
/// `scratch[warpid()]`; group<NCW>::sync(BAR) so all warps' partials
/// are visible; every lane reads all NCW slots into
/// `scalar_out_name` (which the caller declares as a `float` local
/// initialized to 0).
///
/// Composes only TK 2.0 primitives: `kittens::laneid()` /
/// `kittens::warpid()` (group.cuh:29-30) +
/// `kittens::group<NCW>::sync(int id)` (group.cuh:33). The scratch
/// indexing and the for-loop are vanilla C++ wrapping those
/// primitives — same pattern TK uses internally for cross-warp
/// reductions.
pub fn cross_warp_reduce_sum_f32(
    scalar_out_name: &str,
    partial_in_name: &str,
    scratch: &super::handles::ScratchPtr<F32>,
    ncw: u32,
    bar_id: u32,
) -> CuStmt {
    debug_assert!(
        (1..=15).contains(&bar_id),
        "cross_warp_reduce_sum_f32: bar_id ({bar_id}) must be in 1..=15"
    );
    CuStmt::new(format!(
        "if (kittens::laneid() == 0) {{ {scratch}[kittens::warpid()] = {partial}; }}\n\
         kittens::group<{ncw}>::sync({bar_id});\n\
         #pragma unroll\n\
         for (int __cw_i = 0; __cw_i < {ncw}; ++__cw_i) {{ {out} += {scratch}[__cw_i]; }}",
        scratch = scratch.expr(),
        partial = partial_in_name,
        out = scalar_out_name
    ))
}

/// `const float <name> = rsqrtf(<full_sum_name> / <hidden_dim>.0f
/// + <eps>f);` — declare and bind the canonical RMS scale local.
/// Wraps a CUDA math intrinsic + arithmetic; not a TK 2.0 primitive
/// per se but the standard expression used in every RmsNorm-flavor
/// op body.
pub fn decl_rms_scale_local(
    name: &str,
    full_sum_name: &str,
    hidden_dim: u32,
    eps: f32,
) -> (CuStmt, CuExpr) {
    let stmt = CuStmt::new(format!(
        "const float {name} = rsqrtf({sum} / {hidden_dim}.0f + {eps:e}f);",
        sum = full_sum_name
    ));
    (stmt, CuExpr::new(name.to_string()))
}

/// Wrap a sequence of statements in `if (kittens::warpid() == 0) {
/// ... }` — the canonical "warp 0 publishes" gate. Used to gate
/// `kittens::group<1>::arrive(sem)` calls (which auto-laneid-gate
/// internally but don't warp-id-gate; without the warp-id gate
/// every warp would arrive once, multiplying the count by NCW).
pub fn block_warp_zero(stmts: &[CuStmt]) -> CuStmt {
    let body = stmts
        .iter()
        .map(|s| format!("    {}", s.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    CuStmt::new(format!(
        "if (kittens::warpid() == 0) {{\n{body}\n}}"
    ))
}

/// `kittens::warp::apply(dst, src, lambda);` — per-lane unary map.
/// `lambda_body` is a CUDA expression in `x` (the per-lane fp32
/// value) returning a fp32 result. The wrapper takes a 2-arg
/// lambda `(int /*idx*/, float x) -> float` because that's the TK
/// 2.0 signature; we ignore the idx in lambda_body callers.
///
/// Source: `include/ops/group/register/vec/maps.cuh:79-112`
pub fn warp_apply_f32_lambda(
    dst: &Rv<F32>,
    src: &Rv<F32>,
    lambda_body: &str,
) -> CuStmt {
    debug_assert_eq!(dst.len(), src.len());
    CuStmt::new(format!(
        "kittens::warp::apply({dst}, {src}, [] __device__ (int /*idx*/, float x) {{ return {body}; }});",
        dst = dst.expr(),
        src = src.expr(),
        body = lambda_body
    ))
}

// ============================================================
// CUDA local-variable declarations.
// ============================================================

/// `kittens::rv_fl<LEN> <name>;` — declare a local register vector
/// + return a handle bound to it.
pub fn decl_rv_fl(name: &str, len: u32) -> (CuStmt, Rv<F32>) {
    let stmt = CuStmt::new(format!("kittens::rv_fl<{len}> {name};"));
    (stmt, Rv::from_expr(CuExpr::new(name.to_string()), len))
}

/// `float <name> = 0.0f;` — declare a fp32 scalar accumulator;
/// returns the bound `CuExpr` so callers can pass it to
/// [`warp_sum_to_scalar_f32`] etc.
pub fn decl_local_f32(name: &str, init: &str) -> (CuStmt, CuExpr) {
    let stmt = CuStmt::new(format!("float {name} = {init};"));
    (stmt, CuExpr::new(name.to_string()))
}
