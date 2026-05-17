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
use super::handles::{
    Bf16, F32, GmemPtrRaw, Rt, RtCol, RtCudaName, RtLayoutTag, RtRow, Rv, RvCudaName,
    Semaphore, St, Sv,
};

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

/// Per-token TMA gather: emits `kittens::group<1>::tma::expect_bytes`
/// for the total transfer + a `for (int t = 0; t < NUM_TOKENS; ++t)`
/// loop calling non-tensor `kittens::group<1>::tma::load_async` per
/// token, with the source row computed as
/// `embed_table_ptr + input_ids[t] * HIDDEN_DIM` and the
/// destination row at `out_byte_ptr + t * HIDDEN_DIM * sizeof(bf16)`.
///
/// Composes only TK 2.0 primitives:
///   - `kittens::group<1>::tma::expect_bytes`  (util/tma.cuh:18)
///   - `kittens::group<1>::tma::load_async(void*, void*, bytes, sem)`
///                                           (util/tma.cuh:72)
/// + a vanilla C++ `for` loop and `+= tok * row_bytes` pointer
/// arithmetic. The arithmetic is the standard "stride per row"
/// pattern used by every per-token gather; not invented scheduling.
pub fn embed_per_token_gather(
    out_byte_ptr: &CuExpr,
    embed_table_ptr: &GmemPtrRaw<Bf16>,
    input_ids_ptr: &CuExpr,
    hidden_dim: u32,
    num_tokens: u32,
    page_ready: &Semaphore,
) -> CuStmt {
    let row_bytes = hidden_dim * 2;
    let total_bytes = row_bytes * num_tokens;
    CuStmt::new(format!(
        "kittens::group<1>::tma::expect_bytes({sem}, {total_bytes});\n\
         for (int __embed_t = 0; __embed_t < {num_tokens}; ++__embed_t) {{\n\
         \x20   const uint32_t __embed_row = {ids}[__embed_t];\n\
         \x20   void* __embed_dst = static_cast<void*>(\
         reinterpret_cast<char*>({dst}) + __embed_t * {row_bytes});\n\
         \x20   void* __embed_src = static_cast<void*>(\
         {table} + static_cast<size_t>(__embed_row) * {hidden_dim});\n\
         \x20   kittens::group<1>::tma::load_async(__embed_dst, __embed_src, {row_bytes}, {sem});\n\
         }}",
        sem = page_ready.expr(),
        ids = input_ids_ptr,
        dst = out_byte_ptr,
        table = embed_table_ptr.expr(),
    ))
}

/// Per-token TMA store: emits a `for (int t = 0; t < NUM_TOKENS;
/// ++t)` loop calling non-tensor `kittens::group<1>::tma::
/// store_async` per row, target gmem at `out_gmem + t * HIDDEN_DIM`.
/// Source rows are contiguous in shared memory at `src_byte +
/// t * HIDDEN_DIM * sizeof(bf16)`.
///
/// Single TMA store would also work for contiguous gmem regions,
/// but per-token mirrors the loader's gather pattern and stays
/// correct even if the gmem destination layout changes.
pub fn per_token_tma_store(
    out_gmem: &GmemPtrRaw<Bf16>,
    src_byte_ptr: &CuExpr,
    hidden_dim: u32,
    num_tokens: u32,
) -> CuStmt {
    let row_bytes = hidden_dim * 2;
    CuStmt::new(format!(
        "for (int __pt_t = 0; __pt_t < {num_tokens}; ++__pt_t) {{\n\
         \x20   void* __pt_dst = static_cast<void*>(\
         {gmem} + static_cast<size_t>(__pt_t) * {hidden_dim});\n\
         \x20   void* __pt_src = static_cast<void*>(\
         reinterpret_cast<char*>({src}) + __pt_t * {row_bytes});\n\
         \x20   kittens::group<1>::tma::store_async(__pt_dst, __pt_src, {row_bytes});\n\
         }}",
        gmem = out_gmem.expr(),
        src = src_byte_ptr,
    ))
}

/// `ferrite::barrier_signal(slot_ptr, count);` — gmem cross-CTA
/// counter bump from one thread. Provided by ferrite substrate
/// (`crates/ferrite-kernels/csrc/tk/ferrite_barrier.cuh`). The
/// helper does `__threadfence()` + `atomicAdd` from a single
/// thread; we gate to laneid 0 explicitly + intra-warp sync after
/// so other lanes converge.
pub fn ferrite_barrier_signal(slot_ptr: &CuExpr, count: u32) -> CuStmt {
    CuStmt::new(format!(
        "if (kittens::laneid() == 0) {{ ferrite::barrier_signal({slot_ptr}, {count}); }}\n\
         kittens::group<1>::sync();"
    ))
}

/// `ferrite::barrier_wait(slot_ptr, expected);` — gmem cross-CTA
/// volatile spin-load from one thread until the counter reaches
/// `expected`, then `__threadfence()`. Gated to laneid 0;
/// intra-warp sync converges other lanes.
pub fn ferrite_barrier_wait(slot_ptr: &CuExpr, expected: u32) -> CuStmt {
    CuStmt::new(format!(
        "if (kittens::laneid() == 0) {{ ferrite::barrier_wait({slot_ptr}, {expected}); }}\n\
         kittens::group<1>::sync();"
    ))
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

// ============================================================
// Register-tile declarations + ops (Sprint 10 — Gemm).
// ============================================================

/// `kittens::rt_fl<rows, cols> <name>;` — declare local fp32
/// register tile (default row layout). Returned handle is
/// row-layout-typed.
///
/// Source: `include/types/register/rt.cuh:142` (rt_fl alias).
pub fn decl_rt_fl(name: &str, rows: u32, cols: u32) -> (CuStmt, Rt<F32, RtRow>) {
    let stmt = CuStmt::new(format!("kittens::rt_fl<{rows}, {cols}> {name};"));
    (stmt, Rt::from_expr(CuExpr::new(name.to_string()), rows, cols))
}

/// `kittens::rt_bf<rows, cols> <name>;` — declare local bf16
/// register tile in row layout (mma_AB A operand).
///
/// Source: `include/types/register/rt.cuh:143` (rt_bf alias).
pub fn decl_rt_bf_row(name: &str, rows: u32, cols: u32) -> (CuStmt, Rt<Bf16, RtRow>) {
    let stmt = CuStmt::new(format!("kittens::rt_bf<{rows}, {cols}> {name};"));
    (stmt, Rt::from_expr(CuExpr::new(name.to_string()), rows, cols))
}

/// `kittens::rt_bf<rows, cols, kittens::ducks::rt_layout::col>
/// <name>;` — declare local bf16 register tile in col layout
/// (mma_AB B operand).
///
/// Source: `include/types/register/rt.cuh:143` + `rt_layout.cuh`.
pub fn decl_rt_bf_col(name: &str, rows: u32, cols: u32) -> (CuStmt, Rt<Bf16, RtCol>) {
    let stmt = CuStmt::new(format!(
        "kittens::rt_bf<{rows}, {cols}, kittens::ducks::rt_layout::col> {name};"
    ));
    (stmt, Rt::from_expr(CuExpr::new(name.to_string()), rows, cols))
}

/// `kittens::warp::zero(rt);` — set every element of a register
/// tile to zero (canonical accumulator init).
///
/// Source: `include/ops/group/register/tile/maps.cuh:421-424`.
pub fn warp_zero_rt<T: super::handles::DtypeName + RtCudaName, L: RtLayoutTag>(
    rt: &Rt<T, L>,
) -> CuStmt {
    CuStmt::new(format!("kittens::warp::zero({});", rt.expr()))
}

/// `kittens::warp::load(rt, st);` — collaborative shared->register
/// load. With `kittens::warp == kittens::group<1>`, GROUP_WARPS is
/// 1, so the static_asserts in TK 2.0 collapse to
/// `ST::rows == RT::rows && ST::cols == RT::cols`. The RT layout
/// (row vs col) determines whether `ldsm4` or `ldsm4t` is emitted
/// internally — TK handles both.
///
/// Source: `include/ops/group/memory/tile/shared_to_register.cuh:14-128`.
pub fn warp_load_rt_from_st_bf<L: RtLayoutTag>(
    rt: &Rt<Bf16, L>,
    st: &St<Bf16>,
) -> CuStmt {
    debug_assert_eq!(
        rt.rows(),
        st.rows(),
        "warp_load_rt_from_st_bf: rt.rows ({}) must equal st.rows ({})",
        rt.rows(),
        st.rows()
    );
    debug_assert_eq!(
        rt.cols(),
        st.cols(),
        "warp_load_rt_from_st_bf: rt.cols ({}) must equal st.cols ({})",
        rt.cols(),
        st.cols()
    );
    CuStmt::new(format!(
        "kittens::warp::load({rt}, {st});",
        rt = rt.expr(),
        st = st.expr()
    ))
}

/// `kittens::warp::load(rt_fl, st_bf);` — collaborative
/// shared->register load with bf16->fp32 type conversion handled
/// internally by TK 2.0 (`base_types::convertor<T2, U2>` at
/// `shared_to_register.cuh:47-50,96-99,120-123`). Used by
/// `TkFusedGemmAdd` to bring the residual subtile into the fp32
/// accumulator as the mma C operand (`acc = A*B + residual`).
///
/// Source: `include/ops/group/memory/tile/shared_to_register.cuh:14-128`.
pub fn warp_load_rt_fl_from_st_bf(
    rt: &Rt<F32, RtRow>,
    st: &St<Bf16>,
) -> CuStmt {
    debug_assert_eq!(
        rt.rows(),
        st.rows(),
        "warp_load_rt_fl_from_st_bf: rt.rows ({}) must equal st.rows ({})",
        rt.rows(),
        st.rows()
    );
    debug_assert_eq!(
        rt.cols(),
        st.cols(),
        "warp_load_rt_fl_from_st_bf: rt.cols ({}) must equal st.cols ({})",
        rt.cols(),
        st.cols()
    );
    CuStmt::new(format!(
        "kittens::warp::load({rt}, {st});",
        rt = rt.expr(),
        st = st.expr()
    ))
}

/// `kittens::warp::store(st, rt);` — collaborative register->shared
/// store. Same shape constraints as
/// [`warp_load_rt_from_st_bf`]. Used to land the gemm fp32
/// accumulator (downcast to bf16 by TK's internal type-converter)
/// back into a shared `st_bf<M, TILE_N>` slice of out_smem.
///
/// Source: `include/ops/group/memory/tile/shared_to_register.cuh:138-244`.
pub fn warp_store_st_bf_from_rt_fl(
    st: &St<Bf16>,
    rt: &Rt<F32, RtRow>,
) -> CuStmt {
    debug_assert_eq!(
        rt.rows(),
        st.rows(),
        "warp_store_st_bf_from_rt_fl: rt.rows ({}) must equal st.rows ({})",
        rt.rows(),
        st.rows()
    );
    debug_assert_eq!(
        rt.cols(),
        st.cols(),
        "warp_store_st_bf_from_rt_fl: rt.cols ({}) must equal st.cols ({})",
        rt.cols(),
        st.cols()
    );
    CuStmt::new(format!(
        "kittens::warp::store({st}, {rt});",
        st = st.expr(),
        rt = rt.expr()
    ))
}

/// `kittens::warp::mma_AB(d, a, b, c);` — `D = A * B + C` with
/// D=fp32 row, A=bf16 row, B=bf16 col, C=fp32 row. Layouts and
/// shape compat are enforced by TK 2.0 `static_assert`s
/// (D::rows==A::rows, D::cols==B::cols, A::cols==B::rows,
/// D::rows==C::rows, D::cols==C::cols).
///
/// Source: `include/ops/group/mma/warp.cuh:583-632`.
#[allow(non_snake_case)]
pub fn warp_mma_AB(
    d: &Rt<F32, RtRow>,
    a: &Rt<Bf16, RtRow>,
    b: &Rt<Bf16, RtCol>,
    c: &Rt<F32, RtRow>,
) -> CuStmt {
    debug_assert_eq!(
        d.rows(),
        a.rows(),
        "warp_mma_AB: D.rows ({}) must equal A.rows ({})",
        d.rows(),
        a.rows()
    );
    debug_assert_eq!(
        d.cols(),
        b.cols(),
        "warp_mma_AB: D.cols ({}) must equal B.cols ({})",
        d.cols(),
        b.cols()
    );
    debug_assert_eq!(
        a.cols(),
        b.rows(),
        "warp_mma_AB: A.cols ({}) must equal B.rows ({})",
        a.cols(),
        b.rows()
    );
    debug_assert_eq!(
        d.rows(),
        c.rows(),
        "warp_mma_AB: D.rows ({}) must equal C.rows ({})",
        d.rows(),
        c.rows()
    );
    debug_assert_eq!(
        d.cols(),
        c.cols(),
        "warp_mma_AB: D.cols ({}) must equal C.cols ({})",
        d.cols(),
        c.cols()
    );
    CuStmt::new(format!(
        "kittens::warp::mma_AB({d}, {a}, {b}, {c});",
        d = d.expr(),
        a = a.expr(),
        b = b.expr(),
        c = c.expr()
    ))
}

/// `auto <name> = <parent>.template subtile<rows, cols>(int2{row, col});`
/// — declare a named local binding to a shared-tile soft-subtile
/// view (`st_subtile`). Returns the bound `St<Bf16>` handle.
///
/// `kittens::st<>::subtile<rows, cols>(int2)` is a non-const
/// member returning the subtile by VALUE. Binding via `auto`
/// materializes the temporary as a non-const lvalue — required
/// by `kittens::warp::store(ST &dst, const RT &src)` whose
/// `dst` is a non-const lvalue reference.
///
/// `row_idx_expr` / `col_idx_expr` are CUDA expressions in scope
/// (e.g. `"0"`, `"k_iter"`, `"static_cast<int>(kittens::warpid())"`).
///
/// Source: `include/types/shared/st.cuh:152-153,191-272`.
pub fn decl_st_bf_subtile(
    name: &str,
    parent: &St<Bf16>,
    rows: u32,
    cols: u32,
    row_idx_expr: &str,
    col_idx_expr: &str,
) -> (CuStmt, St<Bf16>) {
    let stmt = CuStmt::new(format!(
        "auto {name} = ({parent}).template subtile<{rows}, {cols}>(int2{{{row_idx_expr}, {col_idx_expr}}});",
        parent = parent.expr()
    ));
    let handle = St::from_expr(CuExpr::new(name.to_string()), rows, cols);
    (stmt, handle)
}

// ============================================================
// Tile-flavored non-tensor TMA (used by Gemm loader/storer to
// stage [M, K] activation, [CHUNK_K, N] b_tile, and [M, N]
// output between gmem and shared). Same C++ entry points as
// [`group_tma_load_async_raw`] / [`group_tma_store_async_raw`];
// only the typed dst/src handle differs.
// ============================================================

/// Tile-flavored `kittens::group<N>::tma::load_async(void* dst,
/// void* src, uint32_t bytes, sem& bar);` — stages a contiguous
/// `[rows, cols]` bf16 block from gmem into a shared tile.
///
/// Source: `include/ops/group/util/tma.cuh:72`.
pub fn group_tma_load_async_raw_st_bf(
    n: u32,
    dst: &St<Bf16>,
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

/// Tile-flavored `kittens::group<N>::tma::store_async(void* dst,
/// void* src, uint32_t bytes);` — stores a contiguous
/// `[rows, cols]` bf16 block from a shared tile to gmem.
///
/// Source: `include/ops/group/util/tma.cuh:82`.
pub fn group_tma_store_async_raw_st_bf(
    n: u32,
    dst: &GmemPtrRaw<Bf16>,
    src: &St<Bf16>,
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
