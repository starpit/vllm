// SPDX-License-Identifier: Apache-2.0
//! Rust API surface mirroring TK 2.0's C++ primitive surface.
//!
//! Each function below corresponds to exactly one TK 2.0 entry
//! point in `third_party/thunderkittens/include/`. The Rust
//! signature pins the typed handles (dtype + **const-generic
//! shape**) at codegen build time; the body emits the corresponding
//! CUDA call as a [`CuStmt`](super::cu::CuStmt) (or returns a typed
//! handle wrapping a [`CuExpr`](super::cu::CuExpr) for value-
//! producing ops like `decl_rv_fl`).
//!
//! Per [[feedback-end-to-end-compile-time-proofs]] (`MEGA_IR_PLAN.md`
//! §8.0b), every shape / dim / length the MegaIR encodes as a const
//! generic flows END-TO-END to the `tk20::*` call sites here as a
//! Rust const generic. Wrong-shape handle pass-through (e.g. an
//! `St<Bf16, 16, 64>` where an `St<Bf16, 16, 128>` is expected) is a
//! Rust type error at `cargo check -p ferrite-megakernel` — never a
//! runtime `debug_assert_eq!` panic, never a TK 2.0 nvcc
//! `static_assert` failure later in the pipeline.
//!
//! The group-scope `N` arg (i.e. `kittens::group<N>::*`) is also a
//! const generic. NCW is compile-time at the substrate layer
//! (`SubstrateBudget<_, NUM_CONSUMER_WARPS, _, _, _>`), so flowing
//! it as `<const N: u32>` here keeps the proof chain unbroken.
//!
//! "Calling" a TK 2.0 primitive from emit code is a Rust function
//! call with type-checked args; the emitted CUDA is the function's
//! body. Every function comments its TK 2.0 source citation. NEVER
//! add a function here without first opening the TK 2.0 header and
//! confirming the signature.

use super::cu::{CuExpr, CuStmt};
use super::handles::{
    Bf16, F32, GmemPtrRaw, Rt, RtCol, RtCudaName, RtLayoutTag, RtRow, Rv, RvCudaName,
    Semaphore, ScratchPtr, St, Sv,
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
pub fn group_wait<const N: u32>(sem: &Semaphore, phase: u32) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{N}>::wait({sem}, {phase});",
        sem = sem.expr()
    ))
}

/// `kittens::group<N>::arrive(sem);` — auto-laneid-gates internally.
///
/// Source: `include/ops/group/util/sync.cuh:69`
pub fn group_arrive<const N: u32>(sem: &Semaphore) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{N}>::arrive({sem});",
        sem = sem.expr()
    ))
}

/// `kittens::group<N>::sync(int id);` — named PTX bar.sync,
/// id ∈ 1..=15. `bar_id` is a runtime u32 because the IR's
/// `BarSyncId<ID>` is a 1..=15 sealed witness — the const-generic
/// proof lives upstream at the IR builder; here we just splice the
/// literal int the proc-macro extracted from the typed witness.
///
/// Source: `include/ops/group/group.cuh:33`
pub fn group_sync_named<const N: u32>(bar_id: u32) -> CuStmt {
    debug_assert!(
        (1..=15).contains(&bar_id),
        "group_sync_named: bar_id ({bar_id}) must be in 1..=15"
    );
    CuStmt::new(format!("kittens::group<{N}>::sync({bar_id});"))
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
pub fn group_tma_expect_bytes<const N: u32>(sem: &Semaphore, bytes: u32) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{N}>::tma::expect_bytes({sem}, {bytes});",
        sem = sem.expr()
    ))
}

/// `kittens::group<N>::tma::load_async(void* dst, void* src,
/// uint32_t size_bytes, semaphore& bar);` — auto-laneid-gates.
/// Casts the typed `Sv<Bf16, LEN>` dst and `GmemPtrRaw<Bf16>` src to
/// `void*` at the call site.
///
/// Source: `include/ops/group/util/tma.cuh:72` (group-scope
/// non-tensor variant; delegates to thread-scope at `:86` of
/// `include/ops/thread/util/tma.cuh`).
pub fn group_tma_load_async_raw<const N: u32, const LEN: u32>(
    dst: &Sv<Bf16, LEN>,
    src: &GmemPtrRaw<Bf16>,
    size_bytes: u32,
    sem: &Semaphore,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{N}>::tma::load_async(\
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
pub fn group_tma_store_async_raw<const N: u32, const LEN: u32>(
    dst: &GmemPtrRaw<Bf16>,
    src: &Sv<Bf16, LEN>,
    size_bytes: u32,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{N}>::tma::store_async(\
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
pub fn group_tma_store_async_wait<const N: u32>() -> CuStmt {
    CuStmt::new(format!("kittens::group<{N}>::tma::store_async_wait();"))
}

// ============================================================
// Group-scope shared <-> register transfer.
//
// kittens::group<NCW>::load(rv, sv) and store(sv, rv) auto-slice
// the shared vec into per-warp subvecs when NCW > 1. The
// caller's responsibility: ensure `SV_LEN == RV_LEN * NCW`.
//
// Stable Rust can't express `SV_LEN = RV_LEN * NCW` as a `where`
// clause without `feature(generic_const_exprs)`. The relation is
// proved upstream: the IR's MegaTape const generics
// (`SubstrateBudget::NUM_CONSUMER_WARPS`, the per-op
// `HiddenDim<HD>`, etc.) flow through to roles.rs and produce
// matching `SV_LEN` / `RV_LEN` / `NCW` literals at the call site.
// A wrong relation here fails at TK 2.0 nvcc time via TK's
// internal `static_assert(SV::length == RV::length * GROUP_WARPS)`.
// ============================================================

/// `kittens::group<NCW>::load(rv, sv);` — when NCW == 1 a direct
/// load; when NCW > 1 the SV is auto-sliced into
/// `subvec<RV::length>(warpid())` per-warp.
///
/// Source: `include/ops/group/memory/vec/shared_to_register.cuh:14`
/// (the `else` branch at line 84 does the auto-subvec).
pub fn group_load_sv_to_rv_bf16_to_f32<
    const NCW: u32,
    const RV_LEN: u32,
    const SV_LEN: u32,
>(
    rv: &Rv<F32, RV_LEN>,
    sv: &Sv<Bf16, SV_LEN>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{NCW}>::load({rv}, {sv});",
        rv = rv.expr(),
        sv = sv.expr()
    ))
}

/// `kittens::group<NCW>::store(sv, rv);` — symmetric to load.
///
/// Source: `include/ops/group/memory/vec/shared_to_register.cuh:101`
pub fn group_store_rv_to_sv_f32_to_bf16<
    const NCW: u32,
    const RV_LEN: u32,
    const SV_LEN: u32,
>(
    sv: &Sv<Bf16, SV_LEN>,
    rv: &Rv<F32, RV_LEN>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{NCW}>::store({sv}, {rv});",
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
pub fn warp_copy_rv<T: super::handles::DtypeName + RvCudaName, const LEN: u32>(
    dst: &Rv<T, LEN>,
    src: &Rv<T, LEN>,
) -> CuStmt {
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
pub fn warp_mul_rv_rv<const LEN: u32>(
    dst: &Rv<F32, LEN>,
    lhs: &Rv<F32, LEN>,
    rhs: &Rv<F32, LEN>,
) -> CuStmt {
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
pub fn warp_add_rv_rv<const LEN: u32>(
    dst: &Rv<F32, LEN>,
    lhs: &Rv<F32, LEN>,
    rhs: &Rv<F32, LEN>,
) -> CuStmt {
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
pub fn warp_add_rv_scalar_f32<const LEN: u32>(
    dst: &Rv<F32, LEN>,
    src: &Rv<F32, LEN>,
    scalar: &CuExpr,
) -> CuStmt {
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
pub fn warp_mul_rv_scalar_f32<const LEN: u32>(
    dst: &Rv<F32, LEN>,
    src: &Rv<F32, LEN>,
    scalar: &CuExpr,
) -> CuStmt {
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
pub fn warp_sum_to_scalar_f32<const LEN: u32>(
    scalar_out: &CuExpr,
    src: &Rv<F32, LEN>,
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
pub fn cross_warp_reduce_sum_f32<const NCW: u32>(
    scalar_out_name: &str,
    partial_in_name: &str,
    scratch: &ScratchPtr<F32>,
    bar_id: u32,
) -> CuStmt {
    debug_assert!(
        (1..=15).contains(&bar_id),
        "cross_warp_reduce_sum_f32: bar_id ({bar_id}) must be in 1..=15"
    );
    CuStmt::new(format!(
        "if (kittens::laneid() == 0) {{ {scratch}[kittens::warpid()] = {partial}; }}\n\
         kittens::group<{NCW}>::sync({bar_id});\n\
         #pragma unroll\n\
         for (int __cw_i = 0; __cw_i < {NCW}; ++__cw_i) {{ {out} += {scratch}[__cw_i]; }}",
        scratch = scratch.expr(),
        partial = partial_in_name,
        out = scalar_out_name
    ))
}

/// `const float <name> = rsqrtf(<full_sum_name> / <hidden_dim>.0f
/// + <eps>f);` — declare and bind the canonical RMS scale local.
/// Wraps a CUDA math intrinsic + arithmetic; not a TK 2.0 primitive
/// per se but the standard expression used in every RmsNorm-flavor
/// op body. `HIDDEN_DIM` is const-generic (the IR's `HiddenDim<HD>`
/// proof flowing through end-to-end).
pub fn decl_rms_scale_local<const HIDDEN_DIM: u32>(
    name: &str,
    full_sum_name: &str,
    eps: f32,
) -> (CuStmt, CuExpr) {
    let stmt = CuStmt::new(format!(
        "const float {name} = rsqrtf({sum} / {HIDDEN_DIM}.0f + {eps:e}f);",
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
pub fn embed_per_token_gather<const HIDDEN_DIM: u32, const NUM_TOKENS: u32>(
    out_byte_ptr: &CuExpr,
    embed_table_ptr: &GmemPtrRaw<Bf16>,
    input_ids_ptr: &CuExpr,
    page_ready: &Semaphore,
) -> CuStmt {
    let row_bytes = HIDDEN_DIM * 2;
    let total_bytes = row_bytes * NUM_TOKENS;
    CuStmt::new(format!(
        "kittens::group<1>::tma::expect_bytes({sem}, {total_bytes});\n\
         for (int __embed_t = 0; __embed_t < {NUM_TOKENS}; ++__embed_t) {{\n\
         \x20   const uint32_t __embed_row = {ids}[__embed_t];\n\
         \x20   void* __embed_dst = static_cast<void*>(\
         reinterpret_cast<char*>({dst}) + __embed_t * {row_bytes});\n\
         \x20   void* __embed_src = static_cast<void*>(\
         {table} + static_cast<size_t>(__embed_row) * {HIDDEN_DIM});\n\
         \x20   kittens::group<1>::tma::load_async(__embed_dst, __embed_src, {row_bytes}, {sem});\n\
         }}",
        sem = page_ready.expr(),
        ids = input_ids_ptr,
        dst = out_byte_ptr,
        table = embed_table_ptr.expr(),
    ))
}

/// Per-token cos/sin gather for FusedQkvRopeCache. Emits a
/// `kittens::group<1>::tma::expect_bytes` for the total cos_sin
/// staging volume, then a per-token loop that issues one
/// `kittens::group<1>::tma::load_async` per row. Row index is
/// `positions[t]` (FQRC's rotary indirection), row size is
/// `HEAD_DIM * sizeof(bf16)` bytes (one packed cos/sin row), and
/// rows land contiguously in shared at `dst + t * row_bytes`.
///
/// Mirror of [`embed_per_token_gather`] with the row index source
/// swapped from `input_ids` → `positions` and the row stride
/// swapped from `HIDDEN_DIM` → `HEAD_DIM`. The cos_sin gmem table
/// is shape `[max_pos, HEAD_DIM]` (packed `[cos[0..hd/2],
/// sin[0..hd/2]]` per row) — same convention as
/// `fused_qkv_rope_cache_bf16` in `crates/ferrite-kernels/src/
/// kernels.rs:681`.
///
/// Source: `include/ops/group/util/tma.cuh:18` (expect_bytes) +
/// `include/ops/group/util/tma.cuh:72` (load_async, raw).
pub fn cos_sin_per_token_gather<const HEAD_DIM: u32, const NUM_TOKENS: u32>(
    out_byte_ptr: &CuExpr,
    cos_sin_table_ptr: &GmemPtrRaw<Bf16>,
    positions_ptr: &CuExpr,
    page_ready: &Semaphore,
) -> CuStmt {
    let row_bytes = HEAD_DIM * 2;
    let total_bytes = row_bytes * NUM_TOKENS;
    CuStmt::new(format!(
        "kittens::group<1>::tma::expect_bytes({sem}, {total_bytes});\n\
         for (int __cs_t = 0; __cs_t < {NUM_TOKENS}; ++__cs_t) {{\n\
         \x20   const uint32_t __cs_row = {pos}[__cs_t];\n\
         \x20   void* __cs_dst = static_cast<void*>(\
         reinterpret_cast<char*>({dst}) + __cs_t * {row_bytes});\n\
         \x20   void* __cs_src = static_cast<void*>(\
         {table} + static_cast<size_t>(__cs_row) * {HEAD_DIM});\n\
         \x20   kittens::group<1>::tma::load_async(__cs_dst, __cs_src, {row_bytes}, {sem});\n\
         }}",
        sem = page_ready.expr(),
        pos = positions_ptr,
        dst = out_byte_ptr,
        table = cos_sin_table_ptr.expr(),
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
pub fn per_token_tma_store<const HIDDEN_DIM: u32, const NUM_TOKENS: u32>(
    out_gmem: &GmemPtrRaw<Bf16>,
    src_byte_ptr: &CuExpr,
) -> CuStmt {
    let row_bytes = HIDDEN_DIM * 2;
    CuStmt::new(format!(
        "for (int __pt_t = 0; __pt_t < {NUM_TOKENS}; ++__pt_t) {{\n\
         \x20   void* __pt_dst = static_cast<void*>(\
         {gmem} + static_cast<size_t>(__pt_t) * {HIDDEN_DIM});\n\
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
pub fn warp_apply_f32_lambda<const LEN: u32>(
    dst: &Rv<F32, LEN>,
    src: &Rv<F32, LEN>,
    lambda_body: &str,
) -> CuStmt {
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
pub fn decl_rv_fl<const LEN: u32>(name: &str) -> (CuStmt, Rv<F32, LEN>) {
    let stmt = CuStmt::new(format!("kittens::rv_fl<{LEN}> {name};"));
    (stmt, Rv::<F32, LEN>::from_expr(CuExpr::new(name.to_string())))
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
pub fn decl_rt_fl<const ROWS: u32, const COLS: u32>(
    name: &str,
) -> (CuStmt, Rt<F32, RtRow, ROWS, COLS>) {
    let stmt = CuStmt::new(format!("kittens::rt_fl<{ROWS}, {COLS}> {name};"));
    (
        stmt,
        Rt::<F32, RtRow, ROWS, COLS>::from_expr(CuExpr::new(name.to_string())),
    )
}

/// `kittens::rt_bf<rows, cols> <name>;` — declare local bf16
/// register tile in row layout (mma_AB A operand).
///
/// Source: `include/types/register/rt.cuh:143` (rt_bf alias).
pub fn decl_rt_bf_row<const ROWS: u32, const COLS: u32>(
    name: &str,
) -> (CuStmt, Rt<Bf16, RtRow, ROWS, COLS>) {
    let stmt = CuStmt::new(format!("kittens::rt_bf<{ROWS}, {COLS}> {name};"));
    (
        stmt,
        Rt::<Bf16, RtRow, ROWS, COLS>::from_expr(CuExpr::new(name.to_string())),
    )
}

/// `kittens::rt_bf<rows, cols, kittens::ducks::rt_layout::col>
/// <name>;` — declare local bf16 register tile in col layout
/// (mma_AB B operand).
///
/// Source: `include/types/register/rt.cuh:143` + `rt_layout.cuh`.
pub fn decl_rt_bf_col<const ROWS: u32, const COLS: u32>(
    name: &str,
) -> (CuStmt, Rt<Bf16, RtCol, ROWS, COLS>) {
    let stmt = CuStmt::new(format!(
        "kittens::rt_bf<{ROWS}, {COLS}, kittens::ducks::rt_layout::col> {name};"
    ));
    (
        stmt,
        Rt::<Bf16, RtCol, ROWS, COLS>::from_expr(CuExpr::new(name.to_string())),
    )
}

/// `kittens::warp::zero(rt);` — set every element of a register
/// tile to zero (canonical accumulator init).
///
/// Source: `include/ops/group/register/tile/maps.cuh:421-424`.
pub fn warp_zero_rt<
    T: super::handles::DtypeName + RtCudaName,
    L: RtLayoutTag,
    const ROWS: u32,
    const COLS: u32,
>(
    rt: &Rt<T, L, ROWS, COLS>,
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
pub fn warp_load_rt_from_st_bf<L: RtLayoutTag, const ROWS: u32, const COLS: u32>(
    rt: &Rt<Bf16, L, ROWS, COLS>,
    st: &St<Bf16, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::load({rt}, {st});",
        rt = rt.expr(),
        st = st.expr()
    ))
}

/// `kittens::warp::apply(rt_dst, rt_src, lambda);` — apply a
/// per-element lambda over a register tile. The lambda
/// signature is `(int row, int col, T x) -> T` (3-arg, vs the
/// 2-arg `(idx, x)` of the vector-flavored apply at
/// `vec/maps.cuh:79-112`). Used for tile-level activations
/// (silu / gelu) in `FusedGateUpActivateMul`.
///
/// Source: `include/ops/group/register/tile/maps.cuh:89-115`.
pub fn warp_apply_f32_rt_lambda<const ROWS: u32, const COLS: u32>(
    dst: &Rt<F32, RtRow, ROWS, COLS>,
    src: &Rt<F32, RtRow, ROWS, COLS>,
    lambda_body: &str,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::apply({dst}, {src}, [=] __device__ (int /*row*/, int col, float x) {{ return {body}; }});",
        dst = dst.expr(),
        src = src.expr(),
        body = lambda_body
    ))
}

/// `kittens::warp::mul(dst, lhs, rhs);` — elementwise tile
/// multiplication. Both operands and dst share the same
/// dtype/layout (TK 2.0's `bin_map<base_ops::mul, T>`).
///
/// Source: `include/ops/group/register/tile/maps.cuh:707-710`.
pub fn warp_mul_rt_rt<const ROWS: u32, const COLS: u32>(
    dst: &Rt<F32, RtRow, ROWS, COLS>,
    lhs: &Rt<F32, RtRow, ROWS, COLS>,
    rhs: &Rt<F32, RtRow, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::mul({dst}, {lhs}, {rhs});",
        dst = dst.expr(),
        lhs = lhs.expr(),
        rhs = rhs.expr()
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
pub fn warp_load_rt_fl_from_st_bf<const ROWS: u32, const COLS: u32>(
    rt: &Rt<F32, RtRow, ROWS, COLS>,
    st: &St<Bf16, ROWS, COLS>,
) -> CuStmt {
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
/// back into a shared `st_bf<ROWS, COLS>` slice of out_smem.
///
/// Source: `include/ops/group/memory/tile/shared_to_register.cuh:138-244`.
pub fn warp_store_st_bf_from_rt_fl<const ROWS: u32, const COLS: u32>(
    st: &St<Bf16, ROWS, COLS>,
    rt: &Rt<F32, RtRow, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::store({st}, {rt});",
        st = st.expr(),
        rt = rt.expr()
    ))
}

/// `kittens::warp::mma_AB(d, a, b, c);` — `D = A * B + C` with
/// D=fp32 row, A=bf16 row, B=bf16 col, C=fp32 row. Layouts and
/// shape compat are enforced by Rust const generics:
///   D: [M, N], A: [M, K], B: [K, N], C: [M, N].
/// TK 2.0 `static_assert`s catch the same constraints later as a
/// belt-and-braces check.
///
/// Source: `include/ops/group/mma/warp.cuh:583-632`.
#[allow(non_snake_case)]
pub fn warp_mma_AB<const M: u32, const K: u32, const N: u32>(
    d: &Rt<F32, RtRow, M, N>,
    a: &Rt<Bf16, RtRow, M, K>,
    b: &Rt<Bf16, RtCol, K, N>,
    c: &Rt<F32, RtRow, M, N>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::mma_AB({d}, {a}, {b}, {c});",
        d = d.expr(),
        a = a.expr(),
        b = b.expr(),
        c = c.expr()
    ))
}

/// `kittens::warp::mma_ABt(d, a, b, c);` — `D = A * B^T + C` with
/// D=fp32 row, A=bf16 row, B=bf16 row (note: row, not col — the
/// transpose is the operation, not the storage layout), C=fp32 row.
/// Used for FlashAttention's QK^T pass where Q is loaded into a
/// row-layout register tile and K's per-block tile is similarly
/// row-layout (the transpose folds into the WMMA instruction itself,
/// not into the tile's data layout).
///
/// Const-generic shape contract:
///   D: [M, N]   row,  fp32   (`Rt<F32, RtRow, M, N>`)
///   A: [M, K]   row,  bf16   (`Rt<Bf16, RtRow, M, K>`)
///   B: [N, K]   row,  bf16   (`Rt<Bf16, RtRow, N, K>`)
///   C: [M, N]   row,  fp32   (`Rt<F32, RtRow, M, N>`)
///
/// TK 2.0's `static_assert`s in `mma_ABt` enforce the same
/// invariants (`D::rows == A::rows && D::cols == B::rows`,
/// `A::cols == B::cols`); the Rust const generics catch a wrong
/// shape at `cargo check -p ferrite-megakernel` rather than at
/// nvcc time.
///
/// Source: `include/ops/group/mma/warp.cuh:647-696`.
#[allow(non_snake_case)]
pub fn warp_mma_ABt<const M: u32, const K: u32, const N: u32>(
    d: &Rt<F32, RtRow, M, N>,
    a: &Rt<Bf16, RtRow, M, K>,
    b: &Rt<Bf16, RtRow, N, K>,
    c: &Rt<F32, RtRow, M, N>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::mma_ABt({d}, {a}, {b}, {c});",
        d = d.expr(),
        a = a.expr(),
        b = b.expr(),
        c = c.expr()
    ))
}

/// `auto <name> = <parent>.template subtile<rows, cols>(int2{row, col});`
/// — declare a named local binding to a shared-tile soft-subtile
/// view (`st_subtile`). Returns the bound `St<Bf16, ROWS, COLS>`
/// handle.
///
/// `kittens::st<>::subtile<rows, cols>(int2)` is a non-const
/// member returning the subtile by VALUE. Binding via `auto`
/// materializes the temporary as a non-const lvalue — required
/// by `kittens::warp::store(ST &dst, const RT &src)` whose
/// `dst` is a non-const lvalue reference.
///
/// `row_idx_expr` / `col_idx_expr` are CUDA expressions in scope
/// (e.g. `"0"`, `"k_iter"`, `"static_cast<int>(kittens::warpid())"`).
/// The PARENT_ROWS / PARENT_COLS const generics on the parent are
/// existential here — any parent shape is valid as long as the
/// requested subtile fits, which TK 2.0 catches via its own
/// `static_assert`.
///
/// Source: `include/types/shared/st.cuh:152-153,191-272`.
pub fn decl_st_bf_subtile<
    const PARENT_ROWS: u32,
    const PARENT_COLS: u32,
    const ROWS: u32,
    const COLS: u32,
>(
    name: &str,
    parent: &St<Bf16, PARENT_ROWS, PARENT_COLS>,
    row_idx_expr: &str,
    col_idx_expr: &str,
) -> (CuStmt, St<Bf16, ROWS, COLS>) {
    let stmt = CuStmt::new(format!(
        "auto {name} = ({parent}).template subtile<{ROWS}, {COLS}>(int2{{{row_idx_expr}, {col_idx_expr}}});",
        parent = parent.expr()
    ));
    let handle = St::<Bf16, ROWS, COLS>::from_expr(CuExpr::new(name.to_string()));
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
/// `[ROWS, COLS]` bf16 block from gmem into a shared tile.
///
/// Source: `include/ops/group/util/tma.cuh:72`.
pub fn group_tma_load_async_raw_st_bf<
    const N: u32,
    const ROWS: u32,
    const COLS: u32,
>(
    dst: &St<Bf16, ROWS, COLS>,
    src: &GmemPtrRaw<Bf16>,
    size_bytes: u32,
    sem: &Semaphore,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{N}>::tma::load_async(\
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
/// `[ROWS, COLS]` bf16 block from a shared tile to gmem.
///
/// Source: `include/ops/group/util/tma.cuh:82`.
pub fn group_tma_store_async_raw_st_bf<
    const N: u32,
    const ROWS: u32,
    const COLS: u32,
>(
    dst: &GmemPtrRaw<Bf16>,
    src: &St<Bf16, ROWS, COLS>,
    size_bytes: u32,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::group<{N}>::tma::store_async(\
         reinterpret_cast<void*>({dst}), \
         reinterpret_cast<void*>(&{src}), \
         {size_bytes});",
        dst = dst.expr(),
        src = src.expr()
    ))
}

// ============================================================
// FlashAttention row-reduction + broadcast primitives.
//
// These cover the "Q @ K^T → online softmax → @ V" pattern of the
// `AttentionViaCache` (S16) emit. Each call cited to its TK 2.0
// header (`include/ops/group/register/tile/{maps,reductions}.cuh`).
//
// All operate at warp scope (`kittens::warp == kittens::group<1>`);
// the row vector is stored in fp32 and the tile is fp32 (same as
// FlashAttention's standard accumulator dtype).
// ============================================================

/// `kittens::warp::row_max(rv_dst, rt_src);` — per-row max reduce
/// of a tile into a row vector. Initial pass (no running accum).
///
/// Source: `include/ops/group/register/tile/reductions.cuh:253`.
pub fn warp_row_max_init<const ROWS: u32, const COLS: u32>(
    rv_dst: &Rv<F32, ROWS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::row_max({rv}, {rt});",
        rv = rv_dst.expr(),
        rt = rt_src.expr()
    ))
}

/// `kittens::warp::row_max(rv_dst, rt_src, rv_src_accum);` — per-row
/// max reduce with a running accumulator. Used in the FlashAttention
/// online-softmax loop to track `max(max_prev, row_max(scores))`
/// across KV blocks.
///
/// Source: `include/ops/group/register/tile/reductions.cuh:303`.
pub fn warp_row_max_running<const ROWS: u32, const COLS: u32>(
    rv_dst: &Rv<F32, ROWS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
    rv_src_accum: &Rv<F32, ROWS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::row_max({rv}, {rt}, {acc});",
        rv = rv_dst.expr(),
        rt = rt_src.expr(),
        acc = rv_src_accum.expr()
    ))
}

/// `kittens::warp::row_sum(rv_dst, rt_src);` — per-row sum reduce.
/// Initial pass.
///
/// Source: `include/ops/group/register/tile/reductions.cuh:277`.
pub fn warp_row_sum_init<const ROWS: u32, const COLS: u32>(
    rv_dst: &Rv<F32, ROWS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::row_sum({rv}, {rt});",
        rv = rv_dst.expr(),
        rt = rt_src.expr()
    ))
}

/// `kittens::warp::row_sum(rv_dst, rt_src, rv_src_accum);` — per-row
/// sum reduce with running accumulator. Used in FlashAttention to
/// track `sum_prev * scale + row_sum(exp(scores))`.
///
/// Source: `include/ops/group/register/tile/reductions.cuh:329`.
pub fn warp_row_sum_running<const ROWS: u32, const COLS: u32>(
    rv_dst: &Rv<F32, ROWS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
    rv_src_accum: &Rv<F32, ROWS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::row_sum({rv}, {rt}, {acc});",
        rv = rv_dst.expr(),
        rt = rt_src.expr(),
        acc = rv_src_accum.expr()
    ))
}

/// `kittens::warp::exp(rt_dst, rt_src);` — elementwise exponential
/// over a fp32 register tile. Used to emit `exp(scores - max)` in
/// the online-softmax body.
///
/// Source: `include/ops/group/register/tile/maps.cuh:464`.
pub fn warp_exp_rt<const ROWS: u32, const COLS: u32>(
    rt_dst: &Rt<F32, RtRow, ROWS, COLS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::exp({dst}, {src});",
        dst = rt_dst.expr(),
        src = rt_src.expr()
    ))
}

/// `kittens::warp::exp(rv_dst, rv_src);` — elementwise exponential
/// over a fp32 register vector. Used to compute the per-row scale
/// `exp(max_prev - max_curr)` for FlashAttention rescale.
///
/// Source: `include/ops/group/register/vec/maps.cuh` (vec-flavored
/// `unary_map<base_ops::exp>` analogous to the tile variant).
pub fn warp_exp_rv<const LEN: u32>(rv_dst: &Rv<F32, LEN>, rv_src: &Rv<F32, LEN>) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::exp({dst}, {src});",
        dst = rv_dst.expr(),
        src = rv_src.expr()
    ))
}

/// `kittens::warp::sub(rv_dst, rv_lhs, rv_rhs);` — elementwise
/// subtract over fp32 register vectors. Used to compute
/// `max_prev - max_curr` for the rescale exponent.
///
/// Source: `include/ops/group/register/vec/maps.cuh` (vec-flavored
/// `bin_map<base_ops::sub>`).
pub fn warp_sub_rv_rv<const LEN: u32>(
    rv_dst: &Rv<F32, LEN>,
    rv_lhs: &Rv<F32, LEN>,
    rv_rhs: &Rv<F32, LEN>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::sub({dst}, {lhs}, {rhs});",
        dst = rv_dst.expr(),
        lhs = rv_lhs.expr(),
        rhs = rv_rhs.expr()
    ))
}

/// `kittens::warp::sub_row(rt_dst, rt_src, rv_row_values);` —
/// subtract a per-row column vector from each row of a tile.
/// Used to emit `scores - max_curr` (broadcast) in the
/// online-softmax body.
///
/// Source: `include/ops/group/register/tile/maps.cuh:758`.
pub fn warp_sub_row<const ROWS: u32, const COLS: u32>(
    rt_dst: &Rt<F32, RtRow, ROWS, COLS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
    rv_row_values: &Rv<F32, ROWS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::sub_row({dst}, {src}, {rv});",
        dst = rt_dst.expr(),
        src = rt_src.expr(),
        rv = rv_row_values.expr()
    ))
}

/// `kittens::warp::mul_row(rt_dst, rt_src, rv_row_values);` —
/// multiply each row of a tile by a per-row scalar from a column
/// vector. Used to emit `acc *= scale` (per-row rescale) in the
/// online-softmax body.
///
/// Source: `include/ops/group/register/tile/maps.cuh` (mirror of
/// `add_row` / `sub_row` at :736-784).
pub fn warp_mul_row<const ROWS: u32, const COLS: u32>(
    rt_dst: &Rt<F32, RtRow, ROWS, COLS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
    rv_row_values: &Rv<F32, ROWS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::mul_row({dst}, {src}, {rv});",
        dst = rt_dst.expr(),
        src = rt_src.expr(),
        rv = rv_row_values.expr()
    ))
}

/// `kittens::rv_fl<LEN> name; kittens::warp::zero(name);
/// kittens::warp::add(name, name, scalar);` — declare an fp32 row
/// vector with each lane initialized to the same scalar (e.g.
/// `-INFINITY` for the running-max accumulator before the first KV
/// block).
///
/// The natural TK 2.0 idiom is `kittens::rv_fl<LEN> name; warp::fill(
/// name, scalar)`, but `fill` for rv isn't currently bound. This
/// helper emits a literal-broadcast init via `kittens::warp::add(rv,
/// zeros_rv, scalar)` — equivalent to `dst[i] = 0 + scalar` per
/// lane. Use when the rv is a running accumulator that needs a
/// non-zero initial value.
pub fn decl_rv_fl_init_scalar<const LEN: u32>(
    name: &str,
    init_scalar: &str,
) -> (CuStmt, Rv<F32, LEN>) {
    let stmt = CuStmt::new(format!(
        "kittens::rv_fl<{LEN}> {name};\n\
         kittens::warp::zero({name});\n\
         kittens::warp::add({name}, {name}, {init_scalar});"
    ));
    (stmt, Rv::<F32, LEN>::from_expr(CuExpr::new(name.to_string())))
}

/// `kittens::warp::zero(rv);` — zero a register vector. Used to
/// init the per-row sum accumulator at the start of FlashAttention.
///
/// Source: `include/ops/group/register/vec/maps.cuh` (vec-flavored
/// `kittens::warp::zero`).
pub fn warp_zero_rv<const LEN: u32>(rv: &Rv<F32, LEN>) -> CuStmt {
    CuStmt::new(format!("kittens::warp::zero({});", rv.expr()))
}

// ============================================================
// Runtime control-flow helpers.
//
// Per [[feedback-dogfood-tk20-rust]] runtime control flow
// (`for (...) { ... }`, `if (...) { ... } else { ... }`) inside
// role bodies must go through these helpers — no inline `format!()`
// in roles.rs. Both helpers compose the named CuBlock body into
// the surrounding CuStmt without escaping it back to a raw string.
// ============================================================

/// Emit `for (<header>) { <body> }`. `header` is the full
/// for-loop header (e.g. `"int h = 0; h < NUM_HEADS; ++h"`); the
/// body is the supplied `CuBlock` (rendered with 4-space indent).
/// Used by FQRC's per-head loop + any role-body runtime iteration
/// that's not a tk20 primitive.
pub fn for_loop(header: &str, body: &super::cu::CuBlock) -> CuStmt {
    CuStmt::new(format!("for ({header}) {{\n{}}}", body.render(4)))
}

/// Emit `if (<cond>) { <then_block> } else { <else_block> }`. Used
/// by FQRC's per-head Q/K/V routing (`if (col < q_off + qkv_q) {
/// ... } else if (col < q_off + qkv_q + qkv_k) { ... } else { ...
/// }`). The else block is optional — pass `None` for `if (...) {
/// ... }`.
pub fn if_else(
    cond: &str,
    then_block: &super::cu::CuBlock,
    else_block: Option<&super::cu::CuBlock>,
) -> CuStmt {
    match else_block {
        None => CuStmt::new(format!("if ({cond}) {{\n{}}}", then_block.render(4))),
        Some(eb) => CuStmt::new(format!(
            "if ({cond}) {{\n{}}} else {{\n{}}}",
            then_block.render(4),
            eb.render(4)
        )),
    }
}

/// Emit an `if (c0) { b0 } else if (c1) { b1 } ... else { eb }`
/// chain. Used when a per-iter loop body routes to one of N
/// branches by a runtime predicate (e.g. FQRC's Q/K/V routing).
/// The nested-`if_else` form is logically equivalent but emits
/// `else { if ... }` which differs textually from the chain form
/// readers expect.
pub fn if_chain(
    branches: &[(&str, &super::cu::CuBlock)],
    else_block: Option<&super::cu::CuBlock>,
) -> CuStmt {
    debug_assert!(!branches.is_empty(), "if_chain needs at least one branch");
    let mut out = String::new();
    for (i, (cond, block)) in branches.iter().enumerate() {
        if i == 0 {
            out.push_str(&format!("if ({cond}) {{\n{}}}", block.render(4)));
        } else {
            out.push_str(&format!(" else if ({cond}) {{\n{}}}", block.render(4)));
        }
    }
    if let Some(eb) = else_block {
        out.push_str(&format!(" else {{\n{}}}", eb.render(4)));
    }
    CuStmt::new(out)
}

// ============================================================
// FlashAttention-2 body primitives. Used by
// `render_attention_via_cache` to compose the per-block-iter
// loop. Each binding cited to its TK 2.0 source line.
// ============================================================

/// Emit `__shared__ kittens::semaphore <name>;` — a single
/// block-scope semaphore used to handshake a TMA load from the
/// issuing thread to the waiting consumer warps. Must be
/// initialized via [`init_semaphore_lane0`] before first use.
///
/// Source: `include/types/semaphore.cuh:23` (kittens::semaphore).
pub fn decl_shared_semaphore(name: &str) -> CuStmt {
    CuStmt::new(format!("__shared__ kittens::semaphore {name};"))
}

/// Initialize a block-scope semaphore once across the consumer
/// warpgroup: only consumer-warp 0 lane 0 calls
/// `init_semaphore`, and a `kittens::group<NCW>::sync(bar_id)`
/// follows so all consumer warps see the init before they
/// `wait()` on the semaphore. `expected_arrives` is the count
/// the semaphore arrives to before `wait` returns (1 for a
/// single TMA load).
///
/// The double gate (`warpid() == 0 && laneid() == 0`) prevents
/// the init from being called once per consumer warp (which
/// would clobber the semaphore state).
///
/// `__syncthreads()` is NOT used because the consumer body runs
/// inside `if (wid < NUM_CONSUMER_WARPS)`; loader/storer warps
/// are in the `else` branch and would not reach a block-wide
/// barrier, yielding undefined behavior. Group<NCW>::sync uses
/// a named PTX bar.sync that only requires the consumer warps
/// to participate.
///
/// Source: `include/ops/group/util/sync.cuh:53` (init_semaphore).
pub fn init_semaphore_warp0<const NCW: u32>(
    name: &str,
    expected_arrives: u32,
    bar_id: u32,
) -> CuStmt {
    CuStmt::new(format!(
        "if (kittens::warpid() == 0 && kittens::laneid() == 0) {{ \
         kittens::init_semaphore({name}, 0, {expected_arrives}); \
         }} kittens::group<{NCW}>::sync({bar_id});"
    ))
}

/// `kittens::tma::expect_bytes(sem, bytes);` — warp-scope (group<1>)
/// expect_bytes call. Issuer-thread gating happens inside the TK
/// helper (`include/ops/group/util/tma.cuh:18-22` — `if (laneid()
/// == 0)`).
pub fn warp_tma_expect_bytes(sem: &Semaphore, bytes_expr: &str) -> CuStmt {
    CuStmt::new(format!(
        "kittens::tma::expect_bytes({sem}, {bytes});",
        sem = sem.expr(),
        bytes = bytes_expr,
    ))
}

/// `kittens::tma::load_async(dst, src, bytes, sem);` — warp-scope
/// non-tensor TMA load with a runtime-computed source pointer
/// (used for paged-KV gather where the source is
/// `kv_cache_ptrs[layer] + block_table[p] * KV_BLOCK_BYTES`).
/// Issuer-thread gating is inside the TK helper.
///
/// Source: `include/ops/group/util/tma.cuh:72-76`.
pub fn warp_tma_load_async_raw_st_bf<const ROWS: u32, const COLS: u32>(
    dst: &St<Bf16, ROWS, COLS>,
    src_ptr_expr: &str,
    bytes_expr: &str,
    sem: &Semaphore,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::tma::load_async(\
         reinterpret_cast<void*>(&{dst}), \
         reinterpret_cast<void*>({src}), \
         {bytes}, {sem});",
        dst = dst.expr(),
        src = src_ptr_expr,
        bytes = bytes_expr,
        sem = sem.expr(),
    ))
}

/// `kittens::wait(sem, phase);` — single-warp wait on a semaphore.
/// Each consumer warp polls independently; all unblock once the
/// TMA hardware arrives on the semaphore. `phase` toggles 0↔1
/// per arrival across iterations, so the per-block-iter loop
/// passes `(p & 1)` for the phase.
///
/// Source: `include/ops/group/util/sync.cuh:112` (group<1>::wait).
pub fn warp_wait_sem(sem: &Semaphore, phase_expr: &str) -> CuStmt {
    CuStmt::new(format!(
        "kittens::wait({sem}, {phase});",
        sem = sem.expr(),
        phase = phase_expr,
    ))
}

/// `kittens::warp::mul(rt_dst, rt_src, scalar);` — multiply each
/// element of a register tile by a runtime scalar. Used for
/// `att_block *= attn_scale` in FlashAttention.
///
/// Source: `include/ops/group/register/tile/maps.cuh:728-731`.
pub fn warp_mul_rt_scalar<const ROWS: u32, const COLS: u32>(
    dst: &Rt<F32, RtRow, ROWS, COLS>,
    src: &Rt<F32, RtRow, ROWS, COLS>,
    scalar_expr: &str,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::mul({dst}, {src}, {scalar});",
        dst = dst.expr(),
        src = src.expr(),
        scalar = scalar_expr,
    ))
}

/// `kittens::warp::copy(rt_bf_dst, rt_fl_src);` — element-wise
/// copy with implicit dtype cast (fp32 → bf16). Used to convert
/// the post-softmax probabilities to bf16 before the PV matmul.
///
/// Source: `include/ops/group/register/tile/maps.cuh:445-449`
/// (copy with same shape, different dtype).
pub fn warp_copy_rt_fl_to_bf<const ROWS: u32, const COLS: u32>(
    dst: &Rt<Bf16, RtRow, ROWS, COLS>,
    src: &Rt<F32, RtRow, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::copy({dst}, {src});",
        dst = dst.expr(),
        src = src.expr(),
    ))
}

/// Declare an `int` local with an initializer expression. Used
/// for runtime loop bounds (e.g. `int __attn_num_blocks =
/// (seq_lens[t] + BLOCK_SIZE - 1) / BLOCK_SIZE`).
pub fn decl_local_int(name: &str, init_expr: &str) -> CuStmt {
    CuStmt::new(format!("int {name} = {init_expr};"))
}

/// Declare a `const __nv_bfloat16*` local pointing at a paged-KV
/// block: `cache_base_arr[layer] + block_idx * KV_BLOCK_BYTES /
/// sizeof(bf16)`. The block index expression typically reads
/// `block_table[p]` for the current per-token paged-KV walk.
/// Returns both the decl and a `GmemPtrRaw<Bf16>` handle wrapping
/// the local so downstream TMA bindings type-check.
pub fn decl_paged_kv_block_ptr(
    name: &str,
    cache_base_arr: &str,
    layer: u32,
    block_idx_expr: &str,
    kv_block_bytes: u32,
) -> (CuStmt, GmemPtrRaw<Bf16>) {
    let stmt = CuStmt::new(format!(
        "const __nv_bfloat16* {name} = \
         {cache_base_arr}[{layer}] + \
         (static_cast<size_t>({block_idx_expr}) * \
         {kv_block_bytes} / sizeof(__nv_bfloat16));"
    ));
    let handle = GmemPtrRaw::<Bf16>::from_expr(CuExpr::new(name.to_string()));
    (stmt, handle)
}

/// Declare an fp32 register vector with each lane initialized to
/// negative infinity. Used as the `max_vec` running accumulator
/// in FlashAttention-2's online softmax (initial pass requires
/// max-of-everything to be -INF so any real score wins).
pub fn decl_rv_fl_neg_infty<const LEN: u32>(name: &str) -> (CuStmt, Rv<F32, LEN>) {
    decl_rv_fl_init_scalar::<LEN>(name, "-CUDART_INF_F")
}

/// `kittens::warp::store(st_dst, rt_src);` — store a register
/// tile to a shared tile. Already exists for fp32→bf16 (
/// [`warp_store_st_bf_from_rt_fl`]); this variant covers the
/// same-dtype bf16→bf16 path used when converting the per-Q-head
/// fp32 output back to bf16 in attn_out_page (we go fp32 → bf16
/// register first via [`warp_copy_rt_fl_to_bf`], then bf16 rt →
/// bf16 st via `kittens::warp::store`).
///
/// Source: `include/ops/group/memory/tile/register_to_shared.cuh`.
pub fn warp_store_st_bf_from_rt_bf<const ROWS: u32, const COLS: u32>(
    st: &St<Bf16, ROWS, COLS>,
    rt: &Rt<Bf16, RtRow, ROWS, COLS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::store({st}, {rt});",
        st = st.expr(),
        rt = rt.expr(),
    ))
}

/// Build a `kittens::semaphore` handle reference for a block-
/// scope `__shared__ kittens::semaphore <name>;` previously
/// declared via [`decl_shared_semaphore`].
pub fn local_semaphore_ref(name: &str) -> Semaphore {
    Semaphore::from_expr(CuExpr::new(name.to_string()))
}

/// `kittens::warp::div_row(rt_dst, rt_src, rv_row);` — divide
/// each row of a register tile by the matching scalar in a
/// per-row vector. Used to finalize FlashAttention's output:
/// `o_reg /= sum_vec` per row before the bf16 cast.
///
/// Source: `include/ops/group/register/tile/maps.cuh` (mirror
/// of `mul_row` / `add_row` family).
pub fn warp_div_row<const ROWS: u32, const COLS: u32>(
    rt_dst: &Rt<F32, RtRow, ROWS, COLS>,
    rt_src: &Rt<F32, RtRow, ROWS, COLS>,
    rv_row: &Rv<F32, ROWS>,
) -> CuStmt {
    CuStmt::new(format!(
        "kittens::warp::div_row({dst}, {src}, {rv});",
        dst = rt_dst.expr(),
        src = rt_src.expr(),
        rv = rv_row.expr(),
    ))
}
