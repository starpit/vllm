// SPDX-License-Identifier: Apache-2.0
//! Typed handles for emitted-CUDA values + ferrite substrate
//! accessors. Each handle carries a [`CuExpr`](super::cu::CuExpr)
//! plus phantom dtype tags + (where it matters) runtime length /
//! shape stamps.
//!
//! Phantom dtypes catch wrong-dtype handle pass-through statically.
//! Lengths come from the IR's typed getters at runtime — the Rust
//! type system can't enumerate them, but the CUDA-side
//! `static_assert`s inside `kittens::group<N>::load(rv, sv)` and
//! `mma_AB(...)` will catch shape mismatches at TK 2.0 compile time.
//!
//! Substrate accessors emit `ss.pages[N]` / `ss.page_ready[N]` /
//! `ss.scratch + offset` references against the ferrite-owned
//! `SharedState<Config>` struct from
//! `crates/ferrite-kernels/csrc/tk/ferrite_substrate.cuh`.

use std::marker::PhantomData;

use crate::ir::substrate::{PageRef, ScratchOffsetRef};

use super::cu::CuExpr;

// ============================================================
// Dtype tags
// ============================================================

#[derive(Clone, Copy, Debug)]
pub struct Bf16;
#[derive(Clone, Copy, Debug)]
pub struct F32;
#[derive(Clone, Copy, Debug)]
pub struct U32;
#[derive(Clone, Copy, Debug)]
pub struct I32;

pub trait DtypeName {
    fn cuda_token() -> &'static str;
}
impl DtypeName for Bf16 {
    fn cuda_token() -> &'static str {
        "__nv_bfloat16"
    }
}
impl DtypeName for F32 {
    fn cuda_token() -> &'static str {
        "float"
    }
}
impl DtypeName for U32 {
    fn cuda_token() -> &'static str {
        "uint32_t"
    }
}
impl DtypeName for I32 {
    fn cuda_token() -> &'static str {
        "int32_t"
    }
}

/// CUDA type name for a `kittens::sv_<dtype><LEN>` of this dtype.
pub trait SvCudaName {
    fn sv_token() -> &'static str;
}
impl SvCudaName for Bf16 {
    fn sv_token() -> &'static str {
        "kittens::sv_bf"
    }
}
impl SvCudaName for F32 {
    fn sv_token() -> &'static str {
        "kittens::sv_fl"
    }
}

/// CUDA type name for a `kittens::rv_<dtype><LEN>` of this dtype.
pub trait RvCudaName {
    fn rv_token() -> &'static str;
}
impl RvCudaName for Bf16 {
    fn rv_token() -> &'static str {
        "kittens::rv_bf"
    }
}
impl RvCudaName for F32 {
    fn rv_token() -> &'static str {
        "kittens::rv_fl"
    }
}

/// CUDA type name for a `kittens::rt_<dtype><R, C, layout>` of this dtype.
pub trait RtCudaName {
    fn rt_token() -> &'static str;
}
impl RtCudaName for Bf16 {
    fn rt_token() -> &'static str {
        "kittens::rt_bf"
    }
}
impl RtCudaName for F32 {
    fn rt_token() -> &'static str {
        "kittens::rt_fl"
    }
}

/// CUDA type name for a `kittens::st_<dtype><R, C>` of this dtype.
pub trait StCudaName {
    fn st_token() -> &'static str;
}
impl StCudaName for Bf16 {
    fn st_token() -> &'static str {
        "kittens::st_bf"
    }
}
impl StCudaName for F32 {
    fn st_token() -> &'static str {
        "kittens::st_fl"
    }
}

// Register tile layout tag — `kittens::ducks::rt_layout::row` or
// `::col`. Phantom-typed so wrong-layout pass-through into mma_AB
// fails to typecheck (mma_AB requires A=row, B=col, C=row, D=row).
#[derive(Clone, Copy, Debug)]
pub struct RtRow;
#[derive(Clone, Copy, Debug)]
pub struct RtCol;
pub trait RtLayoutTag {
    /// CUDA token for explicit layout arg (empty string = default
    /// row layout, omitted from the rt_<dtype><R,C> template args).
    fn cuda_layout_arg() -> &'static str;
}
impl RtLayoutTag for RtRow {
    fn cuda_layout_arg() -> &'static str {
        ""
    }
}
impl RtLayoutTag for RtCol {
    fn cuda_layout_arg() -> &'static str {
        ", kittens::ducks::rt_layout::col"
    }
}

// ============================================================
// Shared / register / pointer / semaphore handles
// ============================================================

/// Shared column vector — `kittens::sv_<dtype><len>`.
#[derive(Clone, Debug)]
pub struct Sv<T> {
    expr: CuExpr,
    len: u32,
    _t: PhantomData<T>,
}
impl<T: DtypeName + SvCudaName> Sv<T> {
    pub(super) fn from_expr(expr: CuExpr, len: u32) -> Self {
        Self {
            expr,
            len,
            _t: PhantomData,
        }
    }
    pub(super) fn expr(&self) -> &CuExpr {
        &self.expr
    }
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn cuda_type(&self) -> String {
        format!("{}<{}>", T::sv_token(), self.len)
    }
}

/// Register column vector — `kittens::rv_<dtype><len>`.
#[derive(Clone, Debug)]
pub struct Rv<T> {
    expr: CuExpr,
    len: u32,
    _t: PhantomData<T>,
}
impl<T: DtypeName + RvCudaName> Rv<T> {
    pub(super) fn from_expr(expr: CuExpr, len: u32) -> Self {
        Self {
            expr,
            len,
            _t: PhantomData,
        }
    }
    pub(super) fn expr(&self) -> &CuExpr {
        &self.expr
    }
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn cuda_type(&self) -> String {
        format!("{}<{}>", T::rv_token(), self.len)
    }
}

/// Shared tile — `kittens::st_<dtype><rows, cols>`. Used for the
/// gemm activation tile (`[M, K]`), per-iter b_tile chunk
/// (`[CHUNK_K, N]`), and accumulator landing.
#[derive(Clone, Debug)]
pub struct St<T> {
    expr: CuExpr,
    rows: u32,
    cols: u32,
    _t: PhantomData<T>,
}
impl<T: DtypeName + StCudaName> St<T> {
    pub(super) fn from_expr(expr: CuExpr, rows: u32, cols: u32) -> Self {
        Self {
            expr,
            rows,
            cols,
            _t: PhantomData,
        }
    }
    pub(super) fn expr(&self) -> &CuExpr {
        &self.expr
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn cuda_type(&self) -> String {
        format!("{}<{}, {}>", T::st_token(), self.rows, self.cols)
    }
}

/// Register tile — `kittens::rt_<dtype><rows, cols, layout>`.
/// Layout phantom catches mma_AB row/col mismatches at codegen
/// build time.
#[derive(Clone, Debug)]
pub struct Rt<T, L> {
    expr: CuExpr,
    rows: u32,
    cols: u32,
    _t: PhantomData<T>,
    _l: PhantomData<L>,
}
impl<T: DtypeName + RtCudaName, L: RtLayoutTag> Rt<T, L> {
    pub(super) fn from_expr(expr: CuExpr, rows: u32, cols: u32) -> Self {
        Self {
            expr,
            rows,
            cols,
            _t: PhantomData,
            _l: PhantomData,
        }
    }
    pub(super) fn expr(&self) -> &CuExpr {
        &self.expr
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn cols(&self) -> u32 {
        self.cols
    }
    pub fn cuda_type(&self) -> String {
        format!(
            "{}<{}, {}{}>",
            T::rt_token(),
            self.rows,
            self.cols,
            L::cuda_layout_arg()
        )
    }
}

/// `kittens::semaphore` reference (e.g. `ss.page_ready[3]`).
#[derive(Clone, Debug)]
pub struct Semaphore {
    expr: CuExpr,
}
impl Semaphore {
    pub(super) fn from_expr(expr: CuExpr) -> Self {
        Self { expr }
    }
    pub(super) fn expr(&self) -> &CuExpr {
        &self.expr
    }
}

/// Raw global-memory pointer of dtype `T` (e.g. `act_ptrs[3]`).
/// Used for non-tensor TMA which takes raw pointers + byte counts
/// — matches the existing host-side ABI (`__nv_bfloat16* const*
/// act_ptrs` etc.) without requiring `kittens::gl<...>` host
/// construction.
#[derive(Clone, Debug)]
pub struct GmemPtrRaw<T> {
    expr: CuExpr,
    _t: PhantomData<T>,
}
impl<T: DtypeName> GmemPtrRaw<T> {
    pub(super) fn from_expr(expr: CuExpr) -> Self {
        Self {
            expr,
            _t: PhantomData,
        }
    }
    pub(super) fn expr(&self) -> &CuExpr {
        &self.expr
    }
}

/// Scratch pointer of dtype `T` (e.g. `reinterpret_cast<float*>(
/// ss.scratch + 0)`).
#[derive(Clone, Debug)]
pub struct ScratchPtr<T> {
    expr: CuExpr,
    _t: PhantomData<T>,
}
impl<T: DtypeName> ScratchPtr<T> {
    pub(super) fn from_expr(expr: CuExpr) -> Self {
        Self {
            expr,
            _t: PhantomData,
        }
    }
    pub(super) fn expr(&self) -> &CuExpr {
        &self.expr
    }
}

// ============================================================
// Ferrite substrate accessors. Emit references against the
// `SharedState<ConfigT>& ss` symbol and the kernel-entry args
// (`act_ptrs`, `weight_ptrs`, `barrier_slots`, `input_ids`)
// established by `lower_to_cuda::render_source`.
// ============================================================

/// `ss.page_ready[<page>]` — loader-arrives / consumer-waits.
pub fn page_ready_sem(page: PageRef) -> Semaphore {
    Semaphore::from_expr(CuExpr::new(format!("ss.page_ready[{}]", page.raw())))
}

/// `ss.page_done[<page>]` — consumer-arrives / storer-waits.
pub fn page_done_sem(page: PageRef) -> Semaphore {
    Semaphore::from_expr(CuExpr::new(format!("ss.page_done[{}]", page.raw())))
}

/// `ss.page_consumed[<page>]` — consumer-arrives / loader-waits.
pub fn page_consumed_sem(page: PageRef) -> Semaphore {
    Semaphore::from_expr(CuExpr::new(format!("ss.page_consumed[{}]", page.raw())))
}

/// Reinterpret-cast a substrate page as a `kittens::sv_bf<len>`.
pub fn page_as_sv_bf(page: PageRef, len: u32) -> Sv<Bf16> {
    Sv::from_expr(
        CuExpr::new(format!(
            "(*reinterpret_cast<kittens::sv_bf<{len}>*>(ss.pages[{}]))",
            page.raw()
        )),
        len,
    )
}

/// `reinterpret_cast<T*>(ss.scratch + <offset>)`.
pub fn scratch_as<T: DtypeName>(offset: ScratchOffsetRef) -> ScratchPtr<T> {
    ScratchPtr::from_expr(CuExpr::new(format!(
        "reinterpret_cast<{}*>(ss.scratch + {})",
        T::cuda_token(),
        offset.raw()
    )))
}

/// `act_ptrs[<slot>]` — raw bf16 device pointer to the gmem
/// activation row at slot `slot`. References the kernel's
/// `__nv_bfloat16* const* act_ptrs` parameter directly.
pub fn gmem_act_ptr_raw(slot: u32) -> GmemPtrRaw<Bf16> {
    GmemPtrRaw::from_expr(CuExpr::new(format!("act_ptrs[{slot}]")))
}

/// `const_cast<__nv_bfloat16*>(weight_ptrs[<accessor> *
/// NUM_LAYERS + <layer>])` — raw bf16 pointer with the `const`
/// stripped (TK 2.0's non-tensor `tma::load_async(void* dst,
/// void* src, ...)` takes `void*` non-const for src).
///
/// The cast is safe — the kernel only reads weights — but TK 2.0
/// has no `const void*` src overload, so we strip at the accessor.
pub fn gmem_weight_ptr_raw(
    accessor: u32,
    layer: u32,
    num_layers: u32,
) -> GmemPtrRaw<Bf16> {
    GmemPtrRaw::from_expr(CuExpr::new(format!(
        "const_cast<__nv_bfloat16*>(weight_ptrs[{accessor} * {num_layers} + {layer}])"
    )))
}

/// `const_cast<__nv_bfloat16*>(weight_ptrs[<accessor> *
/// NUM_LAYERS + <layer>]) + <byte_offset> / sizeof(__nv_bfloat16)`
/// — raw bf16 pointer to a sub-region of the weight tensor at
/// `byte_offset` past the accessor's base pointer. Used by
/// `FusedGateUpActivateMul` whose single fused weight tensor is
/// `[gate || up]` concatenated; the up half lives at byte offset
/// `gate_bytes` past the base.
///
/// `byte_offset` is divided by `sizeof(__nv_bfloat16)` (== 2)
/// because the underlying pointer arithmetic is in elements.
pub fn gmem_weight_ptr_raw_offset(
    accessor: u32,
    layer: u32,
    num_layers: u32,
    byte_offset: u32,
) -> GmemPtrRaw<Bf16> {
    let element_offset = byte_offset / 2;
    GmemPtrRaw::from_expr(CuExpr::new(format!(
        "(const_cast<__nv_bfloat16*>(weight_ptrs[{accessor} * {num_layers} + {layer}]) + {element_offset})"
    )))
}

/// `&barrier_slots[<edge>]` — raw int32 device pointer to the
/// gmem cross-CTA barrier counter for the given edge. Used by
/// `ferrite::barrier_signal/wait` (see `ferrite_barrier.cuh`).
/// Returns a [`CuExpr`] since the ferrite-substrate barrier
/// helpers take a raw `int32_t*`.
pub fn gmem_barrier_slot_ptr(edge: u32) -> CuExpr {
    CuExpr::new(format!("&barrier_slots[{edge}]"))
}

/// `input_ids` — raw `const uint32_t*` to the per-token vocab
/// index table. Used by `Embed`'s loader for per-token TMA
/// gather. Returns a [`CuExpr`] (no typed handle).
pub fn gmem_input_ids() -> CuExpr {
    CuExpr::new("input_ids".to_string())
}

/// `positions` — raw `const uint32_t*` to the per-token rotary
/// position table. Sized `[NUM_TOKENS]`. Surfaced by the QKV / Attn
/// tier kernel signatures (see
/// `cuda_emit::LaunchTier` + the host `LaunchArgsQkv::positions`
/// in `crates/ferrite-forward/src/interpreter/mega/mod.rs:295`).
/// Used by `FusedQkvRopeCache`'s loader for per-token cos/sin
/// gather: row index = `positions[t]`, row size = `head_dim *
/// sizeof(bf16)` bytes.
pub fn gmem_positions() -> CuExpr {
    CuExpr::new("positions".to_string())
}

/// `slot_mapping` — raw `const int64_t*` to the per-token paged-KV
/// slot index table. Sized `[NUM_TOKENS]`. Surfaced by QKV / Attn
/// tier kernel signatures (host `LaunchArgsQkv::slot_mapping`
/// at mod.rs:296). FQRC's in-kernel emit doesn't dereference it
/// today (cache writes happen as a follow-up D2D outside the
/// megakernel — see `feedback_ff_mega_cuda_emit_s15a_handoff`),
/// but the accessor exists for future ops that fold the cache
/// write inside.
pub fn gmem_slot_mapping() -> CuExpr {
    CuExpr::new("slot_mapping".to_string())
}

/// `key_cache_ptrs[<layer>]` — raw `__nv_bfloat16*` to layer
/// `layer`'s paged K-cache base pointer. The host stages
/// `[NUM_LAYERS]` device pointers (one per layer's paged KV
/// block pool) in declaration order; the kernel indexes by the
/// per-op compile-time layer constant. Layout per pointer:
/// `[num_blocks, block_size, num_kv_heads, head_dim]` — same
/// vLLM-NHD convention the vendored `flash_api` path uses.
/// Returned as a `GmemPtrRaw<Bf16>` so TK 2.0 TMA primitives
/// accept it without further casts.
pub fn gmem_key_cache_ptr(layer: u32) -> GmemPtrRaw<Bf16> {
    GmemPtrRaw::from_expr(CuExpr::new(format!("key_cache_ptrs[{layer}]")))
}

/// `value_cache_ptrs[<layer>]` — raw `__nv_bfloat16*` to layer
/// `layer`'s paged V-cache base pointer. Same shape and indexing
/// rules as [`gmem_key_cache_ptr`].
pub fn gmem_value_cache_ptr(layer: u32) -> GmemPtrRaw<Bf16> {
    GmemPtrRaw::from_expr(CuExpr::new(format!("value_cache_ptrs[{layer}]")))
}

/// `ss.pages[<page>]` — raw `uint8_t*` byte pointer to the page's
/// shared-memory buffer. Used when the per-token TMA gather needs
/// pointer arithmetic (`+ tok * row_bytes`) rather than a typed
/// `kittens::sv_bf<LEN>` view.
pub fn page_as_byte_ptr(page: PageRef) -> CuExpr {
    CuExpr::new(format!("ss.pages[{}]", page.raw()))
}

/// Reinterpret-cast a substrate page as a `kittens::st_bf<rows, cols>`.
/// Used for the gemm activation tile view of in_page (shape `[M, K]`)
/// and the gemm output tile view of out_page (shape `[M, N]`).
pub fn page_as_st_bf(page: PageRef, rows: u32, cols: u32) -> St<Bf16> {
    St::from_expr(
        CuExpr::new(format!(
            "(*reinterpret_cast<kittens::st_bf<{rows}, {cols}>*>(ss.pages[{}]))",
            page.raw()
        )),
        rows,
        cols,
    )
}

/// Reinterpret-cast a slice of substrate scratch starting at `offset`
/// as a `kittens::st_bf<rows, cols>`. Used for gemm b_tile staging
/// (`[CHUNK_K, N]` per iter).
pub fn scratch_as_st_bf(offset: ScratchOffsetRef, rows: u32, cols: u32) -> St<Bf16> {
    St::from_expr(
        CuExpr::new(format!(
            "(*reinterpret_cast<kittens::st_bf<{rows}, {cols}>*>(ss.scratch + {}))",
            offset.raw()
        )),
        rows,
        cols,
    )
}
