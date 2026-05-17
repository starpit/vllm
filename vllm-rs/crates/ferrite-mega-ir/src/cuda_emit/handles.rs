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

use crate::substrate::{PageRef, ScratchOffsetRef};

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

/// Raw global-memory pointer of dtype `T` (e.g. `g.act_ptrs[3]`).
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
// `SharedState<ConfigT>& ss` and `Globals& g` symbols established
// at the kernel entry by `lower_to_cuda::render_source`.
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

/// `g.act_ptrs[<slot>]` — raw bf16 device pointer to the gmem
/// activation row at slot `slot`. Matches the host-side ABI
/// (`__nv_bfloat16* const* act_ptrs`) — no `kittens::gl<...>`
/// construction required.
pub fn gmem_act_ptr_raw(slot: u32) -> GmemPtrRaw<Bf16> {
    GmemPtrRaw::from_expr(CuExpr::new(format!("g.act_ptrs[{slot}]")))
}

/// `const_cast<__nv_bfloat16*>(g.weight_ptrs[<accessor> *
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
        "const_cast<__nv_bfloat16*>(g.weight_ptrs[{accessor} * {num_layers} + {layer}])"
    )))
}

/// `&g.barrier_slots[<edge>]` — raw int32 device pointer to the
/// gmem cross-CTA barrier counter for the given edge. Used by
/// `ferrite::barrier_signal/wait` (see `ferrite_barrier.cuh`).
/// Returns a [`CuExpr`] since the ferrite-substrate barrier
/// helpers take a raw `int32_t*`.
pub fn gmem_barrier_slot_ptr(edge: u32) -> CuExpr {
    CuExpr::new(format!("&g.barrier_slots[{edge}]"))
}

/// `g.input_ids` — raw `const uint32_t*` to the per-token vocab
/// index table. Used by `Embed`'s loader for per-token TMA
/// gather. Returns a [`CuExpr`] (no typed handle).
pub fn gmem_input_ids() -> CuExpr {
    CuExpr::new("g.input_ids".to_string())
}

/// `ss.pages[<page>]` — raw `uint8_t*` byte pointer to the page's
/// shared-memory buffer. Used when the per-token TMA gather needs
/// pointer arithmetic (`+ tok * row_bytes`) rather than a typed
/// `kittens::sv_bf<LEN>` view.
pub fn page_as_byte_ptr(page: PageRef) -> CuExpr {
    CuExpr::new(format!("ss.pages[{}]", page.raw()))
}
