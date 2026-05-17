// SPDX-License-Identifier: Apache-2.0
//! Typed handles for emitted-CUDA values.
//!
//! Each handle wraps a [`CuExpr`](super::cu::CuExpr) plus a phantom
//! dtype tag (and, where it matters, a runtime length stamp). The
//! [`tk`](super::tk) API surface takes handles by type — passing
//! a `SmemHandle<F32>` to a function expecting `SmemHandle<Bf16>`
//! is a Rust compile error, even though both are string fragments
//! under the covers.
//!
//! Lengths (HIDDEN_DIM, K_PER_WARP, ...) are runtime `u32` values
//! pulled from the IR's typed getters, not const generics.
//! Const-generic lengths would be redundant with the CUDA-side
//! `static_assert`s the TK primitives already carry, and they can't
//! be built from runtime IR values anyway.

use std::marker::PhantomData;

use crate::substrate::{PageRef, ScratchOffsetRef};

use super::cu::CuExpr;

// ============================================================
// Dtype tags — phantoms tracking the CUDA dtype carried by a
// handle. Used for type-checking that e.g. a TK call asking for a
// `bf16` shared tile is given one, not an `fp32` one.
// ============================================================

/// `__nv_bfloat16` activations / weights.
#[derive(Clone, Copy, Debug)]
pub struct Bf16;
/// `float` accumulators / partial-sum scratch.
#[derive(Clone, Copy, Debug)]
pub struct F32;
/// `uint32_t` block tables / counters.
#[derive(Clone, Copy, Debug)]
pub struct U32;

/// CUDA token for the dtype this phantom tag stands for. Used by
/// the `cuda_type()` accessor on handles.
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

// ============================================================
// Shared-memory column-vector handle
// ============================================================

/// Typed handle to a `kittens::sv_<dtype><LEN>` shared-memory column
/// vector reference. The CUDA expression is whatever was used to
/// produce it (`*reinterpret_cast<sv_bf<LEN>*>(ss.pages[i])`,
/// `*reinterpret_cast<sv_bf<K_PER_WARP>*>(<parent> + warpid * ...)`,
/// etc.); the length is a runtime `u32` stamped at handle creation.
#[derive(Clone, Debug)]
pub struct SmemColVec<T> {
    expr: CuExpr,
    len: u32,
    _t: PhantomData<T>,
}

impl<T: DtypeName> SmemColVec<T> {
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
    /// CUDA type token, e.g. `kittens::sv_bf<2048>`. Used by the
    /// few TK calls that need to spell out the type for a local
    /// variable declaration; most consumers just splice `expr()`.
    pub fn cuda_type(&self) -> String {
        let prefix = match T::cuda_token() {
            "__nv_bfloat16" => "kittens::sv_bf",
            "float" => "kittens::sv_fl",
            "uint32_t" => "kittens::sv_u",
            other => panic!("SmemColVec: unsupported dtype {other}"),
        };
        format!("{prefix}<{}>", self.len)
    }
}

// ============================================================
// Register-file column-vector handle
// ============================================================

/// Typed handle to a `kittens::rv_<dtype><LEN>` register vector.
/// Returned by ops like `rms_norm_vec` (whose output is a register
/// vector) and consumed by `warp::store(sv, rv)`.
#[derive(Clone, Debug)]
pub struct RegColVec<T> {
    expr: CuExpr,
    len: u32,
    _t: PhantomData<T>,
}

impl<T: DtypeName> RegColVec<T> {
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
}

// ============================================================
// Pointer / scalar handles
// ============================================================

/// Typed handle to a global-memory pointer of dtype `T`.
#[derive(Clone, Debug)]
pub struct GmemPtr<T> {
    expr: CuExpr,
    _t: PhantomData<T>,
}

impl<T: DtypeName> GmemPtr<T> {
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

/// Typed handle to a scratch pointer of dtype `T` (e.g.
/// `float*` for partial-sum reductions).
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

/// Typed handle to a `kittens::semaphore` reference (e.g.
/// `ss.page_ready[3]`). All page-handoff barriers are unitary
/// semaphores from the substrate's POV, so no dtype phantom.
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

// ============================================================
// Substrate-page accessors. Each function takes a typed
// substrate ref (PageRef from the IR) and returns a handle whose
// CUDA expression points at the corresponding `ss.<...>` slot.
// ============================================================

/// `ss.page_ready[<page>]` — the loader-arrives / consumer-waits
/// page-handoff semaphore.
pub fn page_ready_sem(page: PageRef) -> Semaphore {
    Semaphore::from_expr(CuExpr::new(format!("ss.page_ready[{}]", page.raw())))
}

/// `ss.page_done[<page>]` — the consumer-arrives / storer-waits
/// page-handoff semaphore.
pub fn page_done_sem(page: PageRef) -> Semaphore {
    Semaphore::from_expr(CuExpr::new(format!("ss.page_done[{}]", page.raw())))
}

/// `ss.page_consumed[<page>]` — the consumer-arrives /
/// loader-waits semaphore that gates page reuse for the next
/// instruction.
pub fn page_consumed_sem(page: PageRef) -> Semaphore {
    Semaphore::from_expr(CuExpr::new(format!("ss.page_consumed[{}]", page.raw())))
}

/// Reinterpret-cast a substrate page as a `kittens::sv_bf<len>`
/// shared column vector reference.
pub fn page_as_sv_bf(page: PageRef, len: u32) -> SmemColVec<Bf16> {
    SmemColVec::from_expr(
        CuExpr::new(format!(
            "(*reinterpret_cast<kittens::sv_bf<{len}>*>(ss.pages[{}]))",
            page.raw()
        )),
        len,
    )
}

/// Carve a per-warp slice out of a full-length shared bf16 vector
/// for a consumer warp. The slice is a `kittens::sv_bf<k_per_warp>`
/// view into bytes
/// `[warpid * k_per_warp * sizeof(bf16),
///  (warpid + 1) * k_per_warp * sizeof(bf16))` of the parent.
///
/// Mirrors how `ferrite::tk::rms_norm_vec` (and the matvec helpers)
/// carve their per-warp slices. `parent.len()` must equal
/// `ncw * k_per_warp` — verified at debug-build time on the Rust
/// side and re-checked by CUDA's `static_assert` on
/// `SV::length == HIDDEN_DIM / NUM_CONSUMER_WARPS` inside the
/// `ferrite::tk::*` helpers.
pub fn warp_slice_sv_bf(
    parent: &SmemColVec<Bf16>,
    ncw: u32,
    k_per_warp: u32,
) -> SmemColVec<Bf16> {
    debug_assert_eq!(
        parent.len(),
        ncw * k_per_warp,
        "warp_slice_sv_bf: parent.len() must equal NCW * K_PER_WARP"
    );
    SmemColVec::from_expr(
        CuExpr::new(format!(
            "(*reinterpret_cast<kittens::sv_bf<{k_per_warp}>*>(\
             reinterpret_cast<char*>(&{parent}) + kittens::warpid() * {k_per_warp} * sizeof(__nv_bfloat16)))",
            parent = parent.expr()
        )),
        k_per_warp,
    )
}

/// `reinterpret_cast<T*>(ss.scratch + <offset>)` — typed pointer
/// to a scratch byte offset.
pub fn scratch_as<T: DtypeName>(offset: ScratchOffsetRef) -> ScratchPtr<T> {
    ScratchPtr::from_expr(CuExpr::new(format!(
        "reinterpret_cast<{}*>(ss.scratch + {})",
        T::cuda_token(),
        offset.raw()
    )))
}

/// `g.act_ptrs[<slot>]` — typed gmem pointer pulled from the
/// kernel's `act_ptrs[]` argument. Slot index comes from the IR's
/// `in_act_slot()` / `out_act_slot()` typed getters.
pub fn gmem_act_ptr_bf16(slot: u32) -> GmemPtr<Bf16> {
    GmemPtr::from_expr(CuExpr::new(format!("g.act_ptrs[{slot}]")))
}

/// `g.weight_ptrs[<accessor> * NUM_LAYERS + <layer>]` — typed gmem
/// pointer to the per-layer weight tile for the given weight
/// accessor. Accessor index + layer come from the IR.
pub fn gmem_weight_ptr_bf16(accessor: u32, layer: u32, num_layers: u32) -> GmemPtr<Bf16> {
    GmemPtr::from_expr(CuExpr::new(format!(
        "g.weight_ptrs[{accessor} * {num_layers} + {layer}]"
    )))
}

/// `g.input_ids` — typed gmem pointer to the per-token input id
/// table. Used by `Embed`'s loader to gather one embedding row per
/// token. The Globals struct must include `uint32_t* input_ids;`
/// (added unconditionally by `cuda_emit::lower_to_cuda`).
pub fn gmem_input_ids() -> GmemPtr<U32> {
    GmemPtr::from_expr(CuExpr::new("g.input_ids".to_string()))
}

/// `&g.barrier_slots[<edge>]` — slot pointer into the gmem
/// cross-CTA barrier-counter array for the given edge id. Used by
/// `BarrierSignal` (atomicAdd) / `BarrierWait` (spin-load) — see
/// `ferrite_barrier.cuh`.
pub fn gmem_barrier_slot_ptr(edge: u32) -> CuExpr {
    CuExpr::new(format!("&g.barrier_slots[{edge}]"))
}
