// SPDX-License-Identifier: Apache-2.0
//! Runtime tile-output table consumed by the generated interpreter.
//!
//! Today's ferrite-forward emits a fully-unrolled forward fn with one
//! `let tile_42 = ...;` per FUF tile. Output aliasing rides on Rust's
//! scope rules — `let normed = unsafe { (*delta).as_view() };` borrows
//! from the upstream's `OwnedTensor`, the borrow lives until the
//! upstream goes out of scope.
//!
//! The instruction-list interpreter doesn't have per-tile let-bindings;
//! instead each tile's output occupies a slot in this table, indexed
//! by tile id. The instruction's fields encode "read from these slot
//! ids, write to that slot id." The table is what stands in for Rust
//! scope.
//!
//! # Aliasing
//!
//! [`TileEntry::Owned`] is the storage-owning variant — equivalent to
//! a `let tile_X = OwnedTensor` binding today.
//!
//! [`TileEntry::View`] is the non-owning variant — equivalent to a
//! `let tile_Y = (*tile_X).as_view()` borrow today. The `ref_slot`
//! field points at the slot whose `OwnedTensor` actually owns the
//! storage; reads through the view resolve to that owner.
//!
//! No GPU memory is copied to materialize an alias — `View` is a
//! pointer-sized indirection that the interpreter dereferences when
//! passing arguments to a kernel call.
//!
//! # Drop discipline
//!
//! Slots are dropped via the `FREE` opcode (`opcode::FREE`), emitted
//! by the macro's drop-pass after each slot's last reader. The
//! interpreter handles `FREE` by calling `tiles[slot] = None`,
//! which:
//! - For `Owned`, drops the `OwnedTensor` and returns its GPU memory
//!   to the caching allocator.
//! - For `View`, drops the indirection (no GPU memory belongs to a
//!   view, so this is a Rust-level no-op except for clearing the
//!   slot).
//!
//! Reading a `View` whose `ref_slot` has already been dropped is a
//! programming error — the macro's drop-pass guarantees the owner
//! outlives every view of it. Debug builds assert; release builds
//! UB.
//!
//! # Why not store `TensorView<'_>` directly
//!
//! `TensorView` carries a borrow lifetime; the table can't hold
//! borrows that outlive their owners under Rust's borrow rules.
//! `View { ref_slot }` is the workaround — the indirection is
//! unchecked at compile time but checked structurally by the
//! drop-pass.

#![cfg(feature = "cuda")]
#![allow(dead_code)]

use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::tensor::{GpuTensor, TensorView};

/// One slot in the runtime tile table.
#[derive(Debug)]
pub enum TileEntry {
    /// Storage-owning entry. Drops the `OwnedTensor` when the slot
    /// is freed.
    Owned(OwnedTensor),

    /// Non-owning view of another slot's storage. The macro's
    /// drop-pass guarantees `ref_slot` is still live (i.e.
    /// `Some(_)`) for as long as this entry is.
    View { ref_slot: u32 },
}

impl TileEntry {
    /// Produce the underlying `GpuTensor` for this entry, resolving
    /// `View` indirection through the table. Used by interpreter
    /// arms when passing arguments to kernel calls.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that `tiles` is indexed by tile id and
    /// that any `View::ref_slot` chain terminates at an `Owned`
    /// entry (no view-of-view chains in practice; the macro's
    /// alias-resolution emits direct ref_slot to the owning slot).
    /// Cycles and dangling refs are programming errors.
    #[inline]
    pub fn as_gpu_tensor(&self, tiles: &[Option<TileEntry>]) -> GpuTensor {
        match self {
            Self::Owned(t) => t.as_gpu_tensor(),
            Self::View { ref_slot } => {
                let owner = tiles[*ref_slot as usize]
                    .as_ref()
                    .expect("FREE'd slot referenced via View — drop-pass invariant violated");
                // The drop-pass currently emits direct refs (no view
                // chains). If we ever introduce view-of-view, this
                // single-step resolution becomes recursive and the
                // termination guarantee shifts.
                match owner {
                    Self::Owned(t) => t.as_gpu_tensor(),
                    Self::View { .. } => panic!(
                        "view chain not supported: alias-resolution should always point at the \
                         owning slot directly"
                    ),
                }
            }
        }
    }

    /// Borrow as a `TensorView` for kernels that take views.
    ///
    /// # Safety
    ///
    /// Same as [`as_gpu_tensor`] — caller guarantees the entry is
    /// still live and any indirection resolves cleanly.
    #[inline]
    pub unsafe fn as_view<'a>(&'a self, tiles: &'a [Option<TileEntry>]) -> TensorView<'a> {
        unsafe { TensorView::from_raw(self.as_gpu_tensor(tiles)) }
    }
}

/// Convenience accessor used by generated interpreter arms.
///
/// Reads slot `idx` from `tiles`, panics with a clear message if
/// the slot was never written or has been freed. Generated arms
/// use this rather than open-coding the unwrap so the panic site
/// has a stable name in profiles + backtraces.
#[inline]
pub fn tile_ref(tiles: &[Option<TileEntry>], idx: u32) -> &TileEntry {
    tiles[idx as usize]
        .as_ref()
        .unwrap_or_else(|| panic!("tile slot {idx} read before write or after free"))
}

#[cfg(test)]
mod tests {
    // Tests for the table semantics belong with a real OwnedTensor
    // available — i.e. as integration tests under a feature-gated
    // GPU harness, not unit tests here. Keeping the unit-test stub
    // so future feature-gated tests have a home.
}
