// SPDX-License-Identifier: Apache-2.0
//! extern "C" decl + safe launch wrapper for the step-3b smoke
//! kernel. Kept in a `cuda`-gated module so the Rust crate builds on
//! non-CUDA hosts (the launch wrapper is meaningless without a
//! device, but the crate's type/API surface stays usable).

use cudarc::driver::sys;

/// Fixed shape of the step-3b smoke kernel. Mirrors the constants
/// `SEQ_Q`, `SEQ_K`, `HD` in
/// `crates/ferrite-stencil/csrc/stencil_smoke_sm89.cu`.
pub const SMOKE_SEQ_Q: usize = 8;
pub const SMOKE_SEQ_K: usize = 16;
pub const SMOKE_HD: usize = 8;

unsafe extern "C" {
    /// Defined in `stencil_smoke_sm89.cu`. Inputs/outputs are bf16
    /// row-major: `q [SEQ_Q, HD]`, `k [SEQ_K, HD]`, `v [SEQ_K, HD]`,
    /// `o [SEQ_Q, HD]`. `__nv_bfloat16*` on the device; we pass
    /// opaque `u16*` because the bit-level layout matches and cudarc
    /// has no bf16 alias. Grid/block/stream are chosen inside the
    /// wrapper; 3c swaps this for a configurable launch.
    fn ferrite_stencil_smoke_sm89_launch(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        o: *mut u16,
        stream: u64,
    );
}

/// Launch the smoke kernel. Single CTA of 32 threads (one warp). The
/// kernel's fixed shape is encoded in-device so this wrapper only
/// needs the four device pointers — no dims, no stream (default
/// stream; async launch comes with the first real shape in 3c).
///
/// # Safety
/// Device pointers must be valid allocations of at least
/// `SMOKE_SEQ_Q * SMOKE_HD` (for `q`, `o`) / `SMOKE_SEQ_K * SMOKE_HD`
/// (for `k`, `v`) `u16` elements on the device this process has made
/// current. The kernel reads/writes synchronously on the default
/// stream; the caller must ensure no other stream is touching the
/// same memory concurrently.
pub unsafe fn launch_smoke_sm89(q: u64, k: u64, v: u64, o: u64) -> Result<(), sys::CUresult> {
    unsafe {
        ferrite_stencil_smoke_sm89_launch(
            q as *const u16,
            k as *const u16,
            v as *const u16,
            o as *mut u16,
            0, // default stream
        );
    }
    let rc = unsafe { sys::cuCtxSynchronize() };
    if rc == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(rc)
    }
}
