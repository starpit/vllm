// SPDX-License-Identifier: Apache-2.0
//! FFI to the primitive megakernel persistent kernel.
//!
//! The C++ implementation lives at
//! `vllm-rs/crates/vllm-cuda/csrc/megakernel/prim_mega.cu`. See
//! `MEGA_HANDOFF.md` for the locked design (Phase 1 / PrimMega).
//!
//! This module is a thin `extern "C"` declaration plus a safe
//! launch wrapper. The opcode encoding + tape construction live on
//! the macro side (`interpreters::prim_mega`); nothing
//! megakernel-shaped leaks past this boundary.
//!
//! Symbols are resolved by linking against the
//! `libmegakernels.a` static lib that
//! `ferrite-cuda-builder/build.rs::build_megakernels` produces.

unsafe extern "C" {
    /// Launch the persistent Llama megakernel.
    ///
    /// `tape` points to a device buffer of `tape_len * 32` ints
    /// (one row of `INSTRUCTION_WIDTH = 32` per op). `ptr_table`
    /// is a device array of `void*` the encoder pre-fills with
    /// the run's data pointers; rows index into it for each
    /// pointer-shaped argument.
    ///
    /// The kernel uses `cudaLaunchCooperativeKernel` internally
    /// (grid sync between phases is the phase barrier — see the
    /// .cu's docstring for why `__syncthreads` would deadlock
    /// against early-exiting CTAs from the existing dc_* fns).
    /// `grid_x` therefore must satisfy `numSMs * maxBlocksPerSM`
    /// for the chosen `block_x` + `smem_size`; the launcher in
    /// `ferrite-forward`'s emitted runtime path computes it.
    ///
    /// Returns 0 on success; on failure returns the negation of
    /// the underlying `cudaError_t` (matches the
    /// `cutlass_*_launch` convention).
    pub fn prim_mega_llama_launch(
        tape: *const i32,
        tape_len: i32,
        ptr_table: *const *mut core::ffi::c_void,
        grid_x: i32,
        block_x: i32,
        smem_size: usize,
        stream: u64,
    ) -> i32;
}
