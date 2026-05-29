// SPDX-License-Identifier: Apache-2.0
//! Ferrite-side host launcher for the orchestrator-emitted megakernel.
//!
//! `tk_codegen::emit_kernel` emits a paired `__global__ void <name>(...)`
//! kernel and a C-linkage host wrapper
//! `extern "C" cudaError_t launch_<name>(void* const* bufs, const
//! uint32_t* u32_args, cudaStream_t stream)`. The wrapper:
//!   - sets `cudaFuncAttributeMaxDynamicSharedMemorySize` to
//!     `NUM_PAGES * PAGE_SIZE`,
//!   - launches with `<<<1, total_threads, DYN_SMEM, stream>>>`,
//!   - forwards each `bufs[i]` cast to the declared kernel arg type
//!     and each `u32_args[j]` to the corresponding runtime u32.
//!
//! This module exposes a Rust FFI binding to that wrapper. Linking is
//! provided by the `cuda` feature (which pulls in `ferrite-cuda-builder`,
//! whose `build.rs` emits `rustc-link-lib=static=megakernels`).
//!
//! No vllm `CudaWorker`, no `CandleWorker`. The path is:
//!   ferrite-wavefront → libmegakernels.a → CUDA driver.

use std::ffi::c_void;

// FFI signature for the orchestrator's emitted launcher.
//
// `bufs` MUST be a contiguous array of device pointers in the same
// order the orchestrator declared them (sources first, then per-op
// output staging buffers). `u32_args` carries the runtime u32 args
// the kernel signature lists in `KernelArgs::u32_args` (today only
// `__num_kv_pages` for AttnDecode-bearing programs).
//
// Returns the underlying `cudaError_t` (0 = `cudaSuccess`).
unsafe extern "C" {
    pub fn launch_tk_decode_one_layer(
        bufs: *const *mut c_void,
        u32_args: *const u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Safe wrapper around [`launch_tk_decode_one_layer`].
///
/// `bufs[i]` must point to valid device memory whose layout matches
/// the source / op-output buffer at position `i` in
/// [`crate::fixtures::one_layer_input`]. `stream` is a `CUstream` /
/// `cudaStream_t` cast to `*mut c_void` (the C ABI is identical).
///
/// # Safety
/// Caller must ensure all `bufs` pointers are valid device-side, the
/// stream belongs to the active CUDA context, and the device has at
/// least `NUM_PAGES * PAGE_SIZE` bytes of dynamic shared memory
/// available (true for any sm_90 device).
pub unsafe fn launch_decode_one_layer(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_one_layer(bufs.as_ptr(), u32_args.as_ptr(), stream) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forces the linker to resolve `launch_tk_decode_one_layer` —
    /// proves the orchestrator's emitted launcher made it into
    /// `libmegakernels.a` and the build.rs link directives wired the
    /// archive into our test binary. We never invoke the function
    /// (would need real device memory and a stream), but taking its
    /// address forces the symbol to land in the test binary.
    #[test]
    fn launcher_symbol_resolves() {
        let f: unsafe extern "C" fn(*const *mut c_void, *const u32, *mut c_void) -> i32 =
            launch_tk_decode_one_layer;
        assert_ne!(f as usize, 0);
    }
}
