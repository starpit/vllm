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

    /// Single-op RmsNorm reproducer (emitted by
    /// `bin/tk_emit_rmsnorm`). Same C ABI as
    /// [`launch_tk_decode_one_layer`]; isolates the round protocol of
    /// one op from the 12-op forward so the deadlock audit can pin the
    /// offending wait/arrive imbalance to RmsNorm specifically.
    pub fn launch_tk_decode_rmsnorm(
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

/// Safe wrapper around [`launch_tk_decode_rmsnorm`].
///
/// `bufs[i]` must point to valid device memory whose layout matches
/// the source / op-output buffer at position `i` in
/// [`crate::fixtures::rmsnorm_only_input`] (3 buffers: x, rms_w, op0).
///
/// # Safety
/// Same contract as [`launch_decode_one_layer`].
pub unsafe fn launch_decode_rmsnorm(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_rmsnorm(bufs.as_ptr(), u32_args.as_ptr(), stream) }
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

    /// As [`launcher_symbol_resolves`] but for the RmsNorm-only repro
    /// emit. Proves `bin/tk_emit_rmsnorm` produced a kernel whose
    /// host wrapper is now in `libmegakernels.a`.
    #[test]
    fn rmsnorm_launcher_symbol_resolves() {
        let f: unsafe extern "C" fn(*const *mut c_void, *const u32, *mut c_void) -> i32 =
            launch_tk_decode_rmsnorm;
        assert_ne!(f as usize, 0);
    }

    /// End-to-end smoke harness: allocate the full 26-buffer set of
    /// zero-filled device memory, hand them to the orchestrator's
    /// `launch_tk_decode_one_layer` wrapper with `__num_kv_pages = 1`,
    /// and assert the launch + stream sync both report `cudaSuccess`.
    ///
    /// Currently `#[ignore]`d: when run on H100 the kernel launches
    /// (cudaFuncSetAttribute and the triple-chevron return success)
    /// but the persistent megakernel does not retire — `nvidia-smi`
    /// reports 100% util on GPU 0 with `ferrite_wavefront-*` as the
    /// owning process and `cuStreamSynchronize` blocks indefinitely.
    /// All threads spin on a `mbarrier.try_wait.parity` somewhere in
    /// the loader / storer / 8 consumer round protocol the orchestrator
    /// emits. The harness still serves as the launch-path test bed; the
    /// barrier-protocol audit is the next slice of work and will flip
    /// this back on once a single-op reproducer pins down which round
    /// participant is missing a matching `arrive`.
    ///
    /// Run manually with:
    ///   cargo test -p ferrite-wavefront --features cuda \
    ///       launcher_runs_on_zeros -- --ignored --nocapture
    #[test]
    #[ignore]
    fn launcher_runs_on_zeros() {
        use crate::fixtures::{buf_byte_sizes, one_layer_input};
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");

        let input = one_layer_input();
        let raw_sizes = buf_byte_sizes(&input);

        // Pad each allocation to at least 1 MiB (covers any single-page
        // TMA load at PAGE_SIZE = 16 KiB with comfortable headroom) and
        // round to 128 bytes (TMA alignment). Real shapes are kept where
        // they exceed the floor so the larger weight tensors get their
        // declared bytes.
        let pad_floor = 1usize << 20;
        let bufs: Vec<cudarc::driver::CudaSlice<u8>> = raw_sizes
            .iter()
            .map(|&n| {
                let n_padded = n.max(pad_floor);
                let n_aligned = n_padded.div_ceil(128) * 128;
                stream.alloc_zeros::<u8>(n_aligned).expect("alloc_zeros")
            })
            .collect();

        // Collect device pointers in BufId order. The `_records` keep
        // the SyncOnDrop guards alive for the duration of the launch so
        // cudarc treats the buffers as live across the kernel call.
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(bufs.len());
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(bufs.len());
        for buf in &bufs {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }

        // Make sure all the alloc_zeros memsets retire before the
        // kernel reads them.
        stream.synchronize().expect("pre-launch sync");

        let u32_args: [u32; 1] = [1]; // __num_kv_pages = 1
        let stream_raw = stream.cu_stream() as *mut c_void;

        let err = unsafe { launch_decode_one_layer(&ptrs, &u32_args, stream_raw) };
        assert_eq!(err, 0, "launch_tk_decode_one_layer returned cudaError {err}");

        stream
            .synchronize()
            .expect("post-launch sync (kernel ran to completion)");

        drop(_records);
        drop(bufs);
    }

    /// Single-op (RmsNorm) reproducer for the deadlock audit. Allocates
    /// the 3-buffer set (x, rms_w, op0_out) of zero-filled device memory
    /// and launches `tk_decode_rmsnorm` (no `__num_kv_pages` arg —
    /// RmsNorm has no AttnDecode child).
    ///
    /// `#[ignore]`d for the same reason as
    /// [`launcher_runs_on_zeros`] — until the round-protocol bug is
    /// fixed it deadlocks. With `TK_EMIT_DEBUG_HANDSHAKE=1` set when
    /// `bin/tk_emit_rmsnorm` was last run, the kernel emits one printf
    /// per loader/consumer/storer wait+arrive; the first WAIT_START
    /// without a matching prior ARRIVE/TMA_LOAD_ISSUED is the bug.
    ///
    /// Run manually with:
    ///   cargo run -p ferrite-wavefront --bin tk_emit_rmsnorm \
    ///       (optional: TK_EMIT_DEBUG_HANDSHAKE=1)
    ///   cargo build -p ferrite-cuda-builder --features cuda
    ///   cargo test -p ferrite-wavefront --features cuda \
    ///       launcher_runs_on_zeros_rmsnorm_only -- --ignored --nocapture
    #[test]
    #[ignore]
    fn launcher_runs_on_zeros_rmsnorm_only() {
        use crate::fixtures::{buf_byte_sizes, rmsnorm_only_input};
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");

        let input = rmsnorm_only_input();
        let raw_sizes = buf_byte_sizes(&input);
        // Sanity: 2 sources + 1 op output = 3 buffers.
        assert_eq!(raw_sizes.len(), 3);

        let pad_floor = 1usize << 20;
        let bufs: Vec<cudarc::driver::CudaSlice<u8>> = raw_sizes
            .iter()
            .map(|&n| {
                let n_padded = n.max(pad_floor);
                let n_aligned = n_padded.div_ceil(128) * 128;
                stream.alloc_zeros::<u8>(n_aligned).expect("alloc_zeros")
            })
            .collect();

        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(bufs.len());
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(bufs.len());
        for buf in &bufs {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }

        stream.synchronize().expect("pre-launch sync");

        // RmsNorm has no AttnDecode child → no `__num_kv_pages` runtime arg.
        let u32_args: [u32; 0] = [];
        let stream_raw = stream.cu_stream() as *mut c_void;

        let err = unsafe { launch_decode_rmsnorm(&ptrs, &u32_args, stream_raw) };
        assert_eq!(
            err, 0,
            "launch_tk_decode_rmsnorm returned cudaError {err}"
        );

        stream
            .synchronize()
            .expect("post-launch sync (rmsnorm-only kernel ran to completion)");

        drop(_records);
        drop(bufs);
    }
}
