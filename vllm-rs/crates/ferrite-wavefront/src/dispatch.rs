// SPDX-License-Identifier: Apache-2.0
//! CUDA megakernel dispatch — runtime glue between the orchestrator-
//! emitted `tk_decode_*` kernels and a real ferrite forward path.
//!
//! Mirrors `interpreter/metal/mega_player.rs` for CUDA: resolves each
//! source slot to a device pointer (the caller hands them in already-
//! resolved, since the macro-emitted weight accessors live per-model
//! and aren't easy to thread through a generic trait at this layer),
//! allocates op-output staging buffers as a flat arena per the
//! `LoweringInput`'s shape inference, and calls `launch_<name>(...)`.
//!
//! The narrow API is intentional: the caller (worker / model crate) is
//! the only place that knows where `weights.layer[0].q_proj` lives in
//! HBM, what the runtime hidden-state pointer is, etc. This module
//! takes those pointers as a slice and does only the byte-level
//! plumbing the orchestrator's kernel signature dictates.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use crate::fixtures::{buf_byte_sizes, one_layer_input};

/// Allocate op-output staging buffers for the orchestrator's
/// `tk_decode_one_layer` kernel and dispatch with the caller-supplied
/// source pointers. Returns the allocated buffers (typed as
/// `cudarc::driver::CudaSlice<u8>`) so the caller can read back any
/// op's intermediate / final output.
///
/// `source_ptrs[i]` MUST point to a device-side bf16 buffer matching
/// the `i`-th source in [`one_layer_input`] (same convention as
/// `bin/tk_emit_decode`):
///   0 = x (post-embed hidden state)
///   1 = rms_w0
///   2..4 = q_w / k_w / v_w
///   5,6 = cos / sin
///   7,8 = k_cache / v_cache slices
///   9 = o_w
///   10 = rms_w1
///   11..13 = gate_w / up_w / down_w
///
/// The op-output buffers (positions 14..25 in the kernel arg list) are
/// allocated here as a fresh per-step arena, sized to the per-op
/// shape inference in [`crate::fixtures::buf_byte_sizes`]. `op_outputs[i]`
/// is op `i`'s output (`op11_out` is the final post-MLP residual).
///
/// `num_kv_pages` is the runtime u32 the kernel reads to bound the
/// AttnDecode KV sweep.
///
/// # Safety
/// `source_ptrs[i]` must point to valid device memory of at least the
/// declared bf16 size of source `i`. `stream` must be a valid
/// `cudaStream_t` cast to `*mut c_void` belonging to the active CUDA
/// context that allocated `op_outputs`. Returns `Err(cudaError)` from
/// the launcher if the kernel or `cudaFuncSetAttribute` fails.
pub unsafe fn dispatch_one_layer_decode(
    source_ptrs: &[*mut c_void; 14],
    num_kv_pages: u32,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<Vec<cudarc::driver::CudaSlice<u8>>, i32> {
    use cudarc::driver::DevicePtr;

    let input = one_layer_input();
    let raw_sizes = buf_byte_sizes(&input);
    let n_sources = input.sources.len();
    debug_assert_eq!(n_sources, 14);
    debug_assert_eq!(raw_sizes.len(), 14 + 12);

    // Allocate op-output arena: one zeroed slice per op output. Pad
    // each to 128-byte alignment (TMA requirement); op outputs are
    // small (≤ hidden bf16 = 4096 B), but the larger Gemm outputs at
    // intermediate-size (8192 bf16 = 16384 B = 1 page) need their full
    // bytes. The kernel writes one tile per op, sized by the op's
    // declared output shape.
    let mut op_outputs: Vec<cudarc::driver::CudaSlice<u8>> = Vec::with_capacity(12);
    for &n in &raw_sizes[n_sources..] {
        let n_aligned = n.div_ceil(128) * 128;
        let buf = stream
            .alloc_zeros::<u8>(n_aligned.max(128))
            .map_err(|_| -1_i32)?;
        op_outputs.push(buf);
    }

    // Build the kernel pointer table: 14 caller-supplied source ptrs,
    // followed by 12 op-output ptrs (in BufId order). cudarc gives us
    // device pointers via `DevicePtr::device_ptr(buf, stream)`; we
    // hold the resulting `SyncOnDrop` records alive across the launch.
    let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(14 + 12);
    for &p in source_ptrs.iter() {
        ptrs.push(p);
    }
    let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(12);
    for buf in &op_outputs {
        let (dptr, rec) = DevicePtr::device_ptr(buf, stream.as_ref());
        ptrs.push(dptr as *mut c_void);
        _records.push(rec);
    }

    let u32_args: [u32; 1] = [num_kv_pages];
    let stream_raw = stream.cu_stream() as *mut c_void;

    let err = unsafe { crate::launcher::launch_decode_one_layer(&ptrs, &u32_args, stream_raw) };
    if err != 0 {
        return Err(err);
    }

    drop(_records);
    Ok(op_outputs)
}
