// SPDX-License-Identifier: Apache-2.0
//! CUDA megakernel dispatch — runtime glue between the orchestrator-
//! emitted `tk_decode_*` kernels and a real ferrite forward path.
//!
//! Mirrors `interpreter/metal/mega_player.rs` for CUDA: caller hands
//! us 14 source pointers (already resolved to device addresses by
//! the model-specific weight accessors / runtime context), we
//! allocate the 12-buf op-output staging arena per the
//! `LoweringInput`'s shape inference, and call `launch_<name>(...)`.
//!
//! Uses raw `CUstream` + `*mut u8` allocation directly so it sits in
//! the same layer as `ferrite_cuda_core::GpuDevice` (the worker's
//! existing primitive), with no cudarc bridging required. The
//! launcher's C ABI takes a `*mut c_void` stream — stream-agnostic —
//! so the cudarc-flavored smoke tests in `launcher.rs` keep working
//! unchanged.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use crate::fixtures::{buf_byte_sizes, one_layer_input};

/// One arena buffer's raw allocation. The dispatcher returns these so
/// the caller can read op outputs back / hand them to a follow-on
/// kernel. The pointer is owned: the caller is responsible for
/// freeing it (via the same allocator that produced it). For the
/// worker integration this typically means dropping into the next
/// step's tile reuse, or letting the caching allocator reclaim it
/// at end-of-step.
pub struct ArenaBuf {
    pub ptr: *mut u8,
    pub bytes: usize,
}

/// Allocate the op-output arena via `alloc_fn` and dispatch
/// `launch_tk_decode_one_layer` with the caller-supplied 14 source
/// pointers + the freshly-allocated arena. Returns the arena buffers
/// (pointer + size) so the caller can read any op's intermediate /
/// final output. `op_outputs[i]` is op `i`'s output (`op11_out` is
/// the post-MLP residual final).
///
/// `source_ptrs[i]` MUST point to a device-side bf16 buffer matching
/// the `i`-th source in [`one_layer_input`]:
///   0 = x (post-embed hidden state)
///   1 = rms_w0
///   2..4 = q_w / k_w / v_w
///   5,6 = cos / sin
///   7,8 = k_cache / v_cache slices
///   9 = o_w
///   10 = rms_w1
///   11..13 = gate_w / up_w / down_w
///
/// `alloc_fn(bytes) -> *mut u8` allocates `bytes` of zero-initialized
/// device memory; the implementation should typically be a closure
/// over the worker's `GpuDevice::caching.alloc(bytes)` followed by
/// `cuMemsetD8Async(ptr, 0, bytes, stream)`. Caller controls the
/// lifetime of the returned arena pointers.
///
/// `num_kv_pages` is the runtime u32 the kernel reads to bound the
/// AttnDecode KV sweep (typically `1` for the single-layer slice
/// fixture).
///
/// `stream` is the raw `cudaStream_t` (cast to `*mut c_void`) the
/// kernel launches on.
///
/// # Safety
/// `source_ptrs[i]` must point to valid device memory of at least the
/// declared bf16 size of source `i`. `alloc_fn` must return device
/// pointers from the same CUDA context as `stream`. `stream` must be
/// a valid `cudaStream_t` belonging to the active CUDA context.
/// Returns `Err(cudaError)` from the launcher if the kernel or
/// `cudaFuncSetAttribute` fails.
pub unsafe fn dispatch_one_layer_decode(
    source_ptrs: &[*mut c_void; 14],
    num_kv_pages: u32,
    stream: *mut c_void,
    mut alloc_fn: impl FnMut(usize) -> *mut u8,
) -> Result<Vec<ArenaBuf>, i32> {
    let input = one_layer_input();
    let raw_sizes = buf_byte_sizes(&input);
    let n_sources = input.sources.len();
    debug_assert_eq!(n_sources, 14);
    debug_assert_eq!(raw_sizes.len(), 14 + 12);

    // Allocate op-output arena: one zeroed slice per op output. Pad
    // each to 128-byte alignment (TMA requirement). The kernel writes
    // one tile per op, sized by the op's declared output shape; the
    // post-padding bytes stay zero (initialized by `alloc_fn`).
    let mut arena: Vec<ArenaBuf> = Vec::with_capacity(12);
    for &n in &raw_sizes[n_sources..] {
        let bytes = (n.max(128).div_ceil(128)) * 128;
        let ptr = alloc_fn(bytes);
        if ptr.is_null() {
            return Err(-1);
        }
        arena.push(ArenaBuf { ptr, bytes });
    }

    // Build the kernel pointer table: 14 caller-supplied source ptrs,
    // followed by 12 op-output ptrs (in BufId order).
    let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(14 + 12);
    for &p in source_ptrs.iter() {
        ptrs.push(p);
    }
    for buf in &arena {
        ptrs.push(buf.ptr as *mut c_void);
    }

    let u32_args: [u32; 1] = [num_kv_pages];
    let err = unsafe { crate::launcher::launch_decode_one_layer(&ptrs, &u32_args, stream) };
    if err != 0 {
        return Err(err);
    }

    Ok(arena)
}
