// SPDX-License-Identifier: Apache-2.0
//! PD-wavefront — operand-addressing probe for the persistent megakernel.
//!
//! The trivial on-GPU tape player needs ONE persistent kernel to reach
//! every operand buffer the tape names (~150 weight/arena/cache buffers),
//! far past Metal's ~31 bind-slot limit — so operands cannot be bound as
//! individual kernel arguments. The mechanism: a plain buffer holding an
//! array of `gpuAddress` u64s; the kernel casts a uint64 to a `device`
//! pointer and dereferences it, with the operand buffers kept resident
//! (here via `useResource`; in production the allocator's shared
//! `MetalResidencySet`). This probe proves that the raw `uint64 -> device
//! pointer` cast+deref actually works on this toolchain — the make-or-break
//! for the whole bindless-operand approach the trivial player rides on.
//!
//! Run: `cargo test -p ferrite-forward -F metal --test wavefront_addr_probe`.
#![cfg(feature = "metal")]

use objc2::msg_send;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void addr_probe(
    device const ulong* addrs [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  // The bindless deref under test: a raw GPU virtual address (gpuAddress)
  // read from a plain buffer, cast to a device pointer, dereferenced. This
  // is how the persistent megakernel will reach an operand it was never
  // `setBuffer`-bound, indexing addrs[BufId] from its operand table.
  device const float* src = (device const float*)(addrs[0]);
  out[tid] = src[tid] * 2.0f + 1.0f;
}
"#;

#[test]
fn gpu_address_table_deref_works() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;

    // Compile the probe kernel from source at runtime (no shader-registration churn).
    let source = NSString::from_str(SRC);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("compile probe MSL");
    let func_name = NSString::from_str("addr_probe");
    let function = library
        .newFunctionWithName(&func_name)
        .expect("addr_probe function");
    let pipe = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("addr_probe pipeline");

    // `data`: 64 known floats; `out`: zeroed; `addrs`: [data.gpuAddress()].
    let n = 64usize;
    let data_vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
    let data_bytes: Vec<u8> = data_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let data = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(data_bytes.as_ptr() as *mut c_void).unwrap(),
            data_bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("data buffer");
    let out = device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .expect("out buffer");

    let addr: u64 = data.gpuAddress();
    let addrs_bytes = addr.to_le_bytes();
    let addrs = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(addrs_bytes.as_ptr() as *mut c_void).unwrap(),
            addrs_bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("addrs buffer");

    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&pipe);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&addrs), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
    }
    // `data` is reached ONLY via its raw address — never bound — so the
    // driver must be told to keep it resident or the deref reads garbage.
    unsafe {
        let _: () = msg_send![&*enc, useResource: &*data, usage: MTLResourceUsage::Read];
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: n,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let got = unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, n) };
    for i in 0..n {
        let want = data_vals[i] * 2.0 + 1.0;
        assert!(
            (got[i] - want).abs() < 1e-5,
            "gpuAddress-table deref[{i}] got {} want {want} \
             (raw uint64 -> device pointer cast+deref FAILED on this toolchain)",
            got[i]
        );
    }
}
