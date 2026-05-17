// SPDX-License-Identifier: Apache-2.0
//! Parity test for `slice_trailing_cols_u32`.

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::slice_trailing_cols::{
    dispatch_slice_trailing_cols_u32, SliceTrailingColsKernels,
};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

#[test]
fn slice_trailing_cols_u32_qwen3_moe_top8() {
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let kernels = SliceTrailingColsKernels::new(&mdev.device).expect("kernels");

    let rows = 5usize;
    let axis = 128usize;
    let top_k = 8usize;

    let mut host = vec![0u32; rows * axis];
    for r in 0..rows {
        for c in 0..axis {
            host[r * axis + c] = (r * 1000 + c) as u32;
        }
    }

    let src = mdev
        .device
        .newBufferWithLength_options(rows * axis * 4, MTLResourceOptions::StorageModeShared)
        .expect("src");
    let dst = mdev
        .device
        .newBufferWithLength_options(rows * top_k * 4, MTLResourceOptions::StorageModeShared)
        .expect("dst");
    unsafe {
        std::ptr::copy_nonoverlapping(
            host.as_ptr() as *const u8,
            src.contents().as_ptr() as *mut u8,
            rows * axis * 4,
        );
    }

    dispatch_slice_trailing_cols_u32(
        &kernels,
        &queue,
        &src,
        &dst,
        rows as u32,
        axis as u32,
        top_k as u32,
    )
    .expect("dispatch");

    let mut got = vec![0u32; rows * top_k];
    unsafe {
        std::ptr::copy_nonoverlapping(
            dst.contents().as_ptr() as *const u8,
            got.as_mut_ptr() as *mut u8,
            rows * top_k * 4,
        );
    }

    for r in 0..rows {
        for k in 0..top_k {
            let want = (r * 1000 + (axis - top_k) + k) as u32;
            assert_eq!(got[r * top_k + k], want, "row {r} k {k}");
        }
    }
}
