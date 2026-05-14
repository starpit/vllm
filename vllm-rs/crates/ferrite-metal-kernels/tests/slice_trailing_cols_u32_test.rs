// SPDX-License-Identifier: Apache-2.0
//
// `slice_trailing_cols_u32` CPU-parity tests.

#![cfg(target_os = "macos")]

use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::slice_trailing_cols_u32::{
    dispatch_slice_trailing_cols_u32, slice_trailing_cols_u32_cpu, Buffer, Device,
    SliceTrailingColsU32Kernels,
};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

fn upload<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let nbytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(nbytes.max(1), MTLResourceOptions::StorageModeShared)
        .expect("buf");
    if nbytes > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents().as_ptr() as *mut u8,
                nbytes,
            );
        }
    }
    buf
}

fn download<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    let mut out = vec![unsafe { std::mem::zeroed::<T>() }; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().as_ptr() as *const u8,
            out.as_mut_ptr() as *mut u8,
            n * std::mem::size_of::<T>(),
        );
    }
    out
}

#[test]
fn slice_trailing_cols_qwen3_moe_router_shape() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = SliceTrailingColsU32Kernels::new(&device).expect("kernels");

    // Qwen3-MoE: N=8 (small bucket), num_experts=128, top_k=8.
    let rows = 8usize;
    let src_cols = 128usize;
    let dst_cols = 8usize;
    let src: Vec<u32> = (0..rows * src_cols).map(|i| i as u32).collect();
    let want = slice_trailing_cols_u32_cpu(&src, rows, src_cols, dst_cols);

    let src_buf = upload(&device, &src);
    let dst_buf = upload(&device, &vec![0u32; rows * dst_cols]);

    dispatch_slice_trailing_cols_u32(
        &kernels,
        &queue,
        &src_buf,
        &dst_buf,
        rows as u32,
        src_cols as u32,
        dst_cols as u32,
    )
    .expect("dispatch");

    let got: Vec<u32> = download(&dst_buf, rows * dst_cols);
    assert_eq!(got, want);
}

#[test]
fn slice_trailing_cols_dst_equals_src() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = SliceTrailingColsU32Kernels::new(&device).expect("kernels");

    // dst_cols == src_cols → identity copy.
    let rows = 3usize;
    let cols = 5usize;
    let src: Vec<u32> = (100..100 + rows * cols).map(|i| i as u32).collect();

    let src_buf = upload(&device, &src);
    let dst_buf = upload(&device, &vec![0u32; rows * cols]);

    dispatch_slice_trailing_cols_u32(
        &kernels,
        &queue,
        &src_buf,
        &dst_buf,
        rows as u32,
        cols as u32,
        cols as u32,
    )
    .expect("dispatch");

    let got: Vec<u32> = download(&dst_buf, rows * cols);
    assert_eq!(got, src);
}
