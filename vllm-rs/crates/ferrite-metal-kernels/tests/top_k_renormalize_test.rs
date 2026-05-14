// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]

use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::top_k_renormalize::{
    dispatch_top_k_renormalize, top_k_renormalize_cpu_f32, Buffer, Device, TopKRenormalizeDtype,
    TopKRenormalizeKernels,
};
use half::f16 as Half;
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
fn top_k_renormalize_f16_qwen3_moe_shape() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = TopKRenormalizeKernels::new(&device).expect("kernels");

    // Qwen3-MoE decode: N=4, top_k=8.
    let n = 4usize;
    let top_k = 8usize;
    let scores_f32: Vec<f32> = (0..n * top_k).map(|i| 0.1 + 0.05 * (i as f32)).collect();
    let mut want = scores_f32.clone();
    top_k_renormalize_cpu_f32(&mut want, n, top_k);

    let scores_h: Vec<Half> = scores_f32.iter().map(|&v| Half::from_f32(v)).collect();
    let scores_buf = upload(&device, &scores_h);

    let pipeline = kernels
        .pipeline_for(&device, TopKRenormalizeDtype::F16, top_k as u32)
        .expect("pipeline");
    dispatch_top_k_renormalize(
        &pipeline,
        &queue,
        &scores_buf,
        n as u32,
        top_k as u32,
        2,
    )
    .expect("dispatch");

    let got: Vec<Half> = download(&scores_buf, n * top_k);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let gf = g.to_f32();
        let diff = (gf - w).abs();
        assert!(
            diff < 1e-3,
            "row {} elt {}: got {gf}, want {w}, diff {diff}",
            i / top_k,
            i % top_k,
        );
    }
}
