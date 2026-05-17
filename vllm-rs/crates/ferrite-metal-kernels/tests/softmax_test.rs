// SPDX-License-Identifier: Apache-2.0
//! Parity test for the precise softmax kernel against a CPU
//! reference, exercising the MoE-router shapes from Mixtral (E=8),
//! Qwen2-MoE (E=60), Qwen3-MoE (E=128).

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::softmax::{
    dispatch_softmax, softmax_cpu_f32, SoftmaxDType, SoftmaxKernels,
};
use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

fn fill_random_f32(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = vec![0.0_f32; rows * cols];
    for slot in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = (state as u32) & 0x00FF_FFFF;
        *slot = (bits as f32 / (1u32 << 23) as f32) * 4.0 - 2.0;
    }
    out
}

fn run_case_bf16(rows: usize, cols: usize, seed: u64) {
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("command queue");
    let kernels = SoftmaxKernels::new(&mdev.device).expect("softmax kernels");

    let host_f32 = fill_random_f32(rows, cols, seed);
    let host_bf16: Vec<bf16> = host_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let in_bytes = host_bf16.len() * std::mem::size_of::<bf16>();
    let in_buf = mdev
        .device
        .newBufferWithLength_options(in_bytes, MTLResourceOptions::StorageModeShared)
        .expect("in buf");
    let out_buf = mdev
        .device
        .newBufferWithLength_options(in_bytes, MTLResourceOptions::StorageModeShared)
        .expect("out buf");

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_bf16.as_ptr() as *const u8,
            in_buf.contents().as_ptr() as *mut u8,
            in_bytes,
        );
    }

    dispatch_softmax(
        &kernels,
        &queue,
        &in_buf,
        &out_buf,
        rows as u32,
        cols as u32,
        SoftmaxDType::BF16,
    )
    .expect("softmax dispatch");

    let mut got_bf16 = vec![bf16::ZERO; rows * cols];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents().as_ptr() as *const u8,
            got_bf16.as_mut_ptr() as *mut u8,
            in_bytes,
        );
    }

    let mut want = vec![0.0_f32; rows * cols];
    softmax_cpu_f32(&host_f32, rows, cols, &mut want);

    let mut max_err = 0.0_f32;
    for (i, (&got, &expected)) in got_bf16.iter().zip(want.iter()).enumerate() {
        let g = got.to_f32();
        let err = (g - expected).abs();
        if err > max_err {
            max_err = err;
        }
        // bf16 carries ~3 decimal digits; +/- ulp at the typical
        // probability magnitudes lands around 8e-3 in the worst row.
        assert!(
            err < 1e-2,
            "row{}_col{}: got {} want {} err {}",
            i / cols,
            i % cols,
            g,
            expected,
            err
        );
    }
    eprintln!(
        "softmax_bf16 rows={rows} cols={cols} max_abs_err={:.2e}",
        max_err
    );
}

fn run_case_f16(rows: usize, cols: usize, seed: u64) {
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("command queue");
    let kernels = SoftmaxKernels::new(&mdev.device).expect("softmax kernels");

    let host_f32 = fill_random_f32(rows, cols, seed);
    let host_f16: Vec<f16> = host_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let in_bytes = host_f16.len() * std::mem::size_of::<f16>();
    let in_buf = mdev
        .device
        .newBufferWithLength_options(in_bytes, MTLResourceOptions::StorageModeShared)
        .expect("in buf");
    let out_buf = mdev
        .device
        .newBufferWithLength_options(in_bytes, MTLResourceOptions::StorageModeShared)
        .expect("out buf");

    unsafe {
        std::ptr::copy_nonoverlapping(
            host_f16.as_ptr() as *const u8,
            in_buf.contents().as_ptr() as *mut u8,
            in_bytes,
        );
    }

    dispatch_softmax(
        &kernels,
        &queue,
        &in_buf,
        &out_buf,
        rows as u32,
        cols as u32,
        SoftmaxDType::F16,
    )
    .expect("softmax dispatch");

    let mut got = vec![f16::ZERO; rows * cols];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents().as_ptr() as *const u8,
            got.as_mut_ptr() as *mut u8,
            in_bytes,
        );
    }

    let mut want = vec![0.0_f32; rows * cols];
    softmax_cpu_f32(&host_f32, rows, cols, &mut want);

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g.to_f32() - w).abs();
        assert!(
            err < 1e-3,
            "f16 row{}_col{}: got {} want {} err {}",
            i / cols,
            i % cols,
            g.to_f32(),
            w,
            err
        );
    }
}

#[test]
fn softmax_bf16_mixtral_shape() {
    run_case_bf16(17, 8, 0xC0FFEE);
}

#[test]
fn softmax_bf16_qwen2_moe_shape() {
    run_case_bf16(5, 60, 0xBEEF_F00D);
}

#[test]
fn softmax_bf16_qwen3_moe_shape() {
    run_case_bf16(33, 128, 0xDEAD_BEEF);
}

#[test]
fn softmax_f16_mixtral_shape() {
    run_case_f16(7, 8, 0xCAFE);
}

#[test]
fn softmax_f16_qwen3_moe_shape() {
    run_case_f16(11, 128, 0x1234_5678);
}
