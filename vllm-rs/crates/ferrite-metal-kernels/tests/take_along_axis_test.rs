// SPDX-License-Identifier: Apache-2.0
//! take_along_axis_2d_contig parity test for the MoE router pattern.

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::take_along_axis::{
    dispatch_take_along_axis, TakeAlongAxisKernels, TakeAlongDType,
};
use half::bf16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

#[test]
fn take_along_axis_bf16_qwen3_moe_topk() {
    // gates [N=5, E=128] bf16; inds [N=5, top_k=8] u32.
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let kernels = TakeAlongAxisKernels::new(&mdev.device).expect("kernels");

    let rows = 5usize;
    let e = 128usize;
    let top_k = 8usize;

    let mut state = 0xDEAD_BEEF_DEAD_BEEFu64;
    let mut gates_f32 = vec![0.0_f32; rows * e];
    for slot in &mut gates_f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = ((state as u32) & 0xFFFF) as f32 / 65535.0;
    }
    let gates_bf16: Vec<bf16> = gates_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    // Choose indices: top-8 by gate value per row.
    let mut indices = vec![0u32; rows * top_k];
    for r in 0..rows {
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&a, &b| {
            gates_f32[r * e + b]
                .partial_cmp(&gates_f32[r * e + a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for k in 0..top_k {
            indices[r * top_k + k] = idx[k] as u32;
        }
    }

    let src_buf = mdev
        .device
        .newBufferWithLength_options(rows * e * 2, MTLResourceOptions::StorageModeShared)
        .expect("src");
    let idx_buf = mdev
        .device
        .newBufferWithLength_options(rows * top_k * 4, MTLResourceOptions::StorageModeShared)
        .expect("idx");
    let out_buf = mdev
        .device
        .newBufferWithLength_options(rows * top_k * 2, MTLResourceOptions::StorageModeShared)
        .expect("out");

    unsafe {
        std::ptr::copy_nonoverlapping(
            gates_bf16.as_ptr() as *const u8,
            src_buf.contents().as_ptr() as *mut u8,
            rows * e * 2,
        );
        std::ptr::copy_nonoverlapping(
            indices.as_ptr() as *const u8,
            idx_buf.contents().as_ptr() as *mut u8,
            rows * top_k * 4,
        );
    }

    dispatch_take_along_axis(
        &kernels,
        &queue,
        &src_buf,
        &idx_buf,
        &out_buf,
        rows as u32,
        e as u32,
        top_k as u32,
        TakeAlongDType::BF16,
    )
    .expect("dispatch");

    let mut got = vec![bf16::ZERO; rows * top_k];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents().as_ptr() as *const u8,
            got.as_mut_ptr() as *mut u8,
            rows * top_k * 2,
        );
    }

    for r in 0..rows {
        for k in 0..top_k {
            let want = gates_bf16[r * e + indices[r * top_k + k] as usize];
            assert_eq!(
                got[r * top_k + k].to_bits(),
                want.to_bits(),
                "row {r} k {k}"
            );
        }
    }
}

#[test]
fn take_along_axis_bf16_qwen3_5_moe_topk_e256() {
    // gates [N=9, E=256] bf16; inds [N=9, top_k=8] u32 (Qwen3.5-MoE).
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let kernels = TakeAlongAxisKernels::new(&mdev.device).expect("kernels");

    let rows = 9usize;
    let e = 256usize;
    let top_k = 8usize;

    let mut state = 0xFEED_FACE_DEAD_BEEFu64;
    let mut gates_f32 = vec![0.0_f32; rows * e];
    for slot in &mut gates_f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *slot = ((state as u32) & 0xFFFF) as f32 / 65535.0;
    }
    let gates_bf16: Vec<bf16> = gates_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut indices = vec![0u32; rows * top_k];
    for r in 0..rows {
        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&a, &b| {
            gates_f32[r * e + b]
                .partial_cmp(&gates_f32[r * e + a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for k in 0..top_k {
            indices[r * top_k + k] = idx[k] as u32;
        }
    }

    let src_buf = mdev
        .device
        .newBufferWithLength_options(rows * e * 2, MTLResourceOptions::StorageModeShared)
        .expect("src");
    let idx_buf = mdev
        .device
        .newBufferWithLength_options(rows * top_k * 4, MTLResourceOptions::StorageModeShared)
        .expect("idx");
    let out_buf = mdev
        .device
        .newBufferWithLength_options(rows * top_k * 2, MTLResourceOptions::StorageModeShared)
        .expect("out");

    unsafe {
        std::ptr::copy_nonoverlapping(
            gates_bf16.as_ptr() as *const u8,
            src_buf.contents().as_ptr() as *mut u8,
            rows * e * 2,
        );
        std::ptr::copy_nonoverlapping(
            indices.as_ptr() as *const u8,
            idx_buf.contents().as_ptr() as *mut u8,
            rows * top_k * 4,
        );
    }

    dispatch_take_along_axis(
        &kernels,
        &queue,
        &src_buf,
        &idx_buf,
        &out_buf,
        rows as u32,
        e as u32,
        top_k as u32,
        TakeAlongDType::BF16,
    )
    .expect("dispatch");

    let mut got = vec![bf16::ZERO; rows * top_k];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents().as_ptr() as *const u8,
            got.as_mut_ptr() as *mut u8,
            rows * top_k * 2,
        );
    }

    for r in 0..rows {
        for k in 0..top_k {
            let want = gates_bf16[r * e + indices[r * top_k + k] as usize];
            assert_eq!(
                got[r * top_k + k].to_bits(),
                want.to_bits(),
                "row {r} k {k}"
            );
        }
    }
}
