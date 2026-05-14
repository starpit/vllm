// SPDX-License-Identifier: Apache-2.0
//
// MoE weighted-sum kernel CPU-parity tests.
//
// Validates `moe_weighted_sum_<dtype>` against the CPU reference at
// `ferrite_metal_kernels::moe_weighted_sum::moe_weighted_sum_cpu_f32`.
// Coverage: BF16 + F16, decode (N=1) + small-prefill (N=8) shapes,
// representative Qwen3-MoE dims (top_k=8, hidden=2048).

#![cfg(target_os = "macos")]

use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::moe_weighted_sum::{
    dispatch_moe_weighted_sum, moe_weighted_sum_cpu_f32, Buffer, Device, MoeWeightedSumDtype,
    MoeWeightedSumKernels,
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

fn f32_to_bf16(x: f32) -> u16 {
    // Round-to-nearest-even bf16 conversion. Same convention as MLX.
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[test]
fn moe_weighted_sum_bf16_decode_qwen3_moe_shape() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = MoeWeightedSumKernels::new(&device).expect("MoeWeightedSumKernels::new");

    // Qwen3-MoE-30B decode shape: N=1, top_k=8, hidden=2048.
    let n: u32 = 1;
    let top_k: u32 = 8;
    let hidden: u32 = 2048;

    let pipeline = kernels
        .pipeline_for(&device, MoeWeightedSumDtype::BF16, top_k, hidden)
        .expect("pipeline");

    // Synthetic inputs: deterministic, small magnitudes so float and
    // bf16 stay close.
    let expert_out_f32: Vec<f32> = (0..(n * top_k * hidden) as usize)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
        .collect();
    let scores_f32: Vec<f32> = (0..(n * top_k) as usize)
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect();
    let want = moe_weighted_sum_cpu_f32(
        &expert_out_f32,
        &scores_f32,
        n as usize,
        top_k as usize,
        hidden as usize,
    );

    let expert_out_bf: Vec<u16> = expert_out_f32.iter().copied().map(f32_to_bf16).collect();
    let scores_bf: Vec<u16> = scores_f32.iter().copied().map(f32_to_bf16).collect();

    let expert_buf = upload(&device, &expert_out_bf);
    let scores_buf = upload(&device, &scores_bf);
    let out_buf = upload(&device, &vec![0u16; (n * hidden) as usize]);

    dispatch_moe_weighted_sum(
        &pipeline,
        &queue,
        &out_buf,
        &expert_buf,
        &scores_buf,
        n,
        top_k,
        hidden,
        2,
    )
    .expect("dispatch");

    let got_bf: Vec<u16> = download(&out_buf, (n * hidden) as usize);
    let got_f32: Vec<f32> = got_bf.iter().copied().map(bf16_to_f32).collect();

    // BF16 has ~7 bits of mantissa; the absolute scale of the output
    // here is roughly |scores * expert| × top_k ≈ 0.4 × 0.05 × 8 ≈ 0.16,
    // so an absolute tolerance of 5e-3 covers rounding through both
    // operand quantization and the bf16 store.
    let mut max_err: f32 = 0.0;
    for (g, w) in got_f32.iter().zip(want.iter()) {
        max_err = max_err.max((g - w).abs());
    }
    assert!(
        max_err < 5e-3,
        "moe_weighted_sum bf16 max abs err {max_err} exceeds 5e-3"
    );
}

#[test]
fn moe_weighted_sum_f16_small_prefill() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = MoeWeightedSumKernels::new(&device).expect("MoeWeightedSumKernels::new");

    let n: u32 = 8;
    let top_k: u32 = 4;
    let hidden: u32 = 768;

    let pipeline = kernels
        .pipeline_for(&device, MoeWeightedSumDtype::F16, top_k, hidden)
        .expect("pipeline");

    let expert_out_f32: Vec<f32> = (0..(n * top_k * hidden) as usize)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.005)
        .collect();
    let scores_f32: Vec<f32> = (0..(n * top_k) as usize)
        .map(|i| (i as f32 + 0.5) * 0.03)
        .collect();
    let want = moe_weighted_sum_cpu_f32(
        &expert_out_f32,
        &scores_f32,
        n as usize,
        top_k as usize,
        hidden as usize,
    );

    let expert_out_h: Vec<u16> = expert_out_f32
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .map(half::f16::to_bits)
        .collect();
    let scores_h: Vec<u16> = scores_f32
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .map(half::f16::to_bits)
        .collect();

    let expert_buf = upload(&device, &expert_out_h);
    let scores_buf = upload(&device, &scores_h);
    let out_buf = upload(&device, &vec![0u16; (n * hidden) as usize]);

    dispatch_moe_weighted_sum(
        &pipeline,
        &queue,
        &out_buf,
        &expert_buf,
        &scores_buf,
        n,
        top_k,
        hidden,
        2,
    )
    .expect("dispatch");

    let got_h: Vec<u16> = download(&out_buf, (n * hidden) as usize);
    let got_f32: Vec<f32> = got_h
        .iter()
        .copied()
        .map(|b| half::f16::from_bits(b).to_f32())
        .collect();

    let mut max_err: f32 = 0.0;
    for (g, w) in got_f32.iter().zip(want.iter()) {
        max_err = max_err.max((g - w).abs());
    }
    assert!(
        max_err < 1e-3,
        "moe_weighted_sum f16 max abs err {max_err} exceeds 1e-3"
    );
}
