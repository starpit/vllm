// SPDX-License-Identifier: Apache-2.0
//! Parity test: moe_weighted_sum kernel vs CPU reference for the
//! MoE-decode shape (N=1, top_k=8, hidden=2048 ≈ Qwen3-MoE).

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::moe_weighted_sum::{
    dispatch_moe_weighted_sum, moe_weighted_sum_cpu_f32, MoeSumDType, MoeWeightedSumKernels,
};
use half::bf16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

fn rand_f32(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = vec![0.0_f32; n];
    for v in &mut out {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let b = ((s as u32) & 0xFFFF) as f32 / 65535.0;
        *v = (b - 0.5) * scale;
    }
    out
}

fn run_bf16(rows: usize, top_k: usize, hidden: usize, seed: u64) {
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let kernels = MoeWeightedSumKernels::new(&mdev.device).expect("kernels");

    let expert_f32 = rand_f32(rows * top_k * hidden, seed, 2.0);
    // Probability-like scores in (0, 1) summing to ~1 across top_k.
    let mut scores_f32 = rand_f32(rows * top_k, seed.wrapping_mul(31), 1.0)
        .into_iter()
        .map(|v| v.abs() + 1e-3)
        .collect::<Vec<_>>();
    for r in 0..rows {
        let s = scores_f32[r * top_k..(r + 1) * top_k].iter().sum::<f32>();
        for k in 0..top_k {
            scores_f32[r * top_k + k] /= s;
        }
    }

    let expert_bf16: Vec<bf16> = expert_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let scores_bf16: Vec<bf16> = scores_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let expert_buf = mdev
        .device
        .newBufferWithLength_options(expert_bf16.len() * 2, MTLResourceOptions::StorageModeShared)
        .expect("expert buf");
    let scores_buf = mdev
        .device
        .newBufferWithLength_options(scores_bf16.len() * 2, MTLResourceOptions::StorageModeShared)
        .expect("scores buf");
    let out_buf = mdev
        .device
        .newBufferWithLength_options(rows * hidden * 2, MTLResourceOptions::StorageModeShared)
        .expect("out buf");

    unsafe {
        std::ptr::copy_nonoverlapping(
            expert_bf16.as_ptr() as *const u8,
            expert_buf.contents().as_ptr() as *mut u8,
            expert_bf16.len() * 2,
        );
        std::ptr::copy_nonoverlapping(
            scores_bf16.as_ptr() as *const u8,
            scores_buf.contents().as_ptr() as *mut u8,
            scores_bf16.len() * 2,
        );
    }

    dispatch_moe_weighted_sum(
        &kernels,
        &queue,
        &expert_buf,
        &scores_buf,
        &out_buf,
        rows as u32,
        top_k as u32,
        hidden as u32,
        MoeSumDType::BF16,
    )
    .expect("dispatch");

    let mut got = vec![bf16::ZERO; rows * hidden];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents().as_ptr() as *const u8,
            got.as_mut_ptr() as *mut u8,
            got.len() * 2,
        );
    }

    let expert_f32_round: Vec<f32> = expert_bf16.iter().map(|v| v.to_f32()).collect();
    let scores_f32_round: Vec<f32> = scores_bf16.iter().map(|v| v.to_f32()).collect();
    let mut want = vec![0.0_f32; rows * hidden];
    moe_weighted_sum_cpu_f32(
        &expert_f32_round,
        &scores_f32_round,
        &mut want,
        rows,
        top_k,
        hidden,
    );

    let mut max_err = 0.0_f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let e = (g.to_f32() - w).abs();
        if e > max_err {
            max_err = e;
        }
    }
    // bf16 accumulation through float fma is ~ulp at the typical
    // |Σ_k probability * expert_out| ≈ ~1 magnitudes; tolerance
    // tracks bf16's mantissa precision (~3 decimal digits) plus
    // top_k accumulation.
    assert!(
        max_err < 5e-2,
        "moe_weighted_sum_bf16 rows={rows} top_k={top_k} hidden={hidden} max_err={max_err}"
    );
}

#[test]
fn moe_weighted_sum_bf16_mixtral_decode() {
    run_bf16(/*rows=*/ 1, /*top_k=*/ 2, /*hidden=*/ 4096, 0xC0FFEE);
}

#[test]
fn moe_weighted_sum_bf16_qwen3_moe_decode() {
    run_bf16(/*rows=*/ 1, /*top_k=*/ 8, /*hidden=*/ 2048, 0xDEAD_BEEF);
}

#[test]
fn moe_weighted_sum_bf16_prefill_batch() {
    run_bf16(/*rows=*/ 32, /*top_k=*/ 8, /*hidden=*/ 2048, 0x12345);
}
