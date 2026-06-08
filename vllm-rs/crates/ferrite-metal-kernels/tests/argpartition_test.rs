// SPDX-License-Identifier: Apache-2.0
//! Argsort kernel parity test — compare against a CPU stable sort
//! (ties broken by index) for the MoE router shapes Mixtral (E=8),
//! Qwen2-MoE (E=60), Qwen3-MoE (E=128), Qwen3.5-MoE (E=256, the
//! bn=64 / N_PER_BLOCK=256 instantiation).
//!
//! Top-k semantics: the trailing `top_k` indices of the sorted-
//! ascending output are the indices of the top-k entries by value.

use ferrite_metal_kernels::argpartition::{dispatch_argsort, ArgsortDType, ArgsortKernels};
use ferrite_metal_kernels::device::detect_device;
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

fn cpu_argsort_ascending_f32(input: &[f32], rows: usize, cols: usize) -> Vec<u32> {
    let mut out = vec![0u32; rows * cols];
    for r in 0..rows {
        let mut idx: Vec<usize> = (0..cols).collect();
        let base = r * cols;
        // Stable sort ascending; ties broken by index (matches the
        // MLX block_sort stable-equality behavior).
        idx.sort_by(|&a, &b| {
            input[base + a]
                .partial_cmp(&input[base + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, &j) in idx.iter().enumerate() {
            out[base + i] = j as u32;
        }
    }
    out
}

/// One parity case at any router dtype. Host values are generated
/// as f32; for the half dtypes the GPU input is the cast values and
/// the CPU reference sorts the SAME rounded values (so reference and
/// kernel see identical keys).
fn run_case(dtype: ArgsortDType, rows: usize, cols: usize, top_k: usize, seed: u64) {
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let kernels = ArgsortKernels::new(&mdev.device).expect("argsort kernels");

    let host = fill_random_f32(rows, cols, seed);
    // (gpu input bytes, the values the CPU reference must sort)
    let (in_host_bytes, cmp_host): (Vec<u8>, Vec<f32>) = match dtype {
        ArgsortDType::F32 => {
            let mut bytes = vec![0u8; host.len() * 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            (bytes, host.clone())
        }
        ArgsortDType::F16 => {
            let cast: Vec<f16> = host.iter().map(|&v| f16::from_f32(v)).collect();
            let mut bytes = vec![0u8; cast.len() * 2];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cast.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            (bytes, cast.iter().map(|v| v.to_f32()).collect())
        }
        ArgsortDType::Bf16 => {
            let cast: Vec<bf16> = host.iter().map(|&v| bf16::from_f32(v)).collect();
            let mut bytes = vec![0u8; cast.len() * 2];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cast.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
            (bytes, cast.iter().map(|v| v.to_f32()).collect())
        }
        other => unreachable!("router-prob parity cases are float-typed, got {other:?}"),
    };

    let in_buf = mdev
        .device
        .newBufferWithLength_options(in_host_bytes.len(), MTLResourceOptions::StorageModeShared)
        .expect("in buf");
    let out_bytes = rows * cols * 4;
    let out_buf = mdev
        .device
        .newBufferWithLength_options(out_bytes, MTLResourceOptions::StorageModeShared)
        .expect("out buf");

    unsafe {
        std::ptr::copy_nonoverlapping(
            in_host_bytes.as_ptr(),
            in_buf.contents().as_ptr() as *mut u8,
            in_host_bytes.len(),
        );
    }

    dispatch_argsort(
        &kernels,
        &queue,
        &in_buf,
        &out_buf,
        rows as u32,
        cols as u32,
        dtype,
    )
    .expect("argsort dispatch");

    let mut got = vec![0u32; rows * cols];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents().as_ptr() as *const u8,
            got.as_mut_ptr() as *mut u8,
            out_bytes,
        );
    }

    let want = cpu_argsort_ascending_f32(&cmp_host, rows, cols);

    // Compare trailing top-k slot by slot: those should be exactly
    // the top-k indices (stable order). The interior of the sort
    // can disagree on equal-value runs, but we don't care about
    // those for MoE.
    for r in 0..rows {
        let base = r * cols;
        let mut got_topk: Vec<u32> = got[base + cols - top_k..base + cols].to_vec();
        let mut want_topk: Vec<u32> = want[base + cols - top_k..base + cols].to_vec();
        // Stable sort to make comparison set-wise; the relative
        // order within the top-k can legitimately differ from the
        // CPU stable sort for ties.
        got_topk.sort();
        want_topk.sort();
        assert_eq!(
            got_topk, want_topk,
            "row {r}: {dtype:?} top-{top_k} indices differ \
             (got={got_topk:?}, want={want_topk:?})"
        );
    }
}

#[test]
fn argsort_f32_mixtral_topk() {
    // E=8, top_k=2 (Mixtral default).
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 11,
        /*cols=*/ 8,
        /*top_k=*/ 2,
        0xC0FFEE,
    );
}

#[test]
fn argsort_f32_qwen2_moe_topk() {
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 7,
        /*cols=*/ 60,
        /*top_k=*/ 4,
        0xBEEF_F00D,
    );
}

#[test]
fn argsort_f32_qwen3_moe_topk() {
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 13,
        /*cols=*/ 128,
        /*top_k=*/ 8,
        0xDEAD_BEEF,
    );
}

#[test]
fn argsort_f16_qwen3_moe_topk() {
    // f16 router path at E=128.
    run_case(
        ArgsortDType::F16,
        /*rows=*/ 13,
        /*cols=*/ 128,
        /*top_k=*/ 8,
        0x5EED,
    );
}

#[test]
fn argsort_f32_qwen3_5_moe_topk_bn64() {
    // E=256, top_k=8 (Qwen3.5-MoE-35B-A3B) — exercises the bn=64
    // (N_PER_BLOCK=256) instantiation via pick_pipeline_shape.
    run_case(
        ArgsortDType::F32,
        /*rows=*/ 9,
        /*cols=*/ 256,
        /*top_k=*/ 8,
        0xFEED_FACE,
    );
}

#[test]
fn argsort_f16_qwen3_5_moe_topk_bn64() {
    // E=256 f16 — the W::METAL_DTYPE=F16 router path at bn=64.
    run_case(
        ArgsortDType::F16,
        /*rows=*/ 9,
        /*cols=*/ 256,
        /*top_k=*/ 8,
        0xACE0_F16,
    );
}

#[test]
fn argsort_bf16_qwen3_5_moe_topk_bn64() {
    // E=256 bf16 — the W::METAL_DTYPE=Bf16 router path at bn=64
    // (Qwen3.x MLX checkpoints carry bf16 scales → bf16 router probs).
    run_case(
        ArgsortDType::Bf16,
        /*rows=*/ 9,
        /*cols=*/ 256,
        /*top_k=*/ 8,
        0xBF16_CAFE,
    );
}
