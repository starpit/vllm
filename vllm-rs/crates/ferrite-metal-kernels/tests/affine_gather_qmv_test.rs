// SPDX-License-Identifier: Apache-2.0
//! Parity test for `affine_gather_qmv_fast` / `affine_gather_qmv`:
//! each (token, slot) gather should yield the same result as a
//! plain `qmm_t` over the gathered expert's weight slab.

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::cpu_reference::{affine_qmv_b4_bf16, affine_qmv_b4_bf16_s_bf16};
use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{DequantDtype, MetalAffineGatherQmv, ScaleDtype};
use half::{bf16, f16};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions,
};

fn buf_from_bytes(
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>,
    bytes: &[u8],
) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes nil")
    }
}

fn zeros_buf(
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>,
    n: usize,
) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength nil")
}

fn read_bf16(
    buf: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>,
    n: usize,
) -> Vec<bf16> {
    let ptr = buf.contents().as_ptr() as *const bf16;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn splitmix(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn affine_gather_qmv_bf16_mixtral_decode_fast() {
    // Shape: num_experts=8, n_out=512, k=512 (mini), top_k=2, M=3.
    // n_out=512 multiple of 8, k=512 multiple of 512 → Fast variant.
    let num_experts = 8usize;
    let n_out = 512usize;
    let k = 512usize;
    let group_size = 64usize;
    let top_k = 2usize;
    let num_tokens = 3usize;

    let mut seed = 0xABCD_EF01_2345_6789u64;

    // Pack the per-expert weight slab.
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = (splitmix(&mut seed) as u8) & 0xFF;
    }
    let mut scales_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = f16::from_f32(r * 0.05);
    }
    let mut biases_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = f16::from_f32((r - 0.5) * 0.1);
    }

    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }

    // Indices: each token picks `top_k` distinct experts; choices
    // span the expert range.
    let indices: Vec<u32> = vec![
        0, 5, // token 0
        2, 7, // token 1
        1, 4, // token 2
    ];

    let mdev = detect_device().expect("device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let executor = MetalAffineGatherQmv::new(mdev.device.clone()).expect("gather qmv");

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_f16 =
        |s: &[f16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };

    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&scales_f16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&biases_f16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    let cmdbuf = queue.commandBuffer().expect("cmdbuf");
    let enc = cmdbuf.computeCommandEncoder().expect("enc");
    executor
        .execute(
            &x_buf,
            &w_buf,
            &s_buf,
            &b_buf,
            &idx_buf,
            &y_buf,
            num_tokens as u32,
            top_k as u32,
            n_out as u32,
            k as u32,
            group_size as u32,
            4,
            DequantDtype::Bf16,
            ScaleDtype::F16,
            &enc,
        )
        .expect("execute");
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);

    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 3e-2 || (g - w).abs() / w.abs().max(1e-3) < 3e-2,
                    "n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!(
        "affine_gather_qmv_bf16 num_tokens={num_tokens} top_k={top_k} n={n_out} k={k} max_err={:.3e}",
        max_err
    );
}

#[test]
fn affine_gather_qmv_bf16_generic() {
    // Trigger Generic kernel: k=384 (not %512), n=384.
    let num_experts = 4usize;
    let n_out = 384usize;
    let k = 384usize;
    let group_size = 64usize;
    let top_k = 2usize;
    let num_tokens = 2usize;

    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = (splitmix(&mut seed) as u8) & 0xFF;
    }
    let mut scales_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = f16::from_f32(r * 0.05);
    }
    let mut biases_f16 = vec![f16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_f16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = f16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    let indices: Vec<u32> = vec![0, 3, 1, 2];

    let mdev = detect_device().expect("device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let executor = MetalAffineGatherQmv::new(mdev.device.clone()).expect("gather qmv");

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_f16 =
        |s: &[f16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };

    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&scales_f16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of_f16(&biases_f16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    let cmdbuf = queue.commandBuffer().expect("cmdbuf");
    let enc = cmdbuf.computeCommandEncoder().expect("enc");
    executor
        .execute(
            &x_buf,
            &w_buf,
            &s_buf,
            &b_buf,
            &idx_buf,
            &y_buf,
            num_tokens as u32,
            top_k as u32,
            n_out as u32,
            k as u32,
            group_size as u32,
            4,
            DequantDtype::Bf16,
            ScaleDtype::F16,
            &enc,
        )
        .expect("execute");
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);

    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_f16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                assert!(
                    err < 4e-2 || (g - w).abs() / w.abs().max(1e-3) < 4e-2,
                    "generic n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// BF16 SCALES — Qwen3-MoE-30B-A3B-Instruct ships BF16 scales/biases
// for `switch_mlp.{gate,up,down}_proj.{scales,biases}` (probed across
// mlx-community/Qwen3-30B-A3B-4bit). Validates
// `ScaleDtype::Bf16` end-to-end against a Qwen3-MoE-shaped gather.
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_decode_fast() {
    // Shape mirrors Qwen3-MoE-30B-A3B inner block at decode: M=1
    // (one decoded token), num_experts=128, top_k=8, hidden=2048,
    // moe_intermediate=768. We shrink num_experts to 8 and use the
    // gate_proj shape (n_out=moe_intermediate, k=hidden) at a
    // group_size of 64 (Qwen3 mlx-affine-b4-g64).
    let num_experts = 8usize;
    let n_out = 768usize; // moe_intermediate (multiple of 8)
    let k = 2048usize; // hidden (multiple of 512)
    let group_size = 64usize;
    let top_k = 4usize; // top-4 like Mixtral, fine for parity
    let num_tokens = 1usize; // M=1 decode

    let mut seed = 0xC0DE_BA5E_DEAD_BEEFu64;

    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = (splitmix(&mut seed) as u8) & 0xFF;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    let indices: Vec<u32> = vec![0, 1, 5, 7];

    let mdev = detect_device().expect("device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let executor = MetalAffineGatherQmv::new(mdev.device.clone()).expect("gather qmv");

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    let cmdbuf = queue.commandBuffer().expect("cmdbuf");
    let enc = cmdbuf.computeCommandEncoder().expect("enc");
    executor
        .execute(
            &x_buf,
            &w_buf,
            &s_buf,
            &b_buf,
            &idx_buf,
            &y_buf,
            num_tokens as u32,
            top_k as u32,
            n_out as u32,
            k as u32,
            group_size as u32,
            4,
            DequantDtype::Bf16,
            ScaleDtype::Bf16,
            &enc,
        )
        .expect("execute");
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16_s_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            // K=2048 inflates bf16 accumulation noise: stdev grows as
            // sqrt(K) * bf16_eps * per-elem-magnitude. With bf16_eps ≈
            // 6e-3, per-elem ≈ 0.25, K=2048 → stdev ≈ 0.07. Allowed
            // tolerance ≈ 4× stdev (matches the
            // `worst_abs_error_vs_noise_floor` shape in qmv tests).
            let allowed_abs = 0.3_f32;
            let allowed_rel = 0.20_f32;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                    "qwen3-shape n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!(
        "affine_gather_qmv_bf16_s_bf16 qwen3 num_tokens={num_tokens} top_k={top_k} n={n_out} k={k} max_err={:.3e}",
        max_err
    );
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_down_proj_shape() {
    // Reproduces the exact Qwen3-MoE-30B-A3B down_proj shape that fails
    // end-to-end at dispatch 24: N=hidden=2048, K=moe_inter=768, gs=64.
    // K=768 is NOT a multiple of 512 → generic (non-fast) variant.
    // K=2048 (gate/up_proj) IS mult-of-512 → fast variant; that one
    // is already covered by `qwen3_moe_decode_fast`.
    let num_experts = 8usize;
    let n_out = 2048usize;
    let k = 768usize;
    let group_size = 64usize;
    let top_k = 4usize;
    let num_tokens = 1usize;

    let mut seed = 0xDEAD_C0DE_BABE_F00Du64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = (splitmix(&mut seed) as u8) & 0xFF;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; num_tokens * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    let indices: Vec<u32> = vec![0, 3, 5, 7];

    let mdev = detect_device().expect("device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let executor = MetalAffineGatherQmv::new(mdev.device.clone()).expect("gather qmv");

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, num_tokens * top_k * n_out * 2);

    let cmdbuf = queue.commandBuffer().expect("cmdbuf");
    let enc = cmdbuf.computeCommandEncoder().expect("enc");
    executor
        .execute(
            &x_buf,
            &w_buf,
            &s_buf,
            &b_buf,
            &idx_buf,
            &y_buf,
            num_tokens as u32,
            top_k as u32,
            n_out as u32,
            k as u32,
            group_size as u32,
            4,
            DequantDtype::Bf16,
            ScaleDtype::Bf16,
            &enc,
        )
        .expect("execute");
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let got = read_bf16(&y_buf, num_tokens * top_k * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for n in 0..num_tokens {
        for slot in 0..top_k {
            let expert = indices[n * top_k + slot] as usize;
            let want = affine_qmv_b4_bf16_s_bf16(
                &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
                &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
                &x_bf16[n * k..(n + 1) * k],
                1,
                n_out,
                k,
                group_size,
            );
            let base = (n * top_k + slot) * n_out;
            let allowed_abs = 0.3_f32; // K=768 with bf16 accumulation tolerance
            let allowed_rel = 0.25_f32;
            for c in 0..n_out {
                let g = got[base + c].to_f32();
                let w = want[c].to_f32();
                let err = (g - w).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                    "down_proj_shape n={n} slot={slot} expert={expert} c={c} got={g} want={w} err={err}",
                );
            }
        }
    }
    eprintln!("affine_gather_qmv down_proj shape: max_err={:.3e}", max_err);
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_down_proj_top_k_1_workaround() {
    // Production lowering passes `top_k=1` to the kernel for the
    // down-projection step (handoff workaround in
    // `lower_metal_moe`). x layout is `[T*top_k, moe_inter]` — one
    // row per (token, slot), not per-token broadcast. rhs_indices
    // has `T*top_k` entries.
    //
    // This test reproduces that invocation EXACTLY: top_k_in_kernel=1,
    // x has one row per (token, slot), and the CPU reference iterates
    // the same way.
    let num_experts = 8usize;
    let n_out = 2048usize;
    let k = 768usize;
    let group_size = 64usize;
    let real_top_k = 4usize;
    let num_tokens = 2usize;
    // Effective rows = T*top_k. The kernel sees this as M with top_k=1.
    let m_rows = num_tokens * real_top_k;

    let mut seed = 0x1357_9BDF_2468_ACE0u64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = (splitmix(&mut seed) as u8) & 0xFF;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    // x is m_rows rows of k elements — one per (token, slot).
    let mut x_bf16 = vec![bf16::ZERO; m_rows * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    // indices has m_rows entries — one expert id per row.
    let indices: Vec<u32> = vec![0, 3, 5, 7, 1, 2, 4, 6];

    let mdev = detect_device().expect("device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let executor = MetalAffineGatherQmv::new(mdev.device.clone()).expect("gather qmv");

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, m_rows * n_out * 2);

    let cmdbuf = queue.commandBuffer().expect("cmdbuf");
    let enc = cmdbuf.computeCommandEncoder().expect("enc");
    // The KEY: invoke with `top_k = 1` (workaround) so the kernel
    // reads one x row per nk and one expert per nk. num_tokens is
    // set to m_rows = T*top_k so total dispatched rows = m_rows.
    executor
        .execute(
            &x_buf,
            &w_buf,
            &s_buf,
            &b_buf,
            &idx_buf,
            &y_buf,
            m_rows as u32, // num_tokens, but in workaround we pass T*top_k
            1u32,          // top_k_in_kernel = 1 (workaround)
            n_out as u32,
            k as u32,
            group_size as u32,
            4,
            DequantDtype::Bf16,
            ScaleDtype::Bf16,
            &enc,
        )
        .expect("execute");
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let got = read_bf16(&y_buf, m_rows * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    for nk in 0..m_rows {
        let expert = indices[nk] as usize;
        let want = affine_qmv_b4_bf16_s_bf16(
            &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
            &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &x_bf16[nk * k..(nk + 1) * k],
            1,
            n_out,
            k,
            group_size,
        );
        let base = nk * n_out;
        let allowed_abs = 0.3_f32;
        let allowed_rel = 0.25_f32;
        for c in 0..n_out {
            let g = got[base + c].to_f32();
            let w = want[c].to_f32();
            let err = (g - w).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < allowed_abs || err / w.abs().max(1e-3) < allowed_rel,
                "top_k_1 nk={nk} expert={expert} c={c} got={g} want={w} err={err}",
            );
        }
    }
    eprintln!(
        "affine_gather_qmv top_k=1 workaround down_proj: max_err={:.3e}",
        max_err
    );
}

#[test]
fn affine_gather_qmv_bf16_s_bf16_qwen3_moe_down_proj_production_dispatch() {
    // Reproduces the EXACT production dispatch shape that diverges at
    // dispatch 24 in `vllm chat`:
    //
    //   tg = (1, n_out/8 = 256, bucket_m * top_k = 72)
    //   threads = (32, 2, 1)
    //
    // Qwen3-30B-A3B-Instruct has top_k=8; bucket_m=9 (prefill chunk
    // with 9 tokens) gives m_rows = 72. The production lowering uses
    // the top_k=1 workaround: the kernel sees one x row per (token,
    // slot) and reads one expert per nk.
    //
    // The smaller parity test (`top_k_1_workaround`) uses m_rows=8
    // (tg.z=8) which left the tg.y=256 × tg.z=72 path unexercised.
    // If this test passes, the integration bug is NOT in the kernel
    // — look at binding/weight resolution. If it fails, the kernel
    // has a scale-dependent bug (threadgroup memory / simd ordering /
    // dispatch grid bounds).
    let num_experts = 128usize;
    let n_out = 2048usize;
    let k = 768usize;
    let group_size = 64usize;
    let m_rows = 72usize; // bucket_m=9 * top_k=8

    let mut seed = 0xCAFE_F00D_8BAD_F00Du64;
    let mut packed = vec![0u8; num_experts * n_out * k / 2];
    for b in &mut packed {
        *b = (splitmix(&mut seed) as u8) & 0xFF;
    }
    let mut scales_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for s in &mut scales_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0 + 0.001;
        *s = bf16::from_f32(r * 0.05);
    }
    let mut biases_bf16 = vec![bf16::ZERO; num_experts * n_out * k / group_size];
    for b in &mut biases_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *b = bf16::from_f32((r - 0.5) * 0.1);
    }
    let mut x_bf16 = vec![bf16::ZERO; m_rows * k];
    for v in &mut x_bf16 {
        let r = (splitmix(&mut seed) % 1000) as f32 / 1000.0;
        *v = bf16::from_f32((r - 0.5) * 2.0);
    }
    // Indices: pick experts spanning the 128-expert range to match the
    // bisect-observed `[53, 116, 70, 29, 16, 19, 58, 49, ...]` pattern.
    let indices: Vec<u32> = (0..m_rows)
        .map(|i| ((i * 53 + 7) % num_experts) as u32)
        .collect();

    let mdev = detect_device().expect("device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let executor = MetalAffineGatherQmv::new(mdev.device.clone()).expect("gather qmv");

    let bytes_of =
        |s: &[bf16]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) };
    let bytes_of_u32 =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
    let w_buf = buf_from_bytes(&mdev.device, &packed);
    let s_buf = buf_from_bytes(&mdev.device, bytes_of(&scales_bf16));
    let b_buf = buf_from_bytes(&mdev.device, bytes_of(&biases_bf16));
    let x_buf = buf_from_bytes(&mdev.device, bytes_of(&x_bf16));
    let idx_buf = buf_from_bytes(&mdev.device, bytes_of_u32(&indices));
    let y_buf = zeros_buf(&mdev.device, m_rows * n_out * 2);

    let cmdbuf = queue.commandBuffer().expect("cmdbuf");
    let enc = cmdbuf.computeCommandEncoder().expect("enc");
    executor
        .execute(
            &x_buf,
            &w_buf,
            &s_buf,
            &b_buf,
            &idx_buf,
            &y_buf,
            m_rows as u32, // num_tokens param == m_rows (workaround)
            1u32,          // top_k_in_kernel = 1 (workaround)
            n_out as u32,
            k as u32,
            group_size as u32,
            4,
            DequantDtype::Bf16,
            ScaleDtype::Bf16,
            &enc,
        )
        .expect("execute");
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let got = read_bf16(&y_buf, m_rows * n_out);
    let packed_per_expert = n_out * k / 2;
    let sb_per_expert = n_out * k / group_size;
    let mut max_err = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut fail_count = 0usize;
    for nk in 0..m_rows {
        let expert = indices[nk] as usize;
        let want = affine_qmv_b4_bf16_s_bf16(
            &packed[expert * packed_per_expert..(expert + 1) * packed_per_expert],
            &scales_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &biases_bf16[expert * sb_per_expert..(expert + 1) * sb_per_expert],
            &x_bf16[nk * k..(nk + 1) * k],
            1,
            n_out,
            k,
            group_size,
        );
        let base = nk * n_out;
        let allowed_abs = 0.3_f32;
        let allowed_rel = 0.25_f32;
        for c in 0..n_out {
            let g = got[base + c].to_f32();
            let w = want[c].to_f32();
            let err = (g - w).abs();
            let rel = err / w.abs().max(1e-3);
            if err > max_err {
                max_err = err;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            if err > allowed_abs && rel > allowed_rel {
                fail_count += 1;
                if fail_count < 8 {
                    eprintln!(
                        "FAIL nk={nk} expert={expert} c={c} got={g} want={w} err={err} rel={rel}",
                    );
                }
            }
        }
    }
    eprintln!(
        "affine_gather_qmv production dispatch m_rows={m_rows}: max_err={max_err:.3e} \
         max_rel={max_rel:.3e} fail_count={fail_count}",
    );
    assert_eq!(
        fail_count, 0,
        "production dispatch produced {fail_count} out-of-tolerance elements"
    );
}
