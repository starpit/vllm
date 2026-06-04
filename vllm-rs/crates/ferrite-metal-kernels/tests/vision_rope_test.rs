// SPDX-License-Identifier: Apache-2.0
//! Golden test for `vision_rope_2d` (Qwen3.5-VL / Qwen3-VL ViT 2D RoPE).
//!
//! The kernel is dispatched standalone via `SpecializedPipelineCache` +
//! `get_or_build` (the production path), f32 instantiation. The reference is
//! the GPT-NeoX `rotate_half` rope from mlx-vlm
//! `qwen3_vl/vision.py::apply_rotary_pos_emb_vision`, transcribed here and
//! independently verified == the mlx-vlm golden fixture (rope_block0_q:
//! max_abs_err 9.5e-7, cosine 1.0) — see vllm-rs/tools/vision_parity.

use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

fn buf_f32(device: &Device, data: &[f32]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

fn buf_zero_f32(device: &Device, n: usize) -> Buffer {
    let buf = device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, n * 4) };
    buf
}

fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, n).to_vec() }
}

/// NeoX rotate_half rope reference: out[t,h,d] = x*cos(f) + rotate_half(x)*sin(f),
/// f = freqs[t*half + d%half]; rotate_half = concat(-x[half:], x[:half]).
fn rope_ref(x: &[f32], freqs: &[f32], l: usize, h: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let mut out = vec![0f32; l * h * d];
    for t in 0..l {
        for hh in 0..h {
            let base = (t * h + hh) * d;
            for dd in 0..d {
                let f = freqs[t * half + (dd % half)];
                let (c, s) = (f.cos(), f.sin());
                let partner = if dd < half {
                    -x[base + dd + half]
                } else {
                    x[base + dd - half]
                };
                out[base + dd] = x[base + dd] * c + partner * s;
            }
        }
    }
    out
}

#[test]
fn vision_rope_2d_matches_neox_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let queue = device.newCommandQueue().expect("queue");
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    // Small config exercising the d%half tiling + the rotate_half split.
    let (l, h, d) = (3usize, 2usize, 8usize);
    let half = d / 2;
    let n = l * h * d;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin() * 0.7).collect();
    let freqs: Vec<f32> = (0..l * half).map(|i| 0.05 + (i as f32) * 0.013).collect();

    let want = rope_ref(&x, &freqs, l, h, d);

    let key = PipelineKey::new(
        "vision_rope_2d",
        "vision_rope_2d_f32",
        vec![
            ConstantValue::uint(0, d as u32),
            ConstantValue::uint(1, h as u32),
            ConstantValue::uint(2, n as u32),
        ],
    );
    let pipeline = cache.get_or_build(&key).expect("vision_rope_2d pipeline");

    let x_buf = buf_f32(&device, &x);
    let fr_buf = buf_f32(&device, &freqs);
    let out_buf = buf_zero_f32(&device, n);

    let cb = queue.commandBuffer().expect("commandBuffer");
    let enc = cb.computeCommandEncoder().expect("computeCommandEncoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&fr_buf), 0, 2);
    }
    let tg = n.div_ceil(256);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let got = read_f32(&out_buf, n);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        err < 1e-4,
        "vision_rope_2d max_abs_err={err} (want≈{want:?} got≈{got:?})"
    );
}
