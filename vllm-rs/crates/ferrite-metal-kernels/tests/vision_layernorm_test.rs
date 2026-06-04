// SPDX-License-Identifier: Apache-2.0
//! Golden test for `vision_layernorm` (Qwen3.5-VL / Qwen3-VL ViT LayerNorm).
//!
//! Dispatched standalone via `SpecializedPipelineCache` (production path), f32.
//! Reference = the affine LayerNorm with biased variance + eps-inside-sqrt,
//! independently verified == `mlx.nn.LayerNorm` (max_abs_err 2.4e-7).

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

/// y = (x-mean)*rsqrt(var+eps)*w + b; var = E[x^2]-mean^2 (biased). [M, D].
fn ln_ref(x: &[f32], w: &[f32], b: &[f32], m: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; m * d];
    for r in 0..m {
        let row = &x[r * d..][..d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = (var + eps).sqrt().recip();
        for i in 0..d {
            out[r * d + i] = (row[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

#[test]
fn vision_layernorm_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let queue = device.newCommandQueue().expect("queue");
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    // Production hidden (1152), a handful of rows.
    let (m, d) = (5usize, 1152usize);
    let eps = 1e-6f32;
    let n = m * d;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017).sin() * 1.3 + 0.2)
        .collect();
    let w: Vec<f32> = (0..d)
        .map(|i| 1.0 + ((i as f32) * 0.003).cos() * 0.1)
        .collect();
    let b: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.005).sin() * 0.05).collect();

    let want = ln_ref(&x, &w, &b, m, d, eps);

    let key = PipelineKey::new(
        "vision_layernorm",
        "vision_layernorm_f32",
        vec![
            ConstantValue::uint(0, m as u32),
            ConstantValue::uint(1, d as u32),
            ConstantValue::float(2, eps),
        ],
    );
    let pipeline = cache.get_or_build(&key).expect("vision_layernorm pipeline");

    let x_buf = buf_f32(&device, &x);
    let w_buf = buf_f32(&device, &w);
    let b_buf = buf_f32(&device, &b);
    let out_buf = buf_zero_f32(&device, n);

    let cb = queue.commandBuffer().expect("commandBuffer");
    let enc = cb.computeCommandEncoder().expect("computeCommandEncoder");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 3);
    }
    // one threadgroup per row; 256 threads stride over hidden.
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m,
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
    assert!(err < 1e-4, "vision_layernorm max_abs_err={err}");
}
