// SPDX-License-Identifier: Apache-2.0
//! Parity test: gate_scale kernel vs CPU reference for the Qwen3.5-MoE
//! shared-expert combine `out = routed + shared_y * sigmoid(g[row])`,
//! with `g` `[rows, 1]` row-broadcast. Shapes mirror 35B-A3B decode
//! (rows=1, hidden=2048) and a prefill bucket (rows=64).

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::gate_scale::{
    dispatch_gate_scale, gate_scale_cpu_f32, GateScaleDType, GateScaleKernels,
};
use half::{bf16, f16};
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

fn upload(
    dev: &ferrite_metal_kernels::device::MetalDevice,
    bytes: &[u8],
) -> ferrite_metal_kernels::gate_scale::Buffer {
    let buf = dev
        .device
        .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModeShared)
        .expect("buffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buf.contents().as_ptr() as *mut u8,
            bytes.len(),
        );
    }
    buf
}

fn run(rows: usize, cols: usize, seed: u64, dtype: GateScaleDType) {
    let mdev = detect_device().expect("metal device");
    let queue = mdev.device.newCommandQueue().expect("queue");
    let kernels = GateScaleKernels::new(&mdev.device).expect("gate_scale kernels");

    let routed_f32 = rand_f32(rows * cols, seed, 4.0);
    let shared_f32 = rand_f32(rows * cols, seed.wrapping_mul(7919), 4.0);
    // Gate logits in a realistic pre-sigmoid range.
    let gate_f32 = rand_f32(rows, seed.wrapping_mul(31), 8.0);

    // Round inputs through the storage dtype so the CPU reference sees
    // exactly what the kernel reads.
    let (routed_b, shared_b, gate_b, routed_r, shared_r, gate_r): (
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) = match dtype {
        GateScaleDType::F16 => {
            let r: Vec<f16> = routed_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let s: Vec<f16> = shared_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let g: Vec<f16> = gate_f32.iter().map(|&v| f16::from_f32(v)).collect();
            (
                bytemuck_cast(&r),
                bytemuck_cast(&s),
                bytemuck_cast(&g),
                r.iter().map(|v| v.to_f32()).collect(),
                s.iter().map(|v| v.to_f32()).collect(),
                g.iter().map(|v| v.to_f32()).collect(),
            )
        }
        GateScaleDType::BF16 => {
            let r: Vec<bf16> = routed_f32.iter().map(|&v| bf16::from_f32(v)).collect();
            let s: Vec<bf16> = shared_f32.iter().map(|&v| bf16::from_f32(v)).collect();
            let g: Vec<bf16> = gate_f32.iter().map(|&v| bf16::from_f32(v)).collect();
            (
                bytemuck_cast(&r),
                bytemuck_cast(&s),
                bytemuck_cast(&g),
                r.iter().map(|v| v.to_f32()).collect(),
                s.iter().map(|v| v.to_f32()).collect(),
                g.iter().map(|v| v.to_f32()).collect(),
            )
        }
    };

    let routed_buf = upload(&mdev, &routed_b);
    let shared_buf = upload(&mdev, &shared_b);
    let gate_buf = upload(&mdev, &gate_b);
    let out_buf = mdev
        .device
        .newBufferWithLength_options(rows * cols * 2, MTLResourceOptions::StorageModeShared)
        .expect("out buf");

    dispatch_gate_scale(
        &kernels,
        &queue,
        &routed_buf,
        &shared_buf,
        &gate_buf,
        &out_buf,
        rows as u32,
        cols as u32,
        dtype,
    )
    .expect("gate_scale dispatch");

    let mut want = vec![0.0_f32; rows * cols];
    gate_scale_cpu_f32(&routed_r, &shared_r, &gate_r, &mut want, rows, cols);

    let got_f32: Vec<f32> = unsafe {
        let p = out_buf.contents().as_ptr() as *const u16;
        (0..rows * cols)
            .map(|i| match dtype {
                GateScaleDType::F16 => f16::from_bits(*p.add(i)).to_f32(),
                GateScaleDType::BF16 => bf16::from_bits(*p.add(i)).to_f32(),
            })
            .collect()
    };

    // One storage-dtype rounding of the f32 result.
    let tol = match dtype {
        GateScaleDType::F16 => 1e-2,
        GateScaleDType::BF16 => 8e-2,
    };
    for i in 0..rows * cols {
        let d = (got_f32[i] - want[i]).abs();
        assert!(
            d <= tol,
            "({rows}x{cols} {dtype:?}) idx {i} (row {}): got {} want {} |d|={d}",
            i / cols,
            got_f32[i],
            want[i]
        );
    }
}

/// Raw byte view of a host slice for buffer upload. Callers must
/// pass slices whose element size matches the kernel-side dtype —
/// asserted here so a mismatched `T` fails loudly instead of
/// producing a wrong-length upload.
fn bytemuck_cast<T: Copy>(v: &[T]) -> Vec<u8> {
    assert!(
        matches!(std::mem::size_of::<T>(), 2 | 4),
        "gate_scale uploads are 2-byte (f16/bf16) or 4-byte (f32) elements"
    );
    let bytes = std::mem::size_of_val(v);
    let mut out = vec![0u8; bytes];
    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, out.as_mut_ptr(), bytes);
    }
    out
}

#[test]
fn gate_scale_bf16_decode_row() {
    // M=1 decode, hidden=2048 (Qwen3.5-MoE-35B-A3B).
    run(1, 2048, 0xC0FFEE, GateScaleDType::BF16);
}

#[test]
fn gate_scale_bf16_prefill_bucket() {
    // M=64 prefill bucket — exercises the row-broadcast across many rows.
    run(64, 2048, 0xBEEF_F00D, GateScaleDType::BF16);
}

#[test]
fn gate_scale_f16_decode_row() {
    run(1, 2048, 0x5EED, GateScaleDType::F16);
}

#[test]
fn gate_scale_f16_prefill_bucket() {
    run(64, 2048, 0xFEED_FACE, GateScaleDType::F16);
}
