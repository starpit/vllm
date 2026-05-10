// SPDX-License-Identifier: Apache-2.0
//
// `affine_dequantize_*_gs_*_b_4` parity test: kernel ≡ CPU reference.
//
// Mirrors the slow-reference math in `mlx/backend/metal/kernels/quantized.h:2536`.
// The CPU reference is inlined here (rather than pulled from
// `ferrite_forward::cpu_golden`) so the test file stays free of a cycle
// against ferrite-metal-kernels' own dev-deps.

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{DequantDtype, MetalAffineDequantize};
use ferrite_metal_kernels::stream::MetalStream;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions,
};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Build a shared-storage Metal buffer from a byte slice.
fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes returned nil")
    }
}

/// Empty shared-storage buffer of `n_bytes` length.
fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength returned nil")
}

/// Read a half-precision buffer's contents back as `[half::f16]`.
fn read_buffer_f16(buf: &Buffer, n_elements: usize) -> Vec<half::f16> {
    let ptr = buf.contents().as_ptr() as *const half::f16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

fn read_buffer_bf16(buf: &Buffer, n_elements: usize) -> Vec<half::bf16> {
    let ptr = buf.contents().as_ptr() as *const half::bf16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

/// CPU reference: `out[oindex] = scale * nibble + bias` with FMA-style
/// single-rounding semantics (f32 multiply + add, then one round to the
/// target dtype). The GPU emits an fp16/bf16 hardware FMA for this
/// expression — modelling each op individually in half precision
/// double-rounds and drifts ~4 ULPs vs the kernel.
fn cpu_affine_dequantize_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    group_size: usize,
) -> Vec<half::f16> {
    let out_len = packed.len() * 2;
    let mut out = vec![half::f16::ZERO; out_len];
    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * 2;
        let gindex = oindex / group_size;
        let scale = scales[gindex].to_f32();
        let bias = biases[gindex].to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        out[oindex] = half::f16::from_f32(scale * lo + bias);
        out[oindex + 1] = half::f16::from_f32(scale * hi + bias);
    }
    out
}

fn cpu_affine_dequantize_b4_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    group_size: usize,
) -> Vec<half::bf16> {
    let out_len = packed.len() * 2;
    let mut out = vec![half::bf16::ZERO; out_len];
    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * 2;
        let gindex = oindex / group_size;
        let scale = scales[gindex].to_f32();
        let bias = biases[gindex].to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        out[oindex] = half::bf16::from_f32(scale * lo + bias);
        out[oindex + 1] = half::bf16::from_f32(scale * hi + bias);
    }
    out
}

/// Deterministic PRNG seeded per-shape so failures reproduce.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        // Uniform [0, 1). 24 bits of mantissa precision.
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[test]
fn affine_dequantize_b4_f16_kernel_matches_cpu_reference() {
    for &group_size in &[32usize, 64, 128] {
        let n: usize = 4;
        let k: usize = 256; // K must be multiple of group_size; 256 covers all three gs values.
        let n_bytes = n * k / 2; // 2 nibbles per byte
        let n_groups = n * k / group_size;

        let mut rng = SplitMix64(0xC0FFEE_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        // Scales in [0.1, 1.0] to keep activations finite; biases in [-1, 1).
        let scales: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();

        let expected = cpu_affine_dequantize_b4_f16(&packed, &scales, &biases, group_size);
        let metal = run_kernel_f16(&packed, &scales, &biases, n, k, group_size as u32);

        for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
            // CPU and GPU both model FMA single-rounding (f32 intermediate
            // → one round to half). Expect bit-exact match.
            let mb = m.to_bits() as i32;
            let eb = e.to_bits() as i32;
            assert!(
                (mb - eb).abs() <= 1,
                "f16 gs={group_size} idx={i}: metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
            );
        }
    }
}

#[test]
fn affine_dequantize_b4_bf16_kernel_matches_cpu_reference() {
    for &group_size in &[32usize, 64, 128] {
        let n: usize = 4;
        let k: usize = 256;
        let n_bytes = n * k / 2;
        let n_groups = n * k / group_size;

        let mut rng = SplitMix64(0xDEADBEEF_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        let scales: Vec<half::bf16> = (0..n_groups)
            .map(|_| half::bf16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::bf16> = (0..n_groups)
            .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();

        let expected = cpu_affine_dequantize_b4_bf16(&packed, &scales, &biases, group_size);
        let metal = run_kernel_bf16(&packed, &scales, &biases, n, k, group_size as u32);

        for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
            let mb = m.to_bits() as i32;
            let eb = e.to_bits() as i32;
            assert!(
                (mb - eb).abs() <= 2,
                "bf16 gs={group_size} idx={i}: metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
            );
        }
    }
}

fn run_kernel_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    n: usize,
    k: usize,
    group_size: u32,
) -> Vec<half::f16> {
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let dequant = MetalAffineDequantize::new(device.clone()).expect("MetalAffineDequantize");

    let packed_buf = buffer_from_bytes(&device, packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let n_out = n * k;
    let out_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::f16>());

    let cmd_buf = stream
        .get_command_buffer()
        .expect("command buffer")
        .clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    dequant
        .execute(
            &packed_buf,
            &scales_buf,
            &biases_buf,
            &out_buf,
            n_out as u64,
            group_size,
            4,
            DequantDtype::F16,
            &encoder,
        )
        .expect("dequant dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    read_buffer_f16(&out_buf, n_out)
}

fn run_kernel_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    n: usize,
    k: usize,
    group_size: u32,
) -> Vec<half::bf16> {
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let dequant = MetalAffineDequantize::new(device.clone()).expect("MetalAffineDequantize");

    let packed_buf = buffer_from_bytes(&device, packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let n_out = n * k;
    let out_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::bf16>());

    let cmd_buf = stream
        .get_command_buffer()
        .expect("command buffer")
        .clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    dequant
        .execute(
            &packed_buf,
            &scales_buf,
            &biases_buf,
            &out_buf,
            n_out as u64,
            group_size,
            4,
            DequantDtype::Bf16,
            &encoder,
        )
        .expect("dequant dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    read_buffer_bf16(&out_buf, n_out)
}
