// SPDX-License-Identifier: Apache-2.0
//
// `affine_embed_*_gs_*_b_4` parity test: kernel ≡ CPU reference.
//
// Mirrors the slow-reference math in MLX
// `nn.QuantizedEmbedding.__call__` (`python/mlx/nn/layers/quantized.py:144`):
//   y[token, col] = scale[vocab_idx, col/gs] * nibble(w[vocab_idx, col/2])
//                 + bias[vocab_idx, col/gs]
// where `vocab_idx = indices[token]` and `nibble` extracts the low / high
// 4 bits depending on parity of `col`. The CPU reference uses FMA-style
// single-rounding (f32 multiply + add, then one round to the target
// dtype), matching the GPU FMA — same convention as
// `quantized_dequantize_test.rs`.

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{DequantDtype, MetalAffineEmbed, ScaleDtype};
use ferrite_metal_kernels::stream::MetalStream;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

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

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength returned nil")
}

fn read_buffer_f16(buf: &Buffer, n_elements: usize) -> Vec<half::f16> {
    let ptr = buf.contents().as_ptr() as *const half::f16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

fn read_buffer_bf16(buf: &Buffer, n_elements: usize) -> Vec<half::bf16> {
    let ptr = buf.contents().as_ptr() as *const half::bf16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

/// CPU reference: gather-then-dequant. Each output element is
/// `scale[vocab_idx, group] * nibble + bias[vocab_idx, group]` with
/// FMA single-rounding (the GPU emits one hardware FMA per element).
fn cpu_affine_embed_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: usize,
    group_size: usize,
) -> Vec<half::f16> {
    let bytes_per_row = hidden_size / 2;
    let groups_per_row = hidden_size / group_size;
    let n_tokens = indices.len();
    let mut out = vec![half::f16::ZERO; n_tokens * hidden_size];
    for (token, &vocab_idx) in indices.iter().enumerate() {
        let vocab_idx = vocab_idx as usize;
        for byte_idx in 0..bytes_per_row {
            let byte = packed[vocab_idx * bytes_per_row + byte_idx];
            let col_base = byte_idx * 2;
            let g = vocab_idx * groups_per_row + col_base / group_size;
            let scale = scales[g].to_f32();
            let bias = biases[g].to_f32();
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            out[token * hidden_size + col_base] = half::f16::from_f32(scale * lo + bias);
            out[token * hidden_size + col_base + 1] = half::f16::from_f32(scale * hi + bias);
        }
    }
    out
}

fn cpu_affine_embed_b4_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    let bytes_per_row = hidden_size / 2;
    let groups_per_row = hidden_size / group_size;
    let n_tokens = indices.len();
    let mut out = vec![half::bf16::ZERO; n_tokens * hidden_size];
    for (token, &vocab_idx) in indices.iter().enumerate() {
        let vocab_idx = vocab_idx as usize;
        for byte_idx in 0..bytes_per_row {
            let byte = packed[vocab_idx * bytes_per_row + byte_idx];
            let col_base = byte_idx * 2;
            let g = vocab_idx * groups_per_row + col_base / group_size;
            // Match the kernel's in-register T_scale (f16) → T_act (bf16)
            // cast so the CPU ref matches within ~2 ULP of the GPU FMA.
            let scale = half::bf16::from_f32(scales[g].to_f32()).to_f32();
            let bias = half::bf16::from_f32(biases[g].to_f32()).to_f32();
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            out[token * hidden_size + col_base] = half::bf16::from_f32(scale * lo + bias);
            out[token * hidden_size + col_base + 1] = half::bf16::from_f32(scale * hi + bias);
        }
    }
    out
}

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
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn next_u32_below(&mut self, n: u32) -> u32 {
        ((self.next() >> 32) as u32) % n
    }
}

#[test]
fn affine_embed_b4_f16_kernel_matches_cpu_reference() {
    // Vocab=32 keeps the test fast; hidden_size=256 covers all three gs
    // values; n_tokens=16 exercises multi-row dispatch + 2D grid wraps.
    let vocab_size: usize = 32;
    let hidden_size: usize = 256;
    let n_tokens: usize = 16;

    for &group_size in &[32usize, 64, 128] {
        let n_bytes = vocab_size * hidden_size / 2;
        let n_groups = vocab_size * hidden_size / group_size;

        let mut rng = SplitMix64(0xC0FFEE_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        let scales: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();
        let indices: Vec<u32> = (0..n_tokens)
            .map(|_| rng.next_u32_below(vocab_size as u32))
            .collect();

        let expected =
            cpu_affine_embed_b4_f16(&packed, &scales, &biases, &indices, hidden_size, group_size);
        let metal = run_kernel_f16(
            &packed,
            &scales,
            &biases,
            &indices,
            hidden_size as u32,
            group_size as u32,
        );

        for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
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
fn affine_embed_b4_bf16_kernel_matches_cpu_reference() {
    let vocab_size: usize = 32;
    let hidden_size: usize = 256;
    let n_tokens: usize = 16;

    for &group_size in &[32usize, 64, 128] {
        let n_bytes = vocab_size * hidden_size / 2;
        let n_groups = vocab_size * hidden_size / group_size;

        let mut rng = SplitMix64(0xDEADBEEF_u64 ^ group_size as u64);
        let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
        // P10b: scales/biases ship F16 on disk.
        let scales: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
            .collect();
        let biases: Vec<half::f16> = (0..n_groups)
            .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
            .collect();
        let indices: Vec<u32> = (0..n_tokens)
            .map(|_| rng.next_u32_below(vocab_size as u32))
            .collect();

        let expected =
            cpu_affine_embed_b4_bf16(&packed, &scales, &biases, &indices, hidden_size, group_size);
        let metal = run_kernel_bf16(
            &packed,
            &scales,
            &biases,
            &indices,
            hidden_size as u32,
            group_size as u32,
        );

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

/// Exercises the partial-trailing-threadgroup bounds-check: hidden=2112
/// = 33 groups × 64 → bytes_per_row=1056 > 1024 (typical max
/// threads-per-threadgroup), forces 2 threadgroups in the X dim where
/// the second has only 32 active threads. Without the kernel's
/// `if (index.x * 2 >= hidden_size) return;` guard the trailing 992
/// threads chase past-end bytes of `w` and corrupt out[].
#[test]
fn affine_embed_b4_bf16_partial_trailing_threadgroup() {
    let vocab_size: usize = 8;
    let hidden_size: usize = 2112;
    let n_tokens: usize = 4;
    let group_size: usize = 64;

    let n_bytes = vocab_size * hidden_size / 2;
    let n_groups = vocab_size * hidden_size / group_size;

    let mut rng = SplitMix64(0xBADC0FFE);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    // P10b: scales/biases ship F16 on disk.
    let scales: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(0.1 + 0.9 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();
    let indices: Vec<u32> = (0..n_tokens)
        .map(|_| rng.next_u32_below(vocab_size as u32))
        .collect();

    let expected =
        cpu_affine_embed_b4_bf16(&packed, &scales, &biases, &indices, hidden_size, group_size);
    let metal = run_kernel_bf16(
        &packed,
        &scales,
        &biases,
        &indices,
        hidden_size as u32,
        group_size as u32,
    );

    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mb = m.to_bits() as i32;
        let eb = e.to_bits() as i32;
        assert!(
            (mb - eb).abs() <= 2,
            "bf16 unaligned hidden={hidden_size} idx={i}: \
             metal={m} (bits={mb:04x}) cpu={e} (bits={eb:04x})"
        );
    }
}

fn run_kernel_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: u32,
    group_size: u32,
) -> Vec<half::f16> {
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let embed = MetalAffineEmbed::new(device.clone()).expect("MetalAffineEmbed");

    let packed_buf = buffer_from_bytes(&device, packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let indices_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(indices.as_ptr() as *const u8, std::mem::size_of_val(indices))
    };
    let indices_buf = buffer_from_bytes(&device, indices_bytes);
    let n_out = indices.len() * hidden_size as usize;
    let out_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::f16>());

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    embed
        .execute(
            &packed_buf,
            &scales_buf,
            &biases_buf,
            &indices_buf,
            &out_buf,
            indices.len() as u32,
            hidden_size,
            group_size,
            4,
            DequantDtype::F16,
            ScaleDtype::F16,
            &encoder,
        )
        .expect("affine_embed dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    read_buffer_f16(&out_buf, n_out)
}

fn run_kernel_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    indices: &[u32],
    hidden_size: u32,
    group_size: u32,
) -> Vec<half::bf16> {
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let embed = MetalAffineEmbed::new(device.clone()).expect("MetalAffineEmbed");

    let packed_buf = buffer_from_bytes(&device, packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let indices_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(indices.as_ptr() as *const u8, std::mem::size_of_val(indices))
    };
    let indices_buf = buffer_from_bytes(&device, indices_bytes);
    let n_out = indices.len() * hidden_size as usize;
    let out_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::bf16>());

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    embed
        .execute(
            &packed_buf,
            &scales_buf,
            &biases_buf,
            &indices_buf,
            &out_buf,
            indices.len() as u32,
            hidden_size,
            group_size,
            4,
            DequantDtype::Bf16,
            ScaleDtype::F16,
            &encoder,
        )
        .expect("affine_embed dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    read_buffer_bf16(&out_buf, n_out)
}
