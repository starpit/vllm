// SPDX-License-Identifier: Apache-2.0
//
// `affine_qmv_*_<dtype>_gs_<gs>_b_4_*` parity test: kernel ≡ CPU reference.
//
// Mirrors the slow-reference math: dequantize the packed int4 weight per
// MLX `affine_dequantize` (`quantized.h:2536` — also covered by the
// pre-existing `quantized_dequantize_test`), then do a row-major matmul
// with f32 accumulator and bf16/f16 cast on the way out. That reference
// matches the qmv kernel's algebra: `result = scale * sum(x * nibble) +
// sum_of_x * bias`, which simplifies to `sum(x * (scale * nibble + bias))`.
//
// Three tests, one per dispatched kernel:
//   - qmv_quad:    K∈{64, 128} ∧ pow2 bits → tested with K=128 (Llama-style head_dim)
//   - qmv_fast:    N%8==0 ∧ K%512==0       → tested with N=64, K=512
//   - qmv generic: neither                  → tested with N=12, K=384

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{
    pick_qmv_kernel, DequantDtype, MetalAffineQmv, QmvKernel,
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions,
};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

// ─────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────

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

fn read_buffer_bf16(buf: &Buffer, n_elements: usize) -> Vec<half::bf16> {
    let ptr = buf.contents().as_ptr() as *const half::bf16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

/// SplitMix64 — deterministic per-shape PRNG so failures reproduce.
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
}

/// Generate `(packed, scales, biases, x)` for a `(N, K)` weight + `(M, K)`
/// activation in bf16. Random nibbles, scales in [0.01, 0.05] (small to
/// keep the dequantized weights close to MLX's typical scale range), and
/// biases in [-0.5, 0.5].
fn make_inputs_bf16(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<half::bf16>, Vec<half::bf16>, Vec<half::bf16>) {
    assert_eq!(k % group_size, 0);
    let n_bytes = n * k / 2; // pack_factor = 2 nibbles/byte
    let n_groups = n * k / group_size;

    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    // Activations in [-1, 1) — typical residual-stream magnitude.
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    (packed, scales, biases, x)
}

/// CPU reference: dequantize w → bf16 [N, K], then `y = x @ w^T`
/// accumulating in f32 and casting per-output to bf16.
fn cpu_qmv_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    // Dequantize the full weight tile [N, K] using the cpu_golden formula
    // inlined locally (so this test stays self-contained).
    let mut w = vec![half::bf16::ZERO; n * k];
    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * 2;
        let gindex = oindex / group_size;
        let scale = scales[gindex].to_f32();
        let bias = biases[gindex].to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        w[oindex] = half::bf16::from_f32(scale * lo + bias);
        w[oindex + 1] = half::bf16::from_f32(scale * hi + bias);
    }
    // Matmul: y[i, j] = sum_k(x[i, k] * w[j, k]) — transpose=true.
    let mut y = vec![half::bf16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                acc += x[i * k + kk].to_f32() * w[j * k + kk].to_f32();
            }
            y[i * n + j] = half::bf16::from_f32(acc);
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn run_qmv_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> Vec<half::bf16> {
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let qmv = MetalAffineQmv::new(device.clone()).expect("MetalAffineQmv");

    let packed_buf = buffer_from_bytes(&device, packed);

    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let x_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x))
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);

    let n_out = m * n;
    let y_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::bf16>());

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    qmv.execute(
        &x_buf,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &y_buf,
        m as u32,
        n as u32,
        k as u32,
        1, /* B */
        group_size,
        4,
        DequantDtype::Bf16,
        &encoder,
    )
    .expect("qmv dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    read_buffer_bf16(&y_buf, n_out)
}

/// Worst per-element noise in f32 space. We compare against the expected
/// bf16 accumulation noise floor: with `K` terms summed in f32 and cast
/// to bf16 at the end, the per-output noise std is bounded by
/// `sqrt(K) * eps_bf16 * max(|x|) * max(|w|)` plus a single-step bf16
/// rounding (~`|y| * 2^-7`). The test bounds the per-element absolute
/// error against that noise floor with a 4× safety factor — tighter
/// would catch a real bug, looser would miss systematic drift.
///
/// Returns `(index, metal_value, cpu_value, abs_err, allowed)` for the
/// worst element so failure messages point at the actual mismatch.
// `per_elem_magnitude` is a bound on `|x| * |w_dequant|` per element.
// With x ∈ [-1, 1) and w_dequant magnitude ≲ 0.5 (scale ≲ 0.05,
// |nibble * scale + bias| ≲ 1.25), per-element product magnitude ≲ ~0.5.
fn worst_abs_error_vs_noise_floor(
    metal: &[half::bf16],
    expected: &[half::bf16],
    k_dim: usize,
    per_elem_magnitude: f32,
) -> (usize, f32, f32, f32, f32) {
    // bf16 mantissa eps ≈ 2^-7 ≈ 7.8e-3 (relative). Per-element rounding
    // bounded by half-eps * magnitude. Sum-of-K-terms noise std grows by
    // sqrt(K). Final bf16 cast adds another ~|y| * 2^-7.
    let bf16_eps: f32 = 1.0 / 128.0;
    let sum_noise_std = (k_dim as f32).sqrt() * 0.5 * bf16_eps * per_elem_magnitude;
    let safety = 4.0;
    let mut worst = (0usize, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mf = m.to_f32();
        let ef = e.to_f32();
        let abs_err = (mf - ef).abs();
        let allowed = safety * (sum_noise_std + ef.abs() * bf16_eps);
        if abs_err > worst.3 {
            worst = (i, mf, ef, abs_err, allowed);
        }
    }
    worst
}

// ─────────────────────────────────────────────────────────────────
// Kernel-pick smoke: confirm the dispatcher routes our test shapes
// to the variant we're trying to exercise.
// ─────────────────────────────────────────────────────────────────

#[test]
fn qmv_dispatcher_routes_test_shapes_correctly() {
    // qmv_quad sample: K=128
    assert!(matches!(
        pick_qmv_kernel(64, 128, 4),
        QmvKernel::Quad { d: 128 }
    ));
    // qmv_fast sample: K=512, N=64
    assert_eq!(pick_qmv_kernel(64, 512, 4), QmvKernel::Fast);
    // qmv generic sample: K=384, N=12
    assert_eq!(pick_qmv_kernel(12, 384, 4), QmvKernel::Generic);
}

// ─────────────────────────────────────────────────────────────────
// qmv_quad parity
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmv_quad_b4_bf16_matches_cpu_reference() {
    // K=128 hits the qmv_quad path. N=64 covers a full quadgroup row
    // (8 outputs/quadgroup × 8 quads/simd = 64 outputs/threadgroup).
    let m = 1;
    let n = 64;
    let k = 128;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) = make_inputs_bf16(0xCAFE_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
        let metal = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32);

        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv_quad gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// qmv_fast parity
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmv_fast_b4_bf16_matches_cpu_reference() {
    // N%8==0 ∧ K%512==0 → qmv_fast. K=512 = 1 simd block.
    let m = 1;
    let n = 64;
    let k = 512;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) = make_inputs_bf16(0xBEEF_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
        let metal = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32);

        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv_fast gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// qmv generic parity (unaligned tail path)
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmv_generic_b4_bf16_matches_cpu_reference() {
    // N=12 (not a multiple of 8) ∧ K=384 (not a multiple of 512) →
    // qmv generic with the bounds-checked tail load. K must still be a
    // multiple of group_size; 384 / {32,64,128} all divide.
    let m = 1;
    let n = 12;
    let k = 384;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) = make_inputs_bf16(0xFACE_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
        let metal = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32);

        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv generic gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}
