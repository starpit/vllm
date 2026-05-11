// SPDX-License-Identifier: Apache-2.0
//
// `affine_qvm_*` and `affine_qvm_split_k_*` parity tests: kernel ≡
// CPU reference within bf16 accumulation noise.
//
// MLX dispatches qvm vs qvm_split_k based on K:
//   K < 1024              → qvm           (standard matvec)
//   K >= 1024, K <= 8192  → qvm_split_k 8 (8 K-partitions)
//   K >  8192             → qvm_split_k 32 (32 K-partitions)
//
// All three paths exercise `qvm_impl` byte-for-byte; the splitk
// wrapper just runs the same kernel against per-partition slices
// and the caller sum-reduces the `[split_k, M, N]` intermediate
// (matching `quantized.cpp:415 strided_reduce_general_dispatch`).
//
// W storage convention for transpose=false: `[K, N / pack_factor]`
// u32 (last axis = N is quantized). Scales/biases: `[K, N / gs]`.

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{pick_qvm_kernel, DequantDtype, MetalAffineQvm, QvmKernel};
use ferrite_metal_kernels::stream::MetalStream;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

// ─────────────────────────────────────────────────────────────────
// Test helpers (mirror tests/quantized_qmm_n_test.rs and
// tests/quantized_qmv_test.rs)
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

fn make_inputs_bf16(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<half::bf16>, Vec<half::bf16>, Vec<half::bf16>) {
    assert_eq!(n % group_size, 0, "qvm requires N % group_size == 0");
    let n_bytes = k * n / 2;
    let n_groups = k * n / group_size;

    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    (packed, scales, biases, x)
}

/// CPU reference: dequant + dense matmul for transpose=false (W is
/// `[K, N]` row-major). Same math as qmm_n.
fn cpu_qvm_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    let mut w = vec![half::bf16::ZERO; k * n];
    for kk in 0..k {
        for byte_j in 0..(n / 2) {
            let byte = packed[kk * (n / 2) + byte_j];
            let n_col = 2 * byte_j;
            let group_idx = kk * (n / group_size) + n_col / group_size;
            let scale = scales[group_idx].to_f32();
            let bias = biases[group_idx].to_f32();
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            w[kk * n + n_col] = half::bf16::from_f32(scale * lo + bias);
            w[kk * n + n_col + 1] = half::bf16::from_f32(scale * hi + bias);
        }
    }
    let mut y = vec![half::bf16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                acc += x[i * k + kk].to_f32() * w[kk * n + j].to_f32();
            }
            y[i * n + j] = half::bf16::from_f32(acc);
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn run_qvm_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    expected_kernel: QvmKernel,
) -> Vec<half::bf16> {
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let qvm = MetalAffineQvm::new(device.clone()).expect("MetalAffineQvm");

    let packed_buf = buffer_from_bytes(&device, packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);

    let (out_elements, split_k) = match expected_kernel {
        QvmKernel::Standard => (m * n, 1u32),
        QvmKernel::SplitK { split_k, .. } => (split_k as usize * m * n, split_k),
    };
    let y_buf = zeroed_buffer(&device, out_elements * std::mem::size_of::<half::bf16>());

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    let actual_kernel = qvm
        .execute(
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
        .expect("qvm dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    match (actual_kernel, expected_kernel) {
        (QvmKernel::Standard, QvmKernel::Standard) => {}
        (QvmKernel::SplitK { split_k: a, .. }, QvmKernel::SplitK { split_k: e, .. }) if a == e => {}
        _ => panic!(
            "dispatcher picked {:?}, expected {:?}",
            actual_kernel, expected_kernel
        ),
    }

    let raw = read_buffer_bf16(&y_buf, out_elements);
    if split_k == 1 {
        raw
    } else {
        // Sum-reduce along axis 0 of the [split_k, M, N] partial.
        let mut out = vec![half::bf16::ZERO; m * n];
        for p in 0..split_k as usize {
            for i in 0..m {
                for j in 0..n {
                    let cur = out[i * n + j].to_f32();
                    let add = raw[p * m * n + i * n + j].to_f32();
                    out[i * n + j] = half::bf16::from_f32(cur + add);
                }
            }
        }
        out
    }
}

fn worst_abs_error_vs_noise_floor(
    metal: &[half::bf16],
    expected: &[half::bf16],
    k_dim: usize,
    per_elem_magnitude: f32,
) -> (usize, f32, f32, f32, f32) {
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
// qvm standard (K < 1024) — pure single-partition matvec
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qvm_b4_bf16_standard_k_small_matches_cpu_reference() {
    // M=1 (decode), N=512 (= 8*bn for bn=64, hits 8 N-tiles),
    // K=512 (< 1024 → Standard). gs=64.
    let m = 1;
    let n = 512;
    let k = 512;
    let group_size = 64;

    let expected_kernel = pick_qvm_kernel(k as u32);
    assert_eq!(
        expected_kernel,
        QvmKernel::Standard,
        "K<1024 should route to Standard"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xD1D1_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qvm_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qvm_bf16(
        &packed,
        &scales,
        &biases,
        &x,
        m,
        n,
        k,
        group_size as u32,
        expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qvm Standard gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qvm_split_k (K=2048) — 8 partitions; tests the per-partition
// pointer-shift math in affine_qvm_split_k_kernel.
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qvm_b4_bf16_split_k8_k_2048_matches_cpu_reference() {
    let m = 1;
    let n = 512;
    let k = 2048;
    let group_size = 64;

    let expected_kernel = pick_qvm_kernel(k as u32);
    assert!(
        matches!(expected_kernel, QvmKernel::SplitK { split_k: 8, .. }),
        "K=2048 should route to SplitK split_k=8; got {expected_kernel:?}"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xD2D2_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qvm_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qvm_bf16(
        &packed,
        &scales,
        &biases,
        &x,
        m,
        n,
        k,
        group_size as u32,
        expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qvm SplitK(8) K=2048: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qvm_split_k (K=8192, gs=128) — 8 partitions on the K=8192
// boundary (`K > 8192 ? 32 : 8` → 8 since K == 8192).
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qvm_b4_bf16_split_k8_k_8192_matches_cpu_reference() {
    let m = 1;
    let n = 256;
    let k = 8192;
    let group_size = 128;

    let expected_kernel = pick_qvm_kernel(k as u32);
    assert!(
        matches!(expected_kernel, QvmKernel::SplitK { split_k: 8, .. }),
        "K=8192 should route to SplitK split_k=8 (K > 8192 ? 32 : 8); got {expected_kernel:?}"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xD3D3_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qvm_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qvm_bf16(
        &packed,
        &scales,
        &biases,
        &x,
        m,
        n,
        k,
        group_size as u32,
        expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qvm SplitK(8) K=8192: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qvm_split_k (K=16384) — 32 partitions, exercises the larger
// split_k value and confirms the final-block-size handling agrees
// with MLX for K divisible by split_k.
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qvm_b4_bf16_split_k32_k_16384_matches_cpu_reference() {
    let m = 1;
    let n = 256;
    let k = 16384;
    let group_size = 128;

    let expected_kernel = pick_qvm_kernel(k as u32);
    assert!(
        matches!(expected_kernel, QvmKernel::SplitK { split_k: 32, .. }),
        "K=16384 should route to SplitK split_k=32 (K > 8192); got {expected_kernel:?}"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xD4D4_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qvm_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qvm_bf16(
        &packed,
        &scales,
        &biases,
        &x,
        m,
        n,
        k,
        group_size as u32,
        expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qvm SplitK(32) K=16384: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qvm Standard — M=2 (smoke that the M-row indexing also works
// at M>1, even though the dispatcher rule M < vector_limit=4 makes
// this rare on decode).
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qvm_b4_bf16_standard_m_2_matches_cpu_reference() {
    let m = 2;
    let n = 256;
    let k = 256;
    let group_size = 32;

    let expected_kernel = pick_qvm_kernel(k as u32);
    assert_eq!(expected_kernel, QvmKernel::Standard);

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xD5D5_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qvm_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qvm_bf16(
        &packed,
        &scales,
        &biases,
        &x,
        m,
        n,
        k,
        group_size as u32,
        expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qvm Standard M=2 gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}
