// SPDX-License-Identifier: Apache-2.0
//
// `affine_qmm_t_*` and `affine_qmm_t_splitk_*` parity tests:
// kernel ≡ CPU reference within bf16 accumulation noise.
//
// CPU reference: dequantize the packed int4 weight per MLX
// `affine_dequantize` math (4-bit branch — same `(scale, bias)` per
// `group_size` columns), then do a row-major matmul with f32
// accumulator and bf16 cast on the way out.
//
// Three tests:
//   - qmm_t aligned (N % 32 == 0): hits the unsafe-load fast path
//   - qmm_t unaligned (N % 32 != 0): hits the bounds-checked tail
//   - qmm_t_splitk (B=1, small-M): the dispatcher's small-prefill
//     path; output is the [split_k, M, N] intermediate which we
//     sum-reduce in CPU (mirroring `quantized.cpp:861
//     strided_reduce_general_dispatch`).

use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{
    pick_qmm_t_kernel, DequantDtype, MetalAffineQmmT, QmmTKernel,
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

// ─────────────────────────────────────────────────────────────────
// Test helpers (mirror tests/quantized_qmv_test.rs)
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

/// SplitMix64 — same deterministic PRNG as the qmv test.
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
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    (packed, scales, biases, x)
}

/// Full CPU matmul reference (M × K) @ (N × K)^T → (M × N), bf16 in/out
/// with f32 accumulation. Equivalent semantics to dequant-then-matmul.
fn cpu_qmm_t_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<half::bf16> {
    // Dequantize the full weight tile [N, K]. Inline the 4-bit
    // dequant formula (`quantized.h:521-527` for the per-byte case;
    // we walk packed-byte order which gives 2 nibbles / byte).
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
    // y[i, j] = sum_k(x[i, k] * w[j, k])  (transpose=true).
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
fn run_qmm_t_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    expected_kernel: QmmTKernel,
) -> Vec<half::bf16> {
    let device = detect_device().expect("Metal device").device;
    let mut stream = MetalStream::new(&device);
    let qmm = MetalAffineQmmT::new(device.clone()).expect("MetalAffineQmmT");

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

    // Output sizing differs for splitk: kernel writes [split_k, M, N]
    // intermediate, caller sum-reduces along axis 0.
    let (out_elements, split_k) = match expected_kernel {
        QmmTKernel::Standard => (m * n, 1u32),
        QmmTKernel::SplitK { split_k, .. } => (split_k as usize * m * n, split_k),
    };
    let y_buf = zeroed_buffer(&device, out_elements * std::mem::size_of::<half::bf16>());

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    let actual_kernel = qmm
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
        .expect("qmm_t dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    // Confirm the dispatcher picked the variant we set up for. If a
    // future heuristic change reroutes a test shape, this surfaces
    // the divergence cleanly rather than silently masking it.
    match (actual_kernel, expected_kernel) {
        (QmmTKernel::Standard, QmmTKernel::Standard) => {}
        (QmmTKernel::SplitK { split_k: a, .. }, QmmTKernel::SplitK { split_k: e, .. })
            if a == e => {}
        _ => panic!(
            "dispatcher picked {:?}, expected {:?}",
            actual_kernel, expected_kernel
        ),
    }

    let raw = read_buffer_bf16(&y_buf, out_elements);
    if split_k == 1 {
        raw
    } else {
        // Sum-reduce along the split_k axis: out[i, j] = sum_p raw[p, i, j],
        // then bf16-cast (matches `strided_reduce_general_dispatch` semantics
        // for sum-along-axis-0 on a [split_k, M, N] tensor).
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

/// Per-element bf16 accumulation noise bound, parameterized on K and
/// the per-element product magnitude. Same shape as the qmv test.
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
// qmm_t aligned-N parity — fast unsafe-load path
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_t_aligned_b4_bf16_matches_cpu_reference() {
    // M=64 ≥ vector_limit, B=1, N=64 (% 32 == 0 so aligned_N=true).
    // K=256, gs ∈ {32, 64, 128} all divide. M_tiles=2, N_tiles=2.
    // pick_qmm_t_split_k: tgs = 4, target = 128 → split_k capped by
    // K/gs (= 8/4/2) — for gs=32, splitk picks min(128, 8)=8 → SplitK.
    // To keep this test on the *Standard* (qmm_t) path, choose a
    // shape where splitk collapses to 1: B>1 doesn't apply (B=1 is
    // our wired regime), so use M_tiles × N_tiles large enough that
    // target_split_k = 512 / tgs ≤ 1. M=512, N=64 → tgs = 32 → ts =
    // 16 → splitk wins. Conclusion: with B=1 there's no shape with
    // K%gs==0 that *avoids* SplitK at small N. So this test exercises
    // qmm_t by passing the kernel directly through the inner
    // template instantiation — we force Standard by using a shape
    // with split_k collapsing to 1.
    //
    // Easiest such shape: M=512, N=2048, K=512 → tgs = 16*64 = 1024
    // → target_split_k = 0 → 1 → Standard.
    let m = 512;
    let n = 2048;
    let k = 512;
    let group_size = 64;

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32);
    assert_eq!(
        expected_kernel,
        QmmTKernel::Standard,
        "test shape should route to Standard"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xA1A1_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qmm_t_bf16(
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
        "qmm_t aligned gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qmm_t unaligned-N parity — bounds-checked tail path
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_t_unaligned_b4_bf16_matches_cpu_reference() {
    // N=50 → not a multiple of 32 → aligned_N=false. tgs is computed
    // from N_tiles = ceil(50/32) = 2; M=512 → M_tiles = 16 → tgs =
    // 32 → target_split_k = 16. K/gs = 512/64 = 8 → split_k=8 → that
    // would route to SplitK. Pick a shape that defeats splitk: K=64,
    // gs=64 → K/gs = 1 → splitk capped at 1 → Standard.
    let m = 64;
    let n = 50;
    let k = 64;
    let group_size = 64;

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32);
    assert_eq!(
        expected_kernel,
        QmmTKernel::Standard,
        "test shape should route to Standard"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xB2B2_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qmm_t_bf16(
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
        "qmm_t unaligned gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qmm_t_splitk parity — small-prefill split-K + sum-reduce
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_t_splitk_b4_bf16_matches_cpu_reference() {
    // M=64, N=64 (aligned), K=2048, gs=64 → tgs = 4 → target = 128
    //   → cap K/gs = 32 → 32. K % (32*64) = 0 → split_k=32.
    let m = 64;
    let n = 64;
    let k = 2048;
    let group_size = 64;

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32);
    assert!(
        matches!(expected_kernel, QmmTKernel::SplitK { .. }),
        "test shape should route to SplitK, got {:?}",
        expected_kernel
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xC3C3_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qmm_t_bf16(
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

    // SplitK has an additional sum step in bf16 (32 partial outputs);
    // each partial sums K/split_k = 64 K terms in f32 → bf16-cast →
    // we then sum 32 bf16 partials into a final bf16. Worst-case
    // total noise per cell is `sqrt(split_k) * bf16_eps * |partial|`
    // additional, on top of the per-partial noise.
    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    // SplitK gives bigger noise budgets: ~2× looser to cover the
    // post-kernel bf16 sum we emulate in CPU.
    let allowed = allowed * 2.0;
    assert!(
        abs_err <= allowed,
        "qmm_t_splitk gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}
