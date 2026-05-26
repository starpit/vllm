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

use ferrite_metal_kernels::cpu_reference::affine_qmm_t_b4_bf16 as cpu_qmm_t_bf16;
use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{
    pick_qmm_t_kernel, DequantDtype, MetalAffineQmmT, QmmTKernel, ScaleDtype,
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
) -> (Vec<u8>, Vec<half::f16>, Vec<half::f16>, Vec<half::bf16>) {
    assert_eq!(k % group_size, 0);
    let n_bytes = n * k / 2; // pack_factor = 2 nibbles/byte
    let n_groups = n * k / group_size;

    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    // P10b: scales/biases ship F16 on disk (`T_scale = half`).
    let scales: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    (packed, scales, biases, x)
}

// CPU reference is shared with qmv (same math, transpose=true) — see
// `ferrite_metal_kernels::cpu_reference::affine_qmm_t_b4_bf16`,
// imported above as `cpu_qmm_t_bf16`.

#[allow(clippy::too_many_arguments)]
fn run_qmm_t_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
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
        QmmTKernel::Nax => (m * n, 1u32),
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
            ScaleDtype::F16,
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
        (QmmTKernel::Nax, QmmTKernel::Nax) => {}
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

    let expected_kernel =
        pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
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

    let expected_kernel =
        pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
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
// Llama-3.2-1B-4bit prefill-bucket parity tests. Each linear in the
// model routes to one of (Standard, SplitK) at the prefill bucket
// (bucket_m = 64 in the macro's workload-point set). The standalone
// tests above don't exercise these specific shapes, so a divergence
// at e.g. gate_proj's Standard path would slip through.
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmm_t_b4_bf16_llama_3_2_1b_q_proj_prefill_shape() {
    // q_proj / o_proj at bucket_m=64: M=64, N=2048, K=2048, gs=64.
    // pick: tgs = 64*2 = 128, split_k = 4 → SplitK(4, k_partition=512).
    let m = 64;
    let n = 2048;
    let k = 2048;
    let group_size = 64;

    let expected_kernel =
        pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert!(
        matches!(expected_kernel, QmmTKernel::SplitK { split_k: 4, .. }),
        "Llama-1B q_proj prefill shape should route to SplitK(4), got {:?}",
        expected_kernel
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11B_u64 ^ group_size as u64, n, k, m, group_size);
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
    let allowed = allowed * 2.0;
    assert!(
        abs_err <= allowed,
        "qmm_t Llama-1B q_proj SplitK (M=64, N=2048, K=2048, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

#[test]
fn affine_qmm_t_b4_bf16_llama_3_2_1b_kv_proj_prefill_shape() {
    // k_proj / v_proj at bucket_m=64: M=64, N=512, K=2048, gs=64.
    // pick: tgs = 16*2 = 32, split_k = 16 → SplitK(16, k_partition=128).
    let m = 64;
    let n = 512;
    let k = 2048;
    let group_size = 64;

    let expected_kernel =
        pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert!(
        matches!(expected_kernel, QmmTKernel::SplitK { split_k: 16, .. }),
        "Llama-1B kv_proj prefill shape should route to SplitK(16), got {:?}",
        expected_kernel
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11C_u64 ^ group_size as u64, n, k, m, group_size);
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
    let allowed = allowed * 2.0;
    assert!(
        abs_err <= allowed,
        "qmm_t Llama-1B kv_proj SplitK (M=64, N=512, K=2048, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

#[test]
fn affine_qmm_t_b4_bf16_llama_3_2_1b_gate_up_prefill_shape() {
    // gate_proj / up_proj at bucket_m=64: M=64, N=8192, K=2048, gs=64.
    // pick: tgs = 256*2 = 512, split_k = 1 → Standard.
    let m = 64;
    let n = 8192;
    let k = 2048;
    let group_size = 64;

    let expected_kernel =
        pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert_eq!(
        expected_kernel,
        QmmTKernel::Standard,
        "Llama-1B gate/up_proj prefill shape should route to Standard"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11D_u64 ^ group_size as u64, n, k, m, group_size);
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
        "qmm_t Llama-1B gate/up_proj Standard (M=64, N=8192, K=2048, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

#[test]
fn affine_qmm_t_b4_bf16_llama_3_2_1b_down_proj_prefill_shape() {
    // down_proj at bucket_m=64: M=64, N=2048, K=8192, gs=64.
    // pick: tgs = 64*2 = 128, split_k = 4 → SplitK(4, k_partition=2048).
    let m = 64;
    let n = 2048;
    let k = 8192;
    let group_size = 64;

    let expected_kernel =
        pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert!(
        matches!(expected_kernel, QmmTKernel::SplitK { split_k: 4, .. }),
        "Llama-1B down_proj prefill shape should route to SplitK(4), got {:?}",
        expected_kernel
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11E_u64 ^ group_size as u64, n, k, m, group_size);
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
    let allowed = allowed * 2.0;
    assert!(
        abs_err <= allowed,
        "qmm_t Llama-1B down_proj SplitK (M=64, N=2048, K=8192, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} (allowed {allowed:.5}; metal={mv}, cpu={ev})"
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

    let expected_kernel =
        pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
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

// ─────────────────────────────────────────────────────────────────
// qmm_t NAX (Apple9 / M4+) parity — MetalPerformancePrimitives
// matmul2d hardware MMA path; compares against the same CPU
// reference as Standard / SplitK.
//
// Gated to runtime: the test silently passes on non-NAX hardware
// (M1/M2/M3) where the kernel can't be loaded. Compares against the
// CPU reference, not Standard, since both should match within bf16
// noise but accumulating different per-element FMA orderings shifts
// the result independently.
// ─────────────────────────────────────────────────────────────────

/// Captured reproducer for the open NAX layout bug. Marked `#[ignore]`
/// so CI doesn't fail; run with
///   `cargo test -p ferrite-metal-kernels --test quantized_qmm_test \
///        affine_qmm_t_nax_b4_bf16_matches_cpu_reference -- --ignored --nocapture`
/// to see the every-other-column-unwritten diagnostic dump.
///
/// Symptom: with our shader inlining MLX's vendored `nax.h` exactly,
/// the MetalPerformancePrimitives matmul2d destination cooperative
/// tensor's per-thread layout has the data we set / read with
/// `ct_c[i]` partially landing in MPP "invalid element" slots. The
/// resulting output has every odd column of each 16×16 destination
/// frag untouched (= 0 from buffer init). See `lowering.rs`'s
/// FERRITE_ENABLE_NAX gate.
#[test]
#[ignore = "NAX cooperative-tensor layout bug — reproducer only; see lowering.rs comment"]
fn affine_qmm_t_nax_b4_bf16_matches_cpu_reference() {
    // Skip on non-M4 hardware (look at the auto-detected device
    // profile; same gate the lowering pass uses).
    let dev = ferrite_metal_kernels::device::detect_device().expect("Metal device");
    let is_nax =
        ferrite_metal_kernels::ferrite_metal_targets::is_nax_capable(dev.profile.generation);
    if !is_nax {
        eprintln!(
            "skipping NAX parity test on non-NAX-capable hardware ({:?})",
            dev.profile.generation
        );
        return;
    }

    // Shape that hits NAX: M ≥ 64 (prefill bucket), N % 64 == 0,
    // K % 64 == 0, gs ∈ {64, 128}. Use a small shape so the kernel
    // bug (if any) is isolated to a single output tile.
    let m = 64;
    let n = 64;
    let k = 64;
    let group_size = 64;

    let (packed, scales, biases, x) =
        make_inputs_bf16(0xD4D4_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);

    // Force NAX kernel via execute_with_kernel.
    let device = ferrite_metal_kernels::device::detect_device()
        .expect("Metal device")
        .device;
    let mut stream = ferrite_metal_kernels::stream::MetalStream::new(&device);
    let qmm = ferrite_metal_kernels::quantized::MetalAffineQmmT::new(device.clone())
        .expect("MetalAffineQmmT");

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            scales.as_ptr() as *const u8,
            std::mem::size_of_val(&scales[..]),
        )
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            biases.as_ptr() as *const u8,
            std::mem::size_of_val(&biases[..]),
        )
    };
    let x_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(&x[..]))
    };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);
    let y_buf = zeroed_buffer(&device, m * n * std::mem::size_of::<half::bf16>());

    let cmd_buf = stream.get_command_buffer().expect("command buffer").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    qmm.execute_with_kernel(
        &x_buf,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &y_buf,
        m as u32,
        n as u32,
        k as u32,
        1,
        group_size as u32,
        4,
        DequantDtype::Bf16,
        ScaleDtype::F16,
        QmmTKernel::Nax,
        &encoder,
    )
    .expect("nax dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let metal = read_buffer_bf16(&y_buf, m * n);

    // Diagnostic: count mismatches per simdgroup (BM=BN=64 single tile, WM=WN=2).
    let mut sg_bad = [0usize; 4];
    let mut sg_ok = [0usize; 4];
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let abs_err = (metal[idx].to_f32() - expected[idx].to_f32()).abs();
            let allowed = 4.0
                * ((k as f32).sqrt() * 0.5 * (1.0 / 128.0) * 0.5
                    + expected[idx].to_f32().abs() * (1.0 / 128.0));
            // simd_gid: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
            let sg = (i / 32) * 2 + (j / 32);
            if abs_err > allowed {
                sg_bad[sg] += 1;
            } else {
                sg_ok[sg] += 1;
            }
        }
    }
    eprintln!(
        "simdgroup mismatches: TL(0)={}/{} TR(1)={}/{} BL(2)={}/{} BR(3)={}/{}",
        sg_bad[0],
        sg_ok[0] + sg_bad[0],
        sg_bad[1],
        sg_ok[1] + sg_bad[1],
        sg_bad[2],
        sg_ok[2] + sg_bad[2],
        sg_bad[3],
        sg_ok[3] + sg_bad[3]
    );

    // Print first row of each simdgroup, metal vs cpu
    for &sg_row in &[0, 32] {
        for &sg_col in &[0, 32] {
            let i = sg_row;
            let j = sg_col;
            let pairs: Vec<String> = (0..8)
                .map(|d| {
                    format!(
                        "[{}]m={:.2}/c={:.2}",
                        j + d,
                        metal[i * n + j + d].to_f32(),
                        expected[i * n + j + d].to_f32()
                    )
                })
                .collect();
            eprintln!("row{} col{}+: {}", i, j, pairs.join(" "));
        }
    }

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmm_t NAX gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

/// Diagnostic: time the qmm_t kernel at Llama-3.2-3B prefill shapes
/// in isolation (one command buffer per call, sync between). Compares
/// against the CSV's recorded numbers to see whether the in-isolation
/// per-call time matches what the cost CSV says — if it does, the
/// "5× slower than MLX" prefill gap must live in the *chain* of
/// dispatches (per-dispatch overhead, barriers, scheduler), NOT in
/// the kernel code itself.
#[test]
#[ignore = "Timing diagnostic — run manually with --ignored --nocapture"]
fn qmm_t_timing_at_prefill_shapes() {
    use std::time::Instant;
    let device = ferrite_metal_kernels::device::detect_device()
        .expect("Metal device")
        .device;
    let mut stream = ferrite_metal_kernels::stream::MetalStream::new(&device);
    let qmm = ferrite_metal_kernels::quantized::MetalAffineQmmT::new(device.clone())
        .expect("MetalAffineQmmT");

    let shapes = [
        (
            "Q/O (M=1024, N=3072, K=3072)",
            1024_usize,
            3072_usize,
            3072_usize,
        ),
        ("K/V (M=1024, N=1024, K=3072)", 1024, 1024, 3072),
        ("Gate/Up (M=1024, N=8192, K=3072)", 1024, 8192, 3072),
        ("Down (M=1024, N=3072, K=8192)", 1024, 3072, 8192),
    ];
    let gs: u32 = 64;
    let warmups: u32 = 5;
    let iters: u32 = 20;

    for (label, m, n, k) in shapes {
        let packed_bytes = (n * k) / 2;
        let scales_elems = (n * k) / gs as usize;
        let scales_bytes = scales_elems * 2;
        let biases_bytes = scales_bytes;
        let x_bytes = m * k * 2;
        let y_bytes = m * n * 2 * 32; // worst-case SplitK 32-partition intermediate

        let packed = zeroed_buffer(&device, packed_bytes);
        let scales = zeroed_buffer(&device, scales_bytes);
        let biases = zeroed_buffer(&device, biases_bytes);
        let x = zeroed_buffer(&device, x_bytes);
        let y = zeroed_buffer(&device, y_bytes);

        let mut run = || {
            let cb = stream.get_command_buffer().expect("cmd buf").clone();
            let enc = cb.computeCommandEncoder().expect("encoder");
            qmm.execute(
                &x,
                &packed,
                &scales,
                &biases,
                &y,
                m as u32,
                n as u32,
                k as u32,
                1,
                gs,
                4,
                ferrite_metal_kernels::quantized::DequantDtype::Bf16,
                ferrite_metal_kernels::quantized::ScaleDtype::F16,
                &enc,
            )
            .expect("qmm_t dispatch");
            enc.endEncoding();
            stream.commit().expect("commit");
            stream.synchronize().expect("sync");
        };

        for _ in 0..warmups {
            run();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            run();
        }
        let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!("{label}: {per_call_us:>9.2} µs/call");
    }
}

/// Diagnostic: chain N qmm_t calls into ONE MTL3 command buffer with
/// the SAME pipeline + buffers reused across calls. Measures average
/// per-call time when the call is part of a back-to-back chain (vs.
/// the existing `qmm_t_timing_at_prefill_shapes` test which measures
/// isolated single-call timing with sync between calls).
///
/// If chain-per-call ≈ isolated-per-call: the chain isn't adding
/// overhead; the prefill perf gap is in something else (e.g., MTL4
/// vs MTL3 dispatch, scheduler).  If chain-per-call >> isolated:
/// the chain itself is the bug — per-dispatch state-switching or
/// cache-cold-start eating the time.
#[test]
#[ignore = "Chain-overhead diagnostic — run manually with --ignored --nocapture"]
fn qmm_t_chain_per_call_timing() {
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLDevice};
    use std::time::Instant;
    let device = ferrite_metal_kernels::device::detect_device()
        .expect("Metal device")
        .device;
    let qmm = ferrite_metal_kernels::quantized::MetalAffineQmmT::new(device.clone())
        .expect("MetalAffineQmmT");
    let queue = device.newCommandQueue().expect("queue");

    // Gate/up shape — the heaviest prefill GEMM in Llama-3.2-3B.
    let m: u32 = 1024;
    let n: u32 = 8192;
    let k: u32 = 3072;
    let gs: u32 = 64;

    let packed = zeroed_buffer(&device, (n * k / 2) as usize);
    let scales = zeroed_buffer(&device, (n * k / gs * 2) as usize);
    let biases = zeroed_buffer(&device, (n * k / gs * 2) as usize);
    let x = zeroed_buffer(&device, (m * k * 2) as usize);
    let y = zeroed_buffer(&device, (m * n * 2 * 32) as usize);

    // ── Isolated timing — one call per cmdbuf, sync between ────────
    // 5 warmups + 20 iters.
    for _ in 0..5 {
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        qmm.execute(
            &x,
            &packed,
            &scales,
            &biases,
            &y,
            m,
            n,
            k,
            1,
            gs,
            4,
            ferrite_metal_kernels::quantized::DequantDtype::Bf16,
            ferrite_metal_kernels::quantized::ScaleDtype::F16,
            &enc,
        )
        .expect("dispatch");
        enc.endEncoding();
        cb.commit();
        unsafe {
            cb.waitUntilCompleted();
        }
    }
    let t0 = Instant::now();
    let iters: usize = 20;
    for _ in 0..iters {
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        qmm.execute(
            &x,
            &packed,
            &scales,
            &biases,
            &y,
            m,
            n,
            k,
            1,
            gs,
            4,
            ferrite_metal_kernels::quantized::DequantDtype::Bf16,
            ferrite_metal_kernels::quantized::ScaleDtype::F16,
            &enc,
        )
        .expect("dispatch");
        enc.endEncoding();
        cb.commit();
        unsafe {
            cb.waitUntilCompleted();
        }
    }
    let iso_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    eprintln!("isolated (1 call / cmdbuf / sync): {iso_us:.2} µs/call");

    // ── Chain timing — N calls in ONE cmdbuf, single sync at end ──
    // Mimics what a forward does: many dispatches encoded into one
    // command buffer, no per-call CPU sync. Apple's default MTL3
    // compute encoder serial dispatch type means consecutive
    // dispatches are auto-ordered, equivalent to MTL4 with a
    // barrier between each.
    for &chain_n in &[10_usize, 50, 100, 200] {
        // Warmup
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        for _ in 0..chain_n {
            qmm.execute(
                &x,
                &packed,
                &scales,
                &biases,
                &y,
                m,
                n,
                k,
                1,
                gs,
                4,
                ferrite_metal_kernels::quantized::DequantDtype::Bf16,
                ferrite_metal_kernels::quantized::ScaleDtype::F16,
                &enc,
            )
            .expect("dispatch");
        }
        enc.endEncoding();
        cb.commit();
        unsafe {
            cb.waitUntilCompleted();
        }

        let t = Instant::now();
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        for _ in 0..chain_n {
            qmm.execute(
                &x,
                &packed,
                &scales,
                &biases,
                &y,
                m,
                n,
                k,
                1,
                gs,
                4,
                ferrite_metal_kernels::quantized::DequantDtype::Bf16,
                ferrite_metal_kernels::quantized::ScaleDtype::F16,
                &enc,
            )
            .expect("dispatch");
        }
        enc.endEncoding();
        cb.commit();
        unsafe {
            cb.waitUntilCompleted();
        }
        let elapsed_us = t.elapsed().as_secs_f64() * 1e6;
        let per_call_us = elapsed_us / chain_n as f64;
        let overhead_per_call_us = per_call_us - iso_us;
        eprintln!(
            "chain n={chain_n:>3}: {per_call_us:>9.2} µs/call  (overhead vs isolated: {overhead_per_call_us:>+9.2} µs)"
        );
    }
}

/// Diagnostic probe for MPP `matmul2d` cooperative-tensor per-thread
/// layout. Dumps (capacity, is_valid, row, col) for the exact descriptor
/// used by `affine_qmm_t_nax` so the layout MPP actually picks can be
/// compared against MLX `BaseNAXFrag`'s 2-row × 4-col assumption.
///
/// On M4 the dump reveals MPP returns coords that span a 32-N × 16-M
/// per-simdgroup region rather than the 16×16 / 16×32 contiguous tiles
/// MLX expects, which is why `BaseNAXFrag::mma`'s per-index `ct_c[i] =
/// Cn0[i]` copy scrambles outputs. On M5+/A19+ where NAX hardware exists
/// the layout is expected to match `BaseNAXFrag`. Re-run on any new chip
/// to validate before flipping [`is_nax_capable`] on for that generation.
#[test]
#[ignore = "Diagnostic for NAX layout bug — run manually with --ignored --nocapture"]
fn nax_probe_dump_layout() {
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCommandQueue, MTLComputeCommandEncoder, MTLLibrary, MTLSize};

    let dev = ferrite_metal_kernels::device::detect_device().expect("Metal device");
    let device = dev.device;

    let bytes: &'static [u8] = ferrite_metal_kernels::embedded_metallib!("nax_probe");
    let library = ferrite_metal_kernels::shader_cache::load_library_from_bytes(&device, bytes)
        .expect("nax_probe metallib");
    let function = library
        .newFunctionWithName(&NSString::from_str("nax_probe"))
        .expect("nax_probe function");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("nax_probe pipeline");

    // 3 ops × 32 lanes × 32 cap × 4 fields × 4 bytes
    const OPS: usize = 3;
    const LANES: usize = 32;
    const CAP: usize = 32;
    const FIELDS: usize = 4;
    let out_buf = zeroed_buffer(&device, OPS * LANES * CAP * FIELDS * 4);

    let cmdq: Retained<ProtocolObject<dyn MTLCommandQueue>> =
        device.newCommandQueue().expect("command queue");
    let cmdbuf = cmdq.commandBuffer().expect("command buffer");
    let encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmdbuf.computeCommandEncoder().expect("encoder");
    encoder.setComputePipelineState(&pipeline);
    unsafe { encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 0) };
    let threadgroups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
    encoder.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let ptr = out_buf.contents().as_ptr() as *const i32;
    let out = unsafe { std::slice::from_raw_parts(ptr, OPS * LANES * CAP * FIELDS) };

    for (op_idx, op_name) in [
        "ct_a (16x16 bf16)",
        "ct_b (16x32 bf16)",
        "ct_c (16x32 float)",
    ]
    .iter()
    .enumerate()
    {
        let base = op_idx * LANES * CAP * FIELDS;
        let cap = out[base];
        eprintln!("\n=== {op_name} : capacity={cap} ===");
        for lane in 0..LANES {
            let mut entries = Vec::new();
            for idx in 0..cap as usize {
                let p = base + lane * CAP * FIELDS + idx * FIELDS;
                let valid = out[p + 1];
                let row = out[p + 2];
                let col = out[p + 3];
                entries.push(format!(
                    "[{idx}]({row},{col}){}",
                    if valid == 0 { "!" } else { "" }
                ));
            }
            eprintln!("lane{lane:2}: {}", entries.join(" "));
        }
    }
}

/// All-ones MMA sanity test: A and B both filled with 1.0; MMA should
/// produce C[m, n] = K = 16 for every cell. Validates whether MPP's
/// `op.run(tA, tB, cT)` works at all on M4 and which axis order MPP
/// uses for the destination cooperative tensor's `get_multidimensional_index`.
#[test]
#[ignore = "Diagnostic — run with --ignored --nocapture"]
fn nax_ones_mma_sanity() {
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCommandQueue, MTLComputeCommandEncoder, MTLLibrary, MTLSize};

    let dev = ferrite_metal_kernels::device::detect_device().expect("Metal device");
    let device = dev.device;

    let bytes: &'static [u8] = ferrite_metal_kernels::embedded_metallib!("nax_probe");
    let library = ferrite_metal_kernels::shader_cache::load_library_from_bytes(&device, bytes)
        .expect("nax_probe metallib");
    let function = library
        .newFunctionWithName(&NSString::from_str("nax_ones_mma"))
        .expect("nax_ones_mma function");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("nax_ones_mma pipeline");

    // c_out has two halves of 16x32 float = 2 * 16 * 32 * 4 bytes
    let out_buf = zeroed_buffer(&device, 2 * 16 * 32 * 4);

    let cmdq: Retained<ProtocolObject<dyn MTLCommandQueue>> =
        device.newCommandQueue().expect("command queue");
    let cmdbuf = cmdq.commandBuffer().expect("command buffer");
    let encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmdbuf.computeCommandEncoder().expect("encoder");
    encoder.setComputePipelineState(&pipeline);
    unsafe { encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 0) };
    // 32x16 = 512 bf16 = 1024 bytes for b_ws
    unsafe { encoder.setThreadgroupMemoryLength_atIndex(1024, 0) };
    let threadgroups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
    encoder.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let ptr = out_buf.contents().as_ptr() as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, 2 * 16 * 32) };

    // First half: written as if coord = (M, N)
    eprintln!("\n=== As-if coord = (M, N) — first half (16x32) ===");
    for r in 0..16 {
        let row: Vec<f32> = (0..32).map(|c| out[r * 32 + c]).collect();
        eprintln!("row{r:2}: {row:?}");
    }
    // Second half: written as if coord = (N, M) — for our test (16, 32, 16)
    // with K=16 reduction, every cell should be 16.0.
    eprintln!("\n=== As-if coord = (N, M) — second half (16x32) ===");
    for r in 0..16 {
        let row: Vec<f32> = (0..32).map(|c| out[16 * 32 + r * 32 + c]).collect();
        eprintln!("row{r:2}: {row:?}");
    }
}

/// Compiles `metal_nax.h` + the all-ones MMA kernel at RUNTIME via
/// `newLibraryWithSource` (fast-math off, Metal 4.0 — exactly like mlx),
/// instead of the offline `xcrun metal` metallib the build embeds. If
/// this gives 32 but `nax_frag_mma_check` (offline) gives 16, the offline
/// toolchain miscompiles MPP cooperative tensors and the fix is to JIT
/// NAX shaders at runtime like mlx does.
#[test]
#[ignore = "Offline-vs-runtime compiler A/B — run with --ignored --nocapture"]
fn nax_runtime_compile_check() {
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLLibrary, MTLMathMode, MTLSize,
    };

    let dev = ferrite_metal_kernels::device::detect_device().expect("Metal device");
    let device = dev.device;

    // metal_nax.h carries the MPP include + BaseNAXFrag/NAXTile/tile_matmad
    // in namespace mlx::steel + the internals pragma. Append the kernel.
    let header =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/metal_nax.h"))
            .expect("read metal_nax.h");
    let kernel = r#"
using namespace metal;
[[kernel]] void rt_nax_mma(device float* d_out [[buffer(0)]],
                           uint simd_lid [[thread_index_in_simdgroup]]) {
    using namespace mlx::steel;
    constexpr int SZ = 32;
    threadgroup bfloat a_tg[SZ*SZ]; threadgroup bfloat b_tg[SZ*SZ];
    if (simd_lid == 0) { for (int i=0;i<SZ*SZ;++i){a_tg[i]=bfloat(1.0f);b_tg[i]=bfloat(1.0f);} }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NAXTile<bfloat,2,2> A; NAXTile<bfloat,2,2> B; NAXTile<float,2,2> D; D.clear();
    A.template load<bfloat,SZ,1>(a_tg); B.template load<bfloat,SZ,1>(b_tg);
    tile_matmad_nax(D, A, metal::bool_constant<false>{}, B, metal::bool_constant<true>{});
    D.store(d_out, SZ);
}
"#;
    let source = format!("{header}\n{kernel}");
    let opts = objc2_metal::MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    opts.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version4_0);

    let library = match device
        .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&opts))
    {
        Ok(l) => l,
        Err(e) => {
            eprintln!("runtime compile FAILED: {}", e.localizedDescription());
            panic!("newLibraryWithSource failed");
        }
    };
    let function = library
        .newFunctionWithName(&NSString::from_str("rt_nax_mma"))
        .expect("rt_nax_mma function");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline");

    let out_buf = zeroed_buffer(&device, 32 * 32 * 4);
    let cmdq: Retained<ProtocolObject<dyn MTLCommandQueue>> =
        device.newCommandQueue().expect("queue");
    let cmdbuf = cmdq.commandBuffer().expect("cmdbuf");
    let encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmdbuf.computeCommandEncoder().expect("encoder");
    encoder.setComputePipelineState(&pipeline);
    unsafe { encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 0) };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    let ptr = out_buf.contents().as_ptr() as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, 32 * 32) };
    let bad = (0..1024).filter(|&i| (out[i] - 32.0).abs() > 0.01).count();
    eprintln!("RUNTIME-compiled metal_nax.h: row0={:?}", &out[..8]);
    eprintln!("mismatches vs 32.0: {bad}/1024");
}

/// Kernel-level A/B: times the NAX qmm_t against the production
/// non-NAX pick (Standard/SplitK) on identical prefill shapes, isolated
/// from the rest of the model. Reports median µs over 30 iters (5 warm).
#[test]
#[ignore = "NAX vs Standard qmm_t microbench — run with --ignored --nocapture"]
fn nax_vs_standard_qmm_t_bench() {
    let device = detect_device().expect("Metal device").device;
    let qmm = MetalAffineQmmT::new(device.clone()).expect("MetalAffineQmmT");
    let gs = 64u32;

    // Representative Llama-3B prefill GEMMs (all N,K % 64 == 0).
    let shapes_nk: &[(u32, u32, &str)] = &[
        (3072, 3072, "qkv/o  3072x3072"),
        (8192, 3072, "gate/up 8192x3072"),
        (3072, 8192, "down   3072x8192"),
    ];
    let m_vals = [16u32, 32, 64, 128, 256, 512, 1024];

    let time_kernel = |m: u32, n: u32, k: u32, kernel: QmmTKernel| -> f64 {
        let (packed, scales, biases, x) = make_inputs_bf16(
            0xBEEF ^ (m as u64) ^ ((n as u64) << 20),
            n as usize,
            k as usize,
            m as usize,
            gs as usize,
        );
        let packed_buf = buffer_from_bytes(&device, &packed);
        let sb: &[u8] = unsafe {
            std::slice::from_raw_parts(
                scales.as_ptr() as *const u8,
                std::mem::size_of_val(&scales[..]),
            )
        };
        let bb: &[u8] = unsafe {
            std::slice::from_raw_parts(
                biases.as_ptr() as *const u8,
                std::mem::size_of_val(&biases[..]),
            )
        };
        let xb: &[u8] = unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(&x[..]))
        };
        let scales_buf = buffer_from_bytes(&device, sb);
        let biases_buf = buffer_from_bytes(&device, bb);
        let x_buf = buffer_from_bytes(&device, xb);
        let y_buf = zeroed_buffer(
            &device,
            (m * n) as usize * std::mem::size_of::<half::bf16>() * 32,
        );
        let mut stream = MetalStream::new(&device);

        let mut run = || {
            let cb = stream.get_command_buffer().expect("cb").clone();
            let enc = cb.computeCommandEncoder().expect("enc");
            qmm.execute_with_kernel(
                &x_buf,
                &packed_buf,
                &scales_buf,
                &biases_buf,
                &y_buf,
                m,
                n,
                k,
                1,
                gs,
                4,
                DequantDtype::Bf16,
                ScaleDtype::F16,
                kernel,
                &enc,
            )
            .expect("dispatch");
            enc.endEncoding();
            stream.commit().expect("commit");
            stream.synchronize().expect("sync");
        };
        for _ in 0..5 {
            run();
        } // warmup
        let mut samples = Vec::with_capacity(30);
        for _ in 0..30 {
            let t = std::time::Instant::now();
            run();
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2] // median µs
    };

    eprintln!("\n=== NAX vs non-NAX qmm_t (median µs, 30 iters) — gs={gs} ===");
    eprintln!(
        "{:<22} {:>5} {:>10} {:>12} {:>9}",
        "shape", "M", "NAX µs", "base µs", "speedup"
    );
    for &(n, k, label) in shapes_nk {
        for &m in &m_vals {
            let base_kernel = pick_qmm_t_kernel(m, n, k, 1, gs, /*is_nax=*/ false);
            let nax_us = time_kernel(m, n, k, QmmTKernel::Nax);
            let base_us = time_kernel(m, n, k, base_kernel);
            let base_tag = match base_kernel {
                QmmTKernel::Standard => "Std",
                QmmTKernel::SplitK { .. } => "SplitK",
                QmmTKernel::Nax => "Nax",
            };
            eprintln!(
                "{:<22} {:>5} {:>10.1} {:>8.1}({:<6}) {:>8.2}x",
                label,
                m,
                nax_us,
                base_us,
                base_tag,
                base_us / nax_us
            );
        }
    }
}

/// Prints this GPU's Metal architecture name + parsed gen, the exact
/// inputs to mlx's `is_nax_available()` gate
/// (`gen >= (arch.back()=='p' ? 18 : 17)`).
#[test]
#[ignore = "Diagnostic — run with --ignored --nocapture"]
fn nax_arch_probe() {
    use objc2_metal::MTLDevice;
    let dev = ferrite_metal_kernels::device::detect_device().expect("Metal device");
    let arch = dev.device.architecture();
    let name = arch.name().to_string();
    let bytes = name.as_bytes();
    let (gen, class) = if bytes.len() >= 3 {
        let tens = (bytes[bytes.len() - 3] as char).to_digit(10).unwrap_or(0);
        let ones = (bytes[bytes.len() - 2] as char).to_digit(10).unwrap_or(0);
        (tens * 10 + ones, *bytes.last().unwrap() as char)
    } else {
        (0, '?')
    };
    let threshold = if class == 'p' { 18 } else { 17 };
    eprintln!("Metal architecture: {name:?}");
    eprintln!("parsed gen={gen}, class='{class}', threshold={threshold}");
    eprintln!("mlx is_nax_available() would be: {}", gen >= threshold);
}

/// Faithful-path MMA check: dispatches `nax_frag_mma_check`, which runs
/// the kernel's real `NAXTile`→`tile_matmad_nax`→`store` path with
/// A=B=1.0 over a 32×32×32 tile. Every output must equal K=32. Isolates
/// metal_nax.h's MMA correctness on M5 from the quantized W-loader.
#[test]
#[ignore = "Diagnostic for NAX MMA on M5 — run with --ignored --nocapture"]
fn nax_frag_mma_check() {
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCommandQueue, MTLComputeCommandEncoder, MTLLibrary, MTLSize};

    let dev = ferrite_metal_kernels::device::detect_device().expect("Metal device");
    let device = dev.device;

    let bytes: &'static [u8] = ferrite_metal_kernels::embedded_metallib!("nax_probe");
    let library = ferrite_metal_kernels::shader_cache::load_library_from_bytes(&device, bytes)
        .expect("nax_probe metallib");
    let function = library
        .newFunctionWithName(&NSString::from_str("nax_frag_mma_check"))
        .expect("nax_frag_mma_check function");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("nax_frag_mma_check pipeline");

    let out_buf = zeroed_buffer(&device, 32 * 32 * 4);

    let cmdq: Retained<ProtocolObject<dyn MTLCommandQueue>> =
        device.newCommandQueue().expect("command queue");
    let cmdbuf = cmdq.commandBuffer().expect("command buffer");
    let encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
        cmdbuf.computeCommandEncoder().expect("encoder");
    encoder.setComputePipelineState(&pipeline);
    unsafe { encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 0) };
    let threadgroups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
    encoder.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let ptr = out_buf.contents().as_ptr() as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, 32 * 32) };

    eprintln!("\n=== nax_frag_mma_check D (32x32), expected all 32.0 ===");
    let mut bad = 0usize;
    for r in 0..32 {
        let row: Vec<f32> = (0..32).map(|c| out[r * 32 + c]).collect();
        for &v in &row {
            if (v - 32.0).abs() > 0.01 {
                bad += 1;
            }
        }
        if r < 4 || bad == 0 {
            eprintln!("row{r:2}: {row:?}");
        }
    }
    eprintln!("mismatches vs 32.0: {bad}/1024");
}
