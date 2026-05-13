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

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
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

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
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

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert!(
        matches!(expected_kernel, QmmTKernel::SplitK { split_k: 4, .. }),
        "Llama-1B q_proj prefill shape should route to SplitK(4), got {:?}",
        expected_kernel
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11B_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qmm_t_bf16(
        &packed, &scales, &biases, &x, m, n, k, group_size as u32, expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) =
        worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
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

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert!(
        matches!(expected_kernel, QmmTKernel::SplitK { split_k: 16, .. }),
        "Llama-1B kv_proj prefill shape should route to SplitK(16), got {:?}",
        expected_kernel
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11C_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qmm_t_bf16(
        &packed, &scales, &biases, &x, m, n, k, group_size as u32, expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) =
        worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
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

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert_eq!(
        expected_kernel,
        QmmTKernel::Standard,
        "Llama-1B gate/up_proj prefill shape should route to Standard"
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11D_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qmm_t_bf16(
        &packed, &scales, &biases, &x, m, n, k, group_size as u32, expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) =
        worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
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

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
    assert!(
        matches!(expected_kernel, QmmTKernel::SplitK { split_k: 4, .. }),
        "Llama-1B down_proj prefill shape should route to SplitK(4), got {:?}",
        expected_kernel
    );

    let (packed, scales, biases, x) =
        make_inputs_bf16(0x11E_u64 ^ group_size as u64, n, k, m, group_size);
    let expected = cpu_qmm_t_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let metal = run_qmm_t_bf16(
        &packed, &scales, &biases, &x, m, n, k, group_size as u32, expected_kernel,
    );

    let (idx, mv, ev, abs_err, allowed) =
        worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
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

    let expected_kernel = pick_qmm_t_kernel(m as u32, n as u32, k as u32, 1, group_size as u32, false);
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
    let device = ferrite_metal_kernels::device::detect_device().expect("Metal device").device;
    let mut stream = ferrite_metal_kernels::stream::MetalStream::new(&device);
    let qmm = ferrite_metal_kernels::quantized::MetalAffineQmmT::new(device.clone())
        .expect("MetalAffineQmmT");

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(&scales[..]))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(&biases[..]))
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
            let allowed = 4.0 * ((k as f32).sqrt() * 0.5 * (1.0 / 128.0) * 0.5
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
    eprintln!("simdgroup mismatches: TL(0)={}/{} TR(1)={}/{} BL(2)={}/{} BR(3)={}/{}",
        sg_bad[0], sg_ok[0] + sg_bad[0],
        sg_bad[1], sg_ok[1] + sg_bad[1],
        sg_bad[2], sg_ok[2] + sg_bad[2],
        sg_bad[3], sg_ok[3] + sg_bad[3]);

    // Print first row of each simdgroup, metal vs cpu
    for &sg_row in &[0, 32] {
        for &sg_col in &[0, 32] {
            let i = sg_row;
            let j = sg_col;
            let pairs: Vec<String> = (0..8)
                .map(|d| format!("[{}]m={:.2}/c={:.2}", j + d, metal[i*n+j+d].to_f32(), expected[i*n+j+d].to_f32()))
                .collect();
            eprintln!("row{} col{}+: {}", i, j, pairs.join(" "));
        }
    }

    let (idx, mv, ev, abs_err, allowed) =
        worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmm_t NAX gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
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
    use objc2_metal::{
        MTLCommandQueue, MTLComputeCommandEncoder, MTLLibrary, MTLSize,
    };

    let dev = ferrite_metal_kernels::device::detect_device().expect("Metal device");
    let device = dev.device;

    let bytes: &'static [u8] =
        ferrite_metal_kernels::embedded_metallib!("nax_probe");
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
    let threadgroups = MTLSize { width: 1, height: 1, depth: 1 };
    let threads = MTLSize { width: 32, height: 1, depth: 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
    encoder.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();

    let ptr = out_buf.contents().as_ptr() as *const i32;
    let out = unsafe { std::slice::from_raw_parts(ptr, OPS * LANES * CAP * FIELDS) };

    for (op_idx, op_name) in ["ct_a (16x16 bf16)", "ct_b (16x32 bf16)", "ct_c (16x32 float)"]
        .iter().enumerate()
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
                entries.push(format!("[{idx}]({row},{col}){}", if valid == 0 {"!"} else {""}));
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
    use objc2_metal::{
        MTLCommandQueue, MTLComputeCommandEncoder, MTLLibrary, MTLSize,
    };

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
    let threadgroups = MTLSize { width: 1, height: 1, depth: 1 };
    let threads = MTLSize { width: 32, height: 1, depth: 1 };
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
