// SPDX-License-Identifier: Apache-2.0
//! 8-bit MLX-affine parity tests (Gemma4 MLP projections: 8-bit g64).
//!
//! The qmv/qmm_t template bodies are bits-generic MLX ports; the `_b_8_`
//! entry-point instantiations are new — these tests pin the kernels
//! against the CPU reference (`cpu_reference::affine_qmm_t_b8`) for the
//! production dtype combo (bf16 act × bf16 scales, Gemma4) and the
//! f16 × f16 combo.
//!
//! GPU tests — run with `--test-threads=1`.

use ferrite_metal_kernels::cpu_reference::{affine_qmm_t_b8_bf16_s_bf16, affine_qmm_t_b8_f16};
use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::quantized::{
    DequantDtype, MetalAffineQmmT, MetalAffineQmv, QmmTKernel, ScaleDtype,
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

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
        (self.next() >> 56) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    let buf = device
        .newBufferWithLength_options(bytes.len().max(4), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buf.contents().as_ptr() as *mut u8,
            bytes.len(),
        );
    }
    buf
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    let buf = device
        .newBufferWithLength_options(n_bytes.max(4), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, n_bytes) };
    buf
}

/// Realistic 8-bit affine inputs: one byte per element; scale ≈
/// weight_range/255 with bias = range minimum, so dequantized weights
/// land in ±0.1 like real checkpoint tensors.
#[allow(clippy::type_complexity)]
fn make_inputs_b8_f32(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(k % group_size, 0);
    let n_groups = n * k / group_size;
    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n * k).map(|_| rng.next_byte()).collect();
    let scales: Vec<f32> = (0..n_groups)
        .map(|_| 0.0006 + 0.0004 * rng.next_unit_f32())
        .collect();
    let biases: Vec<f32> = (0..n_groups)
        .map(|_| -0.1 + 0.02 * rng.next_unit_f32())
        .collect();
    let x: Vec<f32> = (0..m * k)
        .map(|_| 2.0 * rng.next_unit_f32() - 1.0)
        .collect();
    (packed, scales, biases, x)
}

enum Op {
    Qmv,
    QmmT,
    /// Force the NAX (Apple9 MMA) qmm_t via `execute_with_kernel` —
    /// the b8 W-loader (byte-per-element dequant) parity gate.
    QmmTNax,
}

#[allow(clippy::too_many_arguments)]
fn run_case_bf16(op: Op, m: usize, n: usize, k: usize, group_size: usize, seed: u64, tol: f32) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (packed, scales_f, biases_f, x_f) = make_inputs_b8_f32(seed, n, k, m, group_size);
    let scales: Vec<half::bf16> = scales_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let biases: Vec<half::bf16> = biases_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let x: Vec<half::bf16> = x_f.iter().map(|&v| half::bf16::from_f32(v)).collect();

    let expected = affine_qmm_t_b8_bf16_s_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);

    let mut stream = MetalStream::new(&device);
    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);

    let cmd_buf = stream.get_command_buffer().expect("cb").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    match op {
        Op::Qmv => {
            let qmv = MetalAffineQmv::new(device.clone()).expect("MetalAffineQmv");
            qmv.execute(
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
                8,
                DequantDtype::Bf16,
                ScaleDtype::Bf16,
                &encoder,
            )
            .expect("qmv b8 dispatch");
        }
        Op::QmmT => {
            let qmm = MetalAffineQmmT::new(device.clone()).expect("MetalAffineQmmT");
            let kernel = qmm
                .execute(
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
                    8,
                    DequantDtype::Bf16,
                    ScaleDtype::Bf16,
                    &encoder,
                )
                .expect("qmm_t b8 dispatch");
            assert!(
                matches!(kernel, QmmTKernel::Standard),
                "b8 must route to Standard (got {kernel:?})"
            );
        }
        Op::QmmTNax => {
            let qmm = MetalAffineQmmT::new(device.clone()).expect("MetalAffineQmmT");
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
                8,
                DequantDtype::Bf16,
                ScaleDtype::Bf16,
                QmmTKernel::Nax,
                &encoder,
            )
            .expect("qmm_t nax b8 dispatch");
        }
    }
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let got: Vec<f32> = unsafe {
        std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const half::bf16, m * n)
            .iter()
            .map(|h| h.to_f32())
            .collect()
    };
    let mut max_err = 0f32;
    for i in 0..m * n {
        max_err = max_err.max((got[i] - expected[i].to_f32()).abs());
    }
    assert!(max_err < tol, "b8 bf16 max_err {max_err} (tol {tol})");
}

#[test]
fn qmv_b8_bf16_gemma4_gate_shape() {
    // Gemma4 MLP gate decode: M=1, N=intermediate 15360 (trimmed to
    // 1920 for CPU-ref speed; same K), K=hidden 3840.
    run_case_bf16(Op::Qmv, 1, 1920, 3840, 64, 11, 0.06);
}

#[test]
fn qmv_b8_bf16_gemma4_down_shape() {
    // Gemma4 MLP down decode: K=intermediate 15360, N=hidden (trimmed).
    run_case_bf16(Op::Qmv, 1, 480, 15360, 64, 13, 0.12);
}

#[test]
fn qmm_t_b8_bf16_prefill_shape() {
    // Prefill M=64 through the Standard qmm_t.
    run_case_bf16(Op::QmmT, 64, 256, 1536, 64, 17, 0.05);
}

/// Timing harness (not a correctness gate): our NAX b8 throughput on
/// the Gemma4 MLP gate/up shape vs mlx `quantized_matmul` (13.1
/// TFLOPS on M5). Run with `--ignored --nocapture`.
#[test]
#[ignore = "perf probe — run manually with --nocapture"]
fn qmm_t_nax_b8_bf16_gemma4_mlp_bench() {
    let Some(di) = detect_device() else {
        return;
    };
    if !ferrite_metal_kernels::ferrite_metal_targets::is_nax_capable(di.profile.generation) {
        return;
    }
    let device = di.device.clone();
    let (m, n, k, gs) = (2930usize, 15360usize, 3840usize, 64usize);
    let (packed, scales_f, biases_f, x_f) = make_inputs_b8_f32(7, n, k, m, gs);
    let scales: Vec<half::bf16> = scales_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let biases: Vec<half::bf16> = biases_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let x: Vec<half::bf16> = x_f.iter().map(|&v| half::bf16::from_f32(v)).collect();

    let mut stream = MetalStream::new(&device);
    let qmm = MetalAffineQmmT::new(device.clone()).expect("MetalAffineQmmT");
    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);

    let iters = 20;
    let mut run = |label: &str, bits: u32| {
        // Values are garbage for bits=4 (the b8 packing re-read as
        // nibbles) — irrelevant for a pure timing probe.
        // 4 dispatches per commit so the per-iter commit+sync host
        // roundtrip amortizes out of the per-op number.
        const BATCH: usize = 4;
        let go = |stream: &mut MetalStream| {
            let cb = stream.get_command_buffer().expect("cb").clone();
            let enc = cb.computeCommandEncoder().expect("encoder");
            for _ in 0..BATCH {
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
                    gs as u32,
                    bits,
                    DequantDtype::Bf16,
                    ScaleDtype::Bf16,
                    QmmTKernel::Nax,
                    &enc,
                )
                .expect("dispatch");
            }
            enc.endEncoding();
            stream.commit().expect("commit");
            stream.synchronize().expect("sync");
        };
        go(&mut stream);
        go(&mut stream);
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            go(&mut stream);
        }
        let dt = t0.elapsed().as_secs_f64() / (iters * BATCH) as f64;
        let tflops = (2.0 * m as f64 * n as f64 * k as f64) / dt / 1e12;
        eprintln!("[{label}] {:.2} ms/iter -> {tflops:.2} TFLOPS", dt * 1e3);
    };
    run("nax b8 M=2930 N=15360 K=3840", 8);
    run("nax b4 M=2930 N=15360 K=3840", 4);

    // Buffer-dims (unspecialized) variant — isolates the cost of
    // function-constant pipeline specialization on MPP code.
    use objc2_foundation::NSString;
    use objc2_metal::MTLComputeCommandEncoder as _;
    use objc2_metal::MTLComputePipelineState as _;
    let lib = ferrite_metal_kernels::shader_cache::compile_nax_library_from_source(&device)
        .expect("nax lib");
    use objc2_metal::MTLLibrary as _;
    for (label, bits) in [("dims b8", 8u32), ("dims b4", 4u32)] {
        let name = format!("affine_qmm_t_nax_dims_bf16_s_bf16_gs_64_b_{bits}_alN_true_batch_0");
        let func = lib
            .newFunctionWithName(&NSString::from_str(&name))
            .expect("dims function");
        let pso = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("dims pipeline");
        let dims: Vec<i32> = vec![k as i32, n as i32, m as i32];
        let dims_buf = buffer_from_bytes(&device, as_bytes(&dims));
        let go = |stream: &mut MetalStream| {
            let cb = stream.get_command_buffer().expect("cb").clone();
            let enc = cb.computeCommandEncoder().expect("encoder");
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&scales_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&biases_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&y_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 4, 6);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 8, 7);
            }
            let tg = objc2_metal::MTLSize {
                width: n.div_ceil(64),
                height: m.div_ceil(64),
                depth: 1,
            };
            let tpg = objc2_metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg, tpg);
            enc.endEncoding();
            stream.commit().expect("commit");
            stream.synchronize().expect("sync");
        };
        go(&mut stream);
        go(&mut stream);
        let t0 = std::time::Instant::now();
        let iters2 = 20;
        for _ in 0..iters2 {
            go(&mut stream);
        }
        let dt = t0.elapsed().as_secs_f64() / iters2 as f64;
        let tflops = (2.0 * m as f64 * n as f64 * k as f64) / dt / 1e12;
        eprintln!(
            "[nax {label} unspecialized] {:.2} ms/iter -> {tflops:.2} TFLOPS",
            dt * 1e3
        );
    }

    // MLX's OWN compiled binary (the wheel's mlx.metallib), same
    // machine, our harness — isolates toolchain codegen from source.
    let mlxlib_path = std::env::var("FERRITE_BENCH_MLX_METALLIB").unwrap_or_default();
    if !mlxlib_path.is_empty() {
        let bytes: &'static [u8] = Box::leak(
            std::fs::read(&mlxlib_path)
                .expect("mlx.metallib")
                .into_boxed_slice(),
        );
        let lib = ferrite_metal_kernels::shader_cache::load_library_from_bytes(&device, bytes)
            .expect("load mlx.metallib");
        let name = "affine_qmm_t_nax_bfloat16_t_gs_64_b_8_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0";
        let func = lib
            .newFunctionWithName(&NSString::from_str(name))
            .expect("mlx nax function");
        let pso = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("mlx nax pipeline");
        let dims: Vec<i32> = vec![k as i32, n as i32, m as i32];
        let dims_buf = buffer_from_bytes(&device, as_bytes(&dims));
        let dummy = zeroed_buffer(&device, 64);
        let go = |stream: &mut MetalStream| {
            let cb = stream.get_command_buffer().expect("cb").clone();
            let enc = cb.computeCommandEncoder().expect("encoder");
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&packed_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&scales_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&biases_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&x_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&y_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 4, 6);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 8, 7);
                for i in 8..16 {
                    enc.setBuffer_offset_atIndex(Some(&dummy), 0, i);
                }
            }
            let tg = objc2_metal::MTLSize {
                width: n.div_ceil(64),
                height: m.div_ceil(64),
                depth: 1,
            };
            let tpg = objc2_metal::MTLSize {
                width: 32,
                height: 2,
                depth: 2,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg, tpg);
            enc.endEncoding();
            stream.commit().expect("commit");
            stream.synchronize().expect("sync");
        };
        go(&mut stream);
        go(&mut stream);
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            go(&mut stream);
        }
        let dt = t0.elapsed().as_secs_f64() / 20.0;
        let tflops = (2.0 * m as f64 * n as f64 * k as f64) / dt / 1e12;
        eprintln!(
            "[nax MLX-BINARY b8] {:.2} ms/iter -> {tflops:.2} TFLOPS",
            dt * 1e3
        );

        // Output parity: the MLX binary must produce the same numbers
        // as OUR kernel on the same inputs — otherwise the timing
        // compares different work (e.g. a mis-bound dim early-out
        // would look "fast"). Run ours into a second buffer, compare.
        let y_ours = zeroed_buffer(&device, m * n * 2);
        {
            let cb = stream.get_command_buffer().expect("cb").clone();
            let enc = cb.computeCommandEncoder().expect("encoder");
            qmm.execute_with_kernel(
                &x_buf,
                &packed_buf,
                &scales_buf,
                &biases_buf,
                &y_ours,
                m as u32,
                n as u32,
                k as u32,
                1,
                gs as u32,
                8,
                DequantDtype::Bf16,
                ScaleDtype::Bf16,
                QmmTKernel::Nax,
                &enc,
            )
            .expect("dispatch");
            enc.endEncoding();
            stream.commit().expect("commit");
            stream.synchronize().expect("sync");
        }
        let a: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const half::bf16, m * n)
        };
        let b: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(y_ours.contents().as_ptr() as *const half::bf16, m * n)
        };
        let mut max_d = 0f32;
        let mut nonzero = 0usize;
        for i in 0..m * n {
            max_d = max_d.max((a[i].to_f32() - b[i].to_f32()).abs());
            if a[i].to_f32() != 0.0 {
                nonzero += 1;
            }
        }
        eprintln!(
            "[nax MLX-BINARY parity] max|mlx-ours|={max_d} nonzero={nonzero}/{}",
            m * n
        );
        assert!(max_d < 0.05, "mlx-binary output mismatch: {max_d}");
        assert!(nonzero > m * n / 2, "mlx-binary output mostly zero");
    }
}

#[test]
fn qmm_t_nax_b8_bf16_prefill_shape() {
    // Prefill M=64 through the NAX qmm_t (b8 byte-per-element
    // W-loader, the Gemma4 MLP prefill path on M5+). Skips on
    // non-NAX-capable hardware.
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !ferrite_metal_kernels::ferrite_metal_targets::is_nax_capable(di.profile.generation) {
        eprintln!(
            "skipping NAX b8 parity on non-NAX hardware ({:?})",
            di.profile.generation
        );
        return;
    }
    // N multiple of 64 (aligned) + an unaligned-N case for the
    // N-tail zeroing path.
    run_case_bf16(Op::QmmTNax, 64, 256, 1536, 64, 19, 0.05);
    run_case_bf16(Op::QmmTNax, 64, 224, 1536, 64, 29, 0.05);
    // M-tail: M not a multiple of 64.
    run_case_bf16(Op::QmmTNax, 40, 256, 1536, 64, 31, 0.05);
    // gs=128 instantiation.
    run_case_bf16(Op::QmmTNax, 64, 256, 1536, 128, 37, 0.05);
}

#[test]
fn qmv_b8_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let (m, n, k, gs) = (1usize, 512usize, 1024usize, 64usize);
    let (packed, scales_f, biases_f, x_f) = make_inputs_b8_f32(23, n, k, m, gs);
    let scales: Vec<half::f16> = scales_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let biases: Vec<half::f16> = biases_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let x: Vec<half::f16> = x_f.iter().map(|&v| half::f16::from_f32(v)).collect();
    let expected = affine_qmm_t_b8_f16(&packed, &scales, &biases, &x, m, n, k, gs);

    let mut stream = MetalStream::new(&device);
    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, as_bytes(&scales));
    let biases_buf = buffer_from_bytes(&device, as_bytes(&biases));
    let x_buf = buffer_from_bytes(&device, as_bytes(&x));
    let y_buf = zeroed_buffer(&device, m * n * 2);
    let cmd_buf = stream.get_command_buffer().expect("cb").clone();
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
    let qmv = MetalAffineQmv::new(device.clone()).expect("MetalAffineQmv");
    qmv.execute(
        &x_buf,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &y_buf,
        m as u32,
        n as u32,
        k as u32,
        1,
        gs as u32,
        8,
        DequantDtype::F16,
        ScaleDtype::F16,
        &encoder,
    )
    .expect("qmv b8 f16 dispatch");
    encoder.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let got: Vec<f32> = unsafe {
        std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const half::f16, m * n)
            .iter()
            .map(|h| h.to_f32())
            .collect()
    };
    let mut max_err = 0f32;
    for i in 0..m * n {
        max_err = max_err.max((got[i] - expected[i].to_f32()).abs());
    }
    assert!(max_err < 0.02, "b8 f16 max_err {max_err}");
}
