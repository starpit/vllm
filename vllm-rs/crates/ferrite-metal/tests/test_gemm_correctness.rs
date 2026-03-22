/// GEMM correctness tests — dispatch the kernel on real GPU hardware and
/// verify output against CPU reference.
use ferrite_metal::atoms::*;
use ferrite_metal::config::MetalGemmConfig;
use ferrite_metal::emitter::{build_gemm_msl, build_standalone_gemm};
use half::f16;
use metal::*;
use std::ffi::c_void;

// ═══════════════════════════════════════════════════════════════════
// Test harness
// ═══════════════════════════════════════════════════════════════════

struct GemmTest {
    device: Device,
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    config: MetalGemmConfig,
}

impl GemmTest {
    fn new(config: MetalGemmConfig) -> Self {
        let msl = build_standalone_gemm(&config);
        Self::from_msl(&msl, config)
    }

    fn from_msl(msl: &str, config: MetalGemmConfig) -> Self {
        let device = Device::system_default().expect("No Metal device");
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_0);

        let library = device
            .new_library_with_source(msl, &options)
            .expect("MSL compilation failed");
        let func = library
            .get_function("gemm", None)
            .expect("gemm function not found");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("Pipeline creation failed");
        let queue = device.new_command_queue();

        Self {
            device,
            pipeline,
            queue,
            config,
        }
    }

    /// Run C = A @ B^T (since B is transposed in our default config).
    /// A: [M, K] row-major f16
    /// B: [N, K] row-major f16 (stored as column-major K×N, i.e. B^T)
    /// C: [M, N] row-major f32
    fn run(&self, m: u32, n: u32, k: u32, a: &[f16], b: &[f16]) -> Vec<f32> {
        assert_eq!(a.len(), (m * k) as usize);
        assert_eq!(b.len(), (n * k) as usize);

        let a_buf = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (a.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (b.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buf = self
            .device
            .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        // matrix_offsets: uint4 with (M, N, K, 0)
        let offsets: [u32; 4] = [m, n, k, 0];
        let offsets_buf = self.device.new_buffer_with_data(
            offsets.as_ptr() as *const c_void,
            16,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_buffer(10, Some(&offsets_buf), 0);

        let m_group = self.config.block_m as u64;
        let n_group = self.config.block_n as u64;
        let grid = MTLSize::new(
            (n as u64 + n_group - 1) / n_group,
            (m as u64 + m_group - 1) / m_group,
            1,
        );
        let tg_size = MTLSize::new(self.config.threadgroup_size() as u64, 1, 1);

        enc.dispatch_thread_groups(grid, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Read back C
        let c_ptr = c_buf.contents() as *const f32;
        let c_slice = unsafe { std::slice::from_raw_parts(c_ptr, (m * n) as usize) };
        c_slice.to_vec()
    }
}

/// CPU reference: C = A @ B^T where A is [M,K] and B is [N,K].
fn cpu_gemm(m: u32, n: u32, k: u32, a: &[f16], b: &[f16]) -> Vec<f32> {
    let mut c = vec![0.0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let a_val = a[(i * k + kk) as usize].to_f32();
                let b_val = b[(j * k + kk) as usize].to_f32();
                sum += a_val * b_val;
            }
            c[(i * n + j) as usize] = sum;
        }
    }
    c
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_rel_error(gpu: &[f32], cpu: &[f32]) -> f32 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|(g, c)| {
            if c.abs() < 1e-6 {
                (g - c).abs()
            } else {
                (g - c).abs() / c.abs()
            }
        })
        .fold(0.0f32, f32::max)
}

// ═══════════════════════════════════════════════════════════════════
// Correctness tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_ones_32x32() {
    let mut config = MetalGemmConfig::default_apple9_f16();
    config.prefer_async_load = false;
    let harness = GemmTest::new(config);

    let m = 32u32;
    let n = 32;
    let k = 32;
    let a = vec![f16::from_f32(1.0); (m * k) as usize];
    let b = vec![f16::from_f32(1.0); (n * k) as usize];

    let gpu_c = harness.run(m, n, k, &a, &b);
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "ones 32×32: max abs error {}, expected all {}",
        err,
        k as f32
    );
}

#[test]
fn test_identity_pattern_32x32() {
    // A = sequential rows, B = identity-like → C should reproduce A's rows
    let mut config = MetalGemmConfig::default_apple9_f16();
    config.prefer_async_load = false;
    let harness = GemmTest::new(config);

    let m = 32u32;
    let n = 32;
    let k = 32;

    // A[i,k] = (i * k_dim + k) as f16 (small values to avoid overflow)
    let a: Vec<f16> = (0..m * k)
        .map(|idx| f16::from_f32((idx as f32) * 0.01))
        .collect();

    // B = identity: B[j,k] = 1.0 if j==k else 0.0
    let mut b = vec![f16::from_f32(0.0); (n * k) as usize];
    for i in 0..n.min(k) {
        b[(i * k + i) as usize] = f16::from_f32(1.0);
    }

    let gpu_c = harness.run(m, n, k, &a, &b);
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(err < 0.1, "identity 32×32: max abs error {}", err);
}

#[test]
fn test_random_64x64() {
    // Larger than one tile — tests edge handling and multi-tile dispatch
    let mut config = MetalGemmConfig::default_apple9_f16();
    config.prefer_async_load = false;
    let harness = GemmTest::new(config);

    let m = 64u32;
    let n = 64;
    let k = 64;

    // Deterministic pseudo-random using simple LCG
    let mut rng = 12345u64;
    let next_f16 = |rng: &mut u64| -> f16 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Map to [-1, 1] range
        let val = ((*rng >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
        f16::from_f32(val)
    };

    let a: Vec<f16> = (0..m * k).map(|_| next_f16(&mut rng)).collect();
    let b: Vec<f16> = (0..n * k).map(|_| next_f16(&mut rng)).collect();

    let gpu_c = harness.run(m, n, k, &a, &b);
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let rel_err = max_rel_error(&gpu_c, &cpu_c);
    let abs_err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        rel_err < 0.02,
        "random 64×64: max relative error {} (abs {})",
        rel_err,
        abs_err
    );
}

#[test]
fn test_non_square_48x64_k32() {
    let mut config = MetalGemmConfig::default_apple9_f16();
    config.prefer_async_load = false;
    let harness = GemmTest::new(config);

    let m = 48u32;
    let n = 64;
    let k = 32;

    let a = vec![f16::from_f32(0.5); (m * k) as usize];
    let b = vec![f16::from_f32(0.25); (n * k) as usize];

    let gpu_c = harness.run(m, n, k, &a, &b);
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    // Expected: each element = 0.5 * 0.25 * 32 = 4.0
    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.1,
        "non-square 48×64 k=32: max abs error {}, expected 4.0, got first={}",
        err,
        gpu_c[0]
    );
}

#[test]
fn test_apple8_config_32x32() {
    // Verify the apple8 (48×48×32 tiles) config also produces correct results
    let config = MetalGemmConfig::default_apple8_f16();
    let harness = GemmTest::new(config);

    let m = 48u32;
    let n = 48;
    let k = 32;
    let a = vec![f16::from_f32(1.0); (m * k) as usize];
    let b = vec![f16::from_f32(1.0); (n * k) as usize];

    let gpu_c = harness.run(m, n, k, &a, &b);
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "apple8 ones 48×48: max abs error {}, expected all {}",
        err,
        k as f32
    );
}

// ═══════════════════════════════════════════════════════════════════
// SiLU epilogue tests
// ═══════════════════════════════════════════════════════════════════

fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn test_gemm_silu_32x32() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_gemm_msl(
        &config,
        &LoopTileCopy,
        &NativeFragmentLoad,
        &IdentityTransform,
        &NativeMma,
        &SiLuEpilogue,
    );
    let harness = GemmTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 32;
    let a = vec![f16::from_f32(0.1); (m * k) as usize];
    let b = vec![f16::from_f32(0.1); (n * k) as usize];

    let gpu_c = harness.run(m, n, k, &a, &b);

    // CPU reference: GEMM then SiLU
    let gemm_c = cpu_gemm(m, n, k, &a, &b);
    let cpu_c: Vec<f32> = gemm_c.iter().map(|&x| cpu_silu(x)).collect();

    // GEMM result: 0.1 * 0.1 * 32 = 0.32, SiLU(0.32) ≈ 0.1715
    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "GEMM+SiLU 32×32: max abs error {}, expected ~{}, got first={}",
        err,
        cpu_c[0],
        gpu_c[0]
    );
}

#[test]
fn test_gemm_silu_negative_values() {
    // SiLU with negative inputs (GEMM result can be negative with mixed signs)
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_gemm_msl(
        &config,
        &LoopTileCopy,
        &NativeFragmentLoad,
        &IdentityTransform,
        &NativeMma,
        &SiLuEpilogue,
    );
    let harness = GemmTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 8;

    // A = -0.5 everywhere, B = 1.0 everywhere → GEMM = -4.0, SiLU(-4.0) ≈ -0.0713
    let a = vec![f16::from_f32(-0.5); (m * k) as usize];
    let b = vec![f16::from_f32(1.0); (n * k) as usize];

    let gpu_c = harness.run(m, n, k, &a, &b);
    let gemm_c = cpu_gemm(m, n, k, &a, &b);
    let cpu_c: Vec<f32> = gemm_c.iter().map(|&x| cpu_silu(x)).collect();

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "GEMM+SiLU negative: max abs error {}, expected ~{}, got first={}",
        err,
        cpu_c[0],
        gpu_c[0]
    );
}
