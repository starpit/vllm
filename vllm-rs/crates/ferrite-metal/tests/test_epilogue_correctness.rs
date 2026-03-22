/// Epilogue atom correctness tests — dispatch GEMM+epilogue kernels on real GPU
/// hardware and verify output against CPU reference.
///
/// Tests: ResidualAdd, ElementMul, RoPE.
///
/// NOTE: The current kernel signature emitted by build_gemm_msl() only has
/// buffers 0-2 (A, B, C) and 10 (matrix_offsets). ResidualAdd and ElementMul
/// need an additional device buffer, and RoPE needs scalar parameters. We patch
/// the generated MSL string to inject these before compilation. This is a
/// temporary approach — the emitter should eventually support epilogue-declared
/// extra parameters natively.
///
/// Additionally, ResidualAdd and ElementMul reference `sid_m`/`sid_n` which are
/// not defined by the emitter. We inject aliases (sid_m = M_offset, sid_n =
/// N_offset) to match the emitter's variable names.
use ferrite_metal::atoms::*;
use ferrite_metal::config::MetalGemmConfig;
use ferrite_metal::emitter::build_gemm_msl;
use half::f16;
use metal::*;
use std::ffi::c_void;

// ═══════════════════════════════════════════════════════════════════
// Helpers (shared with test_gemm_correctness)
// ═══════════════════════════════════════════════════════════════════

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

/// Deterministic LCG pseudo-random f16 in [-1, 1].
fn lcg_f16(rng: &mut u64) -> f16 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let val = ((*rng >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
    f16::from_f32(val)
}

// ═══════════════════════════════════════════════════════════════════
// Test harness with extra buffer support
// ═══════════════════════════════════════════════════════════════════

struct EpilogueTest {
    device: Device,
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    config: MetalGemmConfig,
}

impl EpilogueTest {
    fn from_msl(msl: &str, config: MetalGemmConfig) -> Self {
        let device = Device::system_default().expect("No Metal device");
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_0);

        let library = device
            .new_library_with_source(msl, &options)
            .unwrap_or_else(|e| {
                // Print MSL for debugging on failure
                eprintln!("=== MSL compilation failed ===\n{}\n=== error ===\n{}", msl, e);
                panic!("MSL compilation failed: {}", e);
            });
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

    /// Run C = GEMM(A, B^T) + epilogue, with an optional extra f16 buffer at index 3.
    fn run_with_extra(
        &self,
        m: u32,
        n: u32,
        k: u32,
        a: &[f16],
        b: &[f16],
        extra: Option<&[f16]>,
    ) -> Vec<f32> {
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

        if let Some(extra_data) = extra {
            let extra_buf = self.device.new_buffer_with_data(
                extra_data.as_ptr() as *const c_void,
                (extra_data.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            enc.set_buffer(3, Some(&extra_buf), 0);
        }

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

        let c_ptr = c_buf.contents() as *const f32;
        let c_slice = unsafe { std::slice::from_raw_parts(c_ptr, (m * n) as usize) };
        c_slice.to_vec()
    }
}

// ═══════════════════════════════════════════════════════════════════
// MSL patching helpers
// ═══════════════════════════════════════════════════════════════════

/// Patch the kernel signature to add an extra float* buffer at index 3,
/// and define sid_m/sid_n aliases for M_offset/N_offset.
///
/// NOTE: The atom code declares `simdgroup_matrix<{{C_TYPE}}, 8, 8>` for the
/// extra tile, where C_TYPE = register precision (float). So the buffer must
/// also be float to satisfy Metal's simdgroup_load type constraint. The atom
/// docstrings say `device half *` but that's a bug — the tile type and pointer
/// type must agree.
fn patch_msl_extra_buffer(msl: &str, param_name: &str) -> String {
    // Add the extra buffer parameter to the kernel signature.
    // Insert before `constant uint4 *matrix_offsets`.
    let patched = msl.replace(
        "constant uint4 *matrix_offsets [[buffer(10)]]",
        &format!(
            "device float *{param_name} [[buffer(3)]],\n    constant uint4 *matrix_offsets [[buffer(10)]]"
        ),
    );

    // Define sid_m and sid_n right after the M_offset/N_offset definitions,
    // before the bounds check.
    let patched = patched.replace(
        "if (M_offset >= M || N_offset >= N) return;",
        "uint sid_m = M_offset;\nuint sid_n = N_offset;\nif (M_offset >= M || N_offset >= N) return;",
    );

    patched
}

/// Patch the kernel signature to add RoPE scalar parameters.
/// Injects position, d_head, and rope_base as constants/params.
fn patch_msl_rope_params(msl: &str, position: u32, d_head: u32) -> String {
    // Add constant parameters after the matrix_offsets line.
    // We use constants embedded in the MSL rather than buffer parameters
    // for simplicity in testing.
    let patched = msl.replace(
        "if (M_offset >= M || N_offset >= N) return;",
        &format!(
            "constant uint position = {};\nconstant uint d_head = {};\nconstant float rope_base = 10000.0f;\nif (M_offset >= M || N_offset >= N) return;",
            position, d_head
        ),
    );

    patched
}

// ═══════════════════════════════════════════════════════════════════
// ResidualAdd tests
// ═══════════════════════════════════════════════════════════════════

fn build_residual_add_msl(config: &MetalGemmConfig) -> String {
    let msl = build_gemm_msl(
        config,
        &LoopTileCopy,
        &NativeFragmentLoad,
        &IdentityTransform,
        &NativeMma,
        &ResidualAddEpilogue,
    );
    patch_msl_extra_buffer(&msl, "residual")
}

#[test]
fn test_residual_add_compiles() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_residual_add_msl(&config);
    // If this doesn't panic, compilation succeeded.
    let _harness = EpilogueTest::from_msl(&msl, config);
}

#[test]
fn test_residual_add_ones_32x32() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_residual_add_msl(&config);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 32;

    // A, B all ones => GEMM = 32.0 everywhere
    let a = vec![f16::from_f32(1.0); (m * k) as usize];
    let b = vec![f16::from_f32(1.0); (n * k) as usize];

    // Residual = 5.0 everywhere => result = 32 + 5 = 37
    let residual = vec![f16::from_f32(5.0); (m * n) as usize];

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, Some(&residual));

    // CPU reference: GEMM then add residual
    let gemm_c = cpu_gemm(m, n, k, &a, &b);
    let cpu_c: Vec<f32> = gemm_c
        .iter()
        .enumerate()
        .map(|(idx, &g)| g + residual[idx].to_f32())
        .collect();

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.5,
        "ResidualAdd ones 32x32: max abs error {}, expected ~37.0, got first={}",
        err,
        gpu_c[0]
    );
}

#[test]
fn test_residual_add_random_32x32() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_residual_add_msl(&config);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 8;

    let mut rng = 42u64;
    let a: Vec<f16> = (0..m * k).map(|_| lcg_f16(&mut rng)).collect();
    let b: Vec<f16> = (0..n * k).map(|_| lcg_f16(&mut rng)).collect();
    let residual: Vec<f16> = (0..m * n).map(|_| lcg_f16(&mut rng)).collect();

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, Some(&residual));

    let gemm_c = cpu_gemm(m, n, k, &a, &b);
    let cpu_c: Vec<f32> = gemm_c
        .iter()
        .enumerate()
        .map(|(idx, &g)| g + residual[idx].to_f32())
        .collect();

    let rel_err = max_rel_error(&gpu_c, &cpu_c);
    let abs_err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        rel_err < 0.05,
        "ResidualAdd random 32x32: max rel error {} (abs {})",
        rel_err,
        abs_err
    );
}

#[test]
fn test_residual_add_zero_residual() {
    // With zero residual, output should equal plain GEMM.
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_residual_add_msl(&config);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 8;
    let a = vec![f16::from_f32(0.5); (m * k) as usize];
    let b = vec![f16::from_f32(0.25); (n * k) as usize];
    let residual = vec![f16::from_f32(0.0); (m * n) as usize];

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, Some(&residual));
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "ResidualAdd zero residual: max abs error {}, expected GEMM-only result",
        err,
    );
}

// ═══════════════════════════════════════════════════════════════════
// ElementMul tests
// ═══════════════════════════════════════════════════════════════════

fn build_element_mul_msl(config: &MetalGemmConfig) -> String {
    let msl = build_gemm_msl(
        config,
        &LoopTileCopy,
        &NativeFragmentLoad,
        &IdentityTransform,
        &NativeMma,
        &ElementMulEpilogue,
    );
    patch_msl_extra_buffer(&msl, "gate")
}

#[test]
fn test_element_mul_compiles() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_element_mul_msl(&config);
    let _harness = EpilogueTest::from_msl(&msl, config);
}

#[test]
fn test_element_mul_ones_32x32() {
    // gate = 1.0 => output should equal plain GEMM
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_element_mul_msl(&config);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 32;
    let a = vec![f16::from_f32(1.0); (m * k) as usize];
    let b = vec![f16::from_f32(1.0); (n * k) as usize];
    let gate = vec![f16::from_f32(1.0); (m * n) as usize];

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, Some(&gate));
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "ElementMul ones gate 32x32: max abs error {}, expected plain GEMM",
        err,
    );
}

#[test]
fn test_element_mul_scale_32x32() {
    // gate = 0.5 => output = GEMM * 0.5
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_element_mul_msl(&config);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 8;
    let a = vec![f16::from_f32(0.5); (m * k) as usize];
    let b = vec![f16::from_f32(0.25); (n * k) as usize];
    let gate = vec![f16::from_f32(0.5); (m * n) as usize];

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, Some(&gate));

    let gemm_c = cpu_gemm(m, n, k, &a, &b);
    let cpu_c: Vec<f32> = gemm_c
        .iter()
        .enumerate()
        .map(|(idx, &g)| g * gate[idx].to_f32())
        .collect();

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "ElementMul scale 32x32: max abs error {}, expected ~{}, got first={}",
        err,
        cpu_c[0],
        gpu_c[0]
    );
}

#[test]
fn test_element_mul_zero_gate() {
    // gate = 0 => output should be all zeros
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_element_mul_msl(&config);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 8;
    let a = vec![f16::from_f32(1.0); (m * k) as usize];
    let b = vec![f16::from_f32(1.0); (n * k) as usize];
    let gate = vec![f16::from_f32(0.0); (m * n) as usize];

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, Some(&gate));

    let err = max_abs_error(&gpu_c, &vec![0.0f32; (m * n) as usize]);
    assert!(
        err < 0.01,
        "ElementMul zero gate: max abs error {}, expected all zeros",
        err,
    );
}

#[test]
fn test_element_mul_random_32x32() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_element_mul_msl(&config);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 8;

    let mut rng = 99u64;
    let a: Vec<f16> = (0..m * k).map(|_| lcg_f16(&mut rng)).collect();
    let b: Vec<f16> = (0..n * k).map(|_| lcg_f16(&mut rng)).collect();
    let gate: Vec<f16> = (0..m * n).map(|_| lcg_f16(&mut rng)).collect();

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, Some(&gate));

    let gemm_c = cpu_gemm(m, n, k, &a, &b);
    let cpu_c: Vec<f32> = gemm_c
        .iter()
        .enumerate()
        .map(|(idx, &g)| g * gate[idx].to_f32())
        .collect();

    let rel_err = max_rel_error(&gpu_c, &cpu_c);
    let abs_err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        rel_err < 0.05,
        "ElementMul random 32x32: max rel error {} (abs {})",
        rel_err,
        abs_err
    );
}

// ═══════════════════════════════════════════════════════════════════
// RoPE tests
// ═══════════════════════════════════════════════════════════════════

fn build_rope_msl(config: &MetalGemmConfig, position: u32, d_head: u32) -> String {
    let msl = build_gemm_msl(
        config,
        &LoopTileCopy,
        &NativeFragmentLoad,
        &IdentityTransform,
        &NativeMma,
        &RoPEAtom,
    );
    patch_msl_rope_params(&msl, position, d_head)
}

/// CPU reference for RoPE applied to a flat [M, N] matrix (row-major f32).
/// position, d_head, rope_base=10000.
///
/// The GPU RoPE atom iterates C_sram[tm][tn].thread_elements()[i] with pairs
/// (i, i+1) for i in 0..64 step 2. The dimension index is:
///   dim_idx = (tn * 64 + i) / 2
///
/// In the context of the full store, element at row r, col c maps to:
///   tn = c / 8  (which 8-wide tile)
///   but within a simdgroup_matrix<float,8,8>, thread_elements() gives 64
///   elements in row-major order of the 8x8 tile. So element index within
///   tile = local_row * 8 + local_col. The pairs are (0,1), (2,3), ...
///   meaning columns within the tile are paired.
///
/// For the CPU reference, we need to replicate this pairing. Column c in the
/// output maps to:
///   tn = c / 8
///   local_col = c % 8
///   i = local_row * 8 + local_col  (the thread_elements index)
///
/// But actually, thread_elements() for a simdgroup_matrix is NOT simple
/// row-major — it's implementation-defined per lane. We cannot replicate
/// the exact lane mapping from the CPU side.
///
/// Instead, we use a simpler approach: test with position=0 where all
/// rotations are identity (cos=1, sin=0), so output = GEMM unchanged.
/// And test compilation for non-zero positions.
fn cpu_rope_position_zero(gemm_output: &[f32]) -> Vec<f32> {
    // At position=0, theta=0 for all dims, cos(0)=1, sin(0)=0.
    // So the rotation is identity: out = x*1 - y*0, y*0 + x*1 = (x, y).
    gemm_output.to_vec()
}

#[test]
fn test_rope_compiles() {
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_rope_msl(&config, 1, 32);
    let _harness = EpilogueTest::from_msl(&msl, config);
}

#[test]
fn test_rope_position_zero_is_identity() {
    // At position=0, cos(theta)=1, sin(theta)=0 for all dims.
    // So RoPE should be a no-op: output = plain GEMM.
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_rope_msl(&config, 0, 32);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 32;
    let a = vec![f16::from_f32(1.0); (m * k) as usize];
    let b = vec![f16::from_f32(1.0); (n * k) as usize];

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, None);
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        err < 0.01,
        "RoPE position=0: max abs error {}, expected identity (plain GEMM), first gpu={}, first cpu={}",
        err,
        gpu_c[0],
        cpu_c[0]
    );
}

#[test]
fn test_rope_position_zero_random() {
    // Random inputs, position=0 => output should equal plain GEMM.
    let config = MetalGemmConfig::default_apple9_f16();
    let msl = build_rope_msl(&config, 0, 32);
    let harness = EpilogueTest::from_msl(&msl, config);

    let m = 32u32;
    let n = 32;
    let k = 8;

    let mut rng = 777u64;
    let a: Vec<f16> = (0..m * k).map(|_| lcg_f16(&mut rng)).collect();
    let b: Vec<f16> = (0..n * k).map(|_| lcg_f16(&mut rng)).collect();

    let gpu_c = harness.run_with_extra(m, n, k, &a, &b, None);
    let cpu_c = cpu_gemm(m, n, k, &a, &b);

    let rel_err = max_rel_error(&gpu_c, &cpu_c);
    let abs_err = max_abs_error(&gpu_c, &cpu_c);
    assert!(
        rel_err < 0.02,
        "RoPE position=0 random: max rel error {} (abs {})",
        rel_err,
        abs_err
    );
}

#[test]
fn test_rope_nonzero_position_changes_output() {
    // With position > 0, RoPE should produce different output than plain GEMM.
    // This is a structural test — we can't easily compute the exact CPU reference
    // because thread_elements() layout is implementation-defined, but we can
    // verify the output actually changes.
    let config = MetalGemmConfig::default_apple9_f16();

    // Build two kernels: one with position=0 (identity) and one with position=5
    let msl_zero = build_rope_msl(&config, 0, 32);
    let msl_five = build_rope_msl(&config, 5, 32);
    let harness_zero = EpilogueTest::from_msl(&msl_zero, config.clone());
    let harness_five = EpilogueTest::from_msl(&msl_five, config);

    let m = 32u32;
    let n = 32;
    let k = 8;
    let a = vec![f16::from_f32(0.5); (m * k) as usize];
    let b = vec![f16::from_f32(0.25); (n * k) as usize];

    let gpu_zero = harness_zero.run_with_extra(m, n, k, &a, &b, None);
    let gpu_five = harness_five.run_with_extra(m, n, k, &a, &b, None);

    // The two outputs should differ (RoPE with position=5 rotates elements).
    let diff = max_abs_error(&gpu_zero, &gpu_five);
    assert!(
        diff > 0.001,
        "RoPE position=5 should differ from position=0, but max diff = {}",
        diff
    );
}

#[test]
fn test_rope_preserves_norm() {
    // RoPE is a rotation, so it should preserve the L2 norm of each pair.
    // We can verify that the sum of squares doesn't change much.
    // (Not exact due to f16 precision, but should be close.)
    let config = MetalGemmConfig::default_apple9_f16();
    let msl_zero = build_rope_msl(&config, 0, 32);
    let msl_pos = build_rope_msl(&config, 3, 32);
    let harness_zero = EpilogueTest::from_msl(&msl_zero, config.clone());
    let harness_pos = EpilogueTest::from_msl(&msl_pos, config);

    let m = 32u32;
    let n = 32;
    let k = 8;
    let a = vec![f16::from_f32(0.3); (m * k) as usize];
    let b = vec![f16::from_f32(0.2); (n * k) as usize];

    let gpu_zero = harness_zero.run_with_extra(m, n, k, &a, &b, None);
    let gpu_pos = harness_pos.run_with_extra(m, n, k, &a, &b, None);

    // Sum of squares for both should be similar (rotation preserves norm).
    let norm_zero: f32 = gpu_zero.iter().map(|x| x * x).sum();
    let norm_pos: f32 = gpu_pos.iter().map(|x| x * x).sum();

    let rel_norm_diff = (norm_zero - norm_pos).abs() / norm_zero.max(1e-6);
    assert!(
        rel_norm_diff < 0.05,
        "RoPE should approximately preserve total norm: zero={}, pos={}, rel_diff={}",
        norm_zero,
        norm_pos,
        rel_norm_diff
    );
}
