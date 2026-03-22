/// RmsNorm GPU correctness tests — dispatch the generated MSL kernel on real
/// Apple GPU hardware and verify output against a CPU reference implementation.
///
/// CPU reference:
///   rms = sqrt(mean(x^2) + eps)
///   output[i] = (x[i] / rms) * gamma[i]
use ferrite_metal::atoms::RmsNormAtom;
use ferrite_metal::msl_builder::MslBuilder;
use half::f16;
use metal::*;
use std::ffi::c_void;

// ═══════════════════════════════════════════════════════════════════
// Test harness
// ═══════════════════════════════════════════════════════════════════

const EPS: f32 = 1e-5;
const SIMDGROUPS_PER_TG: u32 = 4;

struct RmsNormTest {
    device: Device,
    pipeline: ComputePipelineState,
    queue: CommandQueue,
}

impl RmsNormTest {
    fn new(dtype: &str, eps: f32, simdgroups: u32) -> Self {
        let atom = RmsNormAtom::new(eps, dtype, simdgroups);
        let msl_source = atom.emit_kernel(MslBuilder::new());

        let device = Device::system_default().expect("No Metal device");
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_0);

        let library = device
            .new_library_with_source(&msl_source, &options)
            .unwrap_or_else(|e| {
                panic!("MSL compilation failed: {}\n\nSource:\n{}", e, msl_source);
            });
        let func = library
            .get_function("rmsnorm", None)
            .expect("rmsnorm function not found");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("Pipeline creation failed");
        let queue = device.new_command_queue();

        Self {
            device,
            pipeline,
            queue,
        }
    }

    /// Run RmsNorm on GPU.
    ///
    /// `input`: [num_rows, hidden_size] flattened f32 values
    /// `gamma`: [hidden_size] f32 values
    ///
    /// Returns: [num_rows, hidden_size] flattened f32 output
    fn run_f32(&self, num_rows: u32, hidden_size: u32, input: &[f32], gamma: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), (num_rows * hidden_size) as usize);
        assert_eq!(gamma.len(), hidden_size as usize);

        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (input.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let gamma_buf = self.device.new_buffer_with_data(
            gamma.as_ptr() as *const c_void,
            (gamma.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self.device.new_buffer(
            (num_rows * hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hidden_size_buf = self.device.new_buffer_with_data(
            &hidden_size as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&input_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_buffer(2, Some(&output_buf), 0);
        enc.set_buffer(3, Some(&hidden_size_buf), 0);

        // One threadgroup per row, each threadgroup = SIMDGROUPS_PER_TG * 32 threads
        let threads_per_tg = SIMDGROUPS_PER_TG as u64 * 32;
        let grid = MTLSize::new(num_rows as u64, 1, 1);
        let tg_size = MTLSize::new(threads_per_tg, 1, 1);

        enc.dispatch_thread_groups(grid, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = output_buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, (num_rows * hidden_size) as usize) };
        slice.to_vec()
    }

    /// Run RmsNorm on GPU with f16 input/gamma, read back as f16 then convert to f32.
    fn run_f16(
        &self,
        num_rows: u32,
        hidden_size: u32,
        input: &[f16],
        gamma: &[f16],
    ) -> Vec<f32> {
        assert_eq!(input.len(), (num_rows * hidden_size) as usize);
        assert_eq!(gamma.len(), hidden_size as usize);

        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (input.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let gamma_buf = self.device.new_buffer_with_data(
            gamma.as_ptr() as *const c_void,
            (gamma.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self.device.new_buffer(
            (num_rows * hidden_size * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hidden_size_buf = self.device.new_buffer_with_data(
            &hidden_size as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&input_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_buffer(2, Some(&output_buf), 0);
        enc.set_buffer(3, Some(&hidden_size_buf), 0);

        let threads_per_tg = SIMDGROUPS_PER_TG as u64 * 32;
        let grid = MTLSize::new(num_rows as u64, 1, 1);
        let tg_size = MTLSize::new(threads_per_tg, 1, 1);

        enc.dispatch_thread_groups(grid, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = output_buf.contents() as *const f16;
        let slice = unsafe { std::slice::from_raw_parts(ptr, (num_rows * hidden_size) as usize) };
        slice.iter().map(|v| v.to_f32()).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// CPU reference
// ═══════════════════════════════════════════════════════════════════

/// CPU RmsNorm reference: for each row, compute
///   rms = sqrt(mean(x^2) + eps)
///   output[i] = (x[i] / rms) * gamma[i]
fn cpu_rmsnorm(input: &[f32], gamma: &[f32], hidden_size: usize, eps: f32) -> Vec<f32> {
    let num_rows = input.len() / hidden_size;
    let mut output = vec![0.0f32; input.len()];
    for row in 0..num_rows {
        let offset = row * hidden_size;
        let row_data = &input[offset..offset + hidden_size];

        // sum of squares
        let sum_sq: f32 = row_data.iter().map(|x| x * x).sum();
        let mean_sq = sum_sq / hidden_size as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();

        for i in 0..hidden_size {
            output[offset + i] = row_data[i] * scale * gamma[i];
        }
    }
    output
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
// Correctness tests (float dtype)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_rmsnorm_ones() {
    let harness = RmsNormTest::new("float", EPS, SIMDGROUPS_PER_TG);

    let hidden_size = 128u32;
    let num_rows = 1u32;
    let input = vec![1.0f32; (num_rows * hidden_size) as usize];
    let gamma = vec![1.0f32; hidden_size as usize];

    let gpu_out = harness.run_f32(num_rows, hidden_size, &input, &gamma);
    let cpu_out = cpu_rmsnorm(&input, &gamma, hidden_size as usize, EPS);

    // For x=1.0 everywhere: mean(x^2) = 1.0, rms = sqrt(1+eps) ≈ 1.0
    // output ≈ 1.0 / 1.0 * 1.0 = 1.0
    let err = max_abs_error(&gpu_out, &cpu_out);
    assert!(
        err < 1e-4,
        "rmsnorm_ones: max abs error {}, gpu[0]={}, cpu[0]={}",
        err,
        gpu_out[0],
        cpu_out[0]
    );

    // Verify output is approximately 1.0
    for (i, &v) in gpu_out.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-3,
            "rmsnorm_ones: output[{}] = {}, expected ~1.0",
            i,
            v
        );
    }
}

#[test]
fn test_rmsnorm_with_gamma() {
    let harness = RmsNormTest::new("float", EPS, SIMDGROUPS_PER_TG);

    let hidden_size = 128u32;
    let num_rows = 1u32;
    let input = vec![1.0f32; (num_rows * hidden_size) as usize];

    // gamma varies: cycle through 0.5, 1.0, 2.0, 3.0
    let gamma_values = [0.5f32, 1.0, 2.0, 3.0];
    let gamma: Vec<f32> = (0..hidden_size as usize)
        .map(|i| gamma_values[i % gamma_values.len()])
        .collect();

    let gpu_out = harness.run_f32(num_rows, hidden_size, &input, &gamma);
    let cpu_out = cpu_rmsnorm(&input, &gamma, hidden_size as usize, EPS);

    let err = max_abs_error(&gpu_out, &cpu_out);
    assert!(
        err < 1e-4,
        "rmsnorm_with_gamma: max abs error {}, gpu[0]={}, cpu[0]={}",
        err,
        gpu_out[0],
        cpu_out[0]
    );

    // With x=1.0, rms ≈ 1.0, so output[i] ≈ gamma[i]
    // Verify the gamma scaling pattern is preserved
    for i in 0..hidden_size as usize {
        let expected = gamma[i]; // approximately, since rms ≈ 1.0
        assert!(
            (gpu_out[i] - expected).abs() < 1e-3,
            "rmsnorm_with_gamma: output[{}] = {}, expected ~{}",
            i,
            gpu_out[i],
            expected
        );
    }
}

#[test]
fn test_rmsnorm_negative_values() {
    let harness = RmsNormTest::new("float", EPS, SIMDGROUPS_PER_TG);

    let hidden_size = 128u32;
    let num_rows = 1u32;

    // Alternating positive and negative values
    let input: Vec<f32> = (0..hidden_size as usize)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let gamma = vec![1.0f32; hidden_size as usize];

    let gpu_out = harness.run_f32(num_rows, hidden_size, &input, &gamma);
    let cpu_out = cpu_rmsnorm(&input, &gamma, hidden_size as usize, EPS);

    let err = max_abs_error(&gpu_out, &cpu_out);
    assert!(
        err < 1e-4,
        "rmsnorm_negative: max abs error {}, gpu[0]={}, cpu[0]={}",
        err,
        gpu_out[0],
        cpu_out[0]
    );

    // Verify signs are preserved: even indices positive, odd indices negative
    for i in 0..hidden_size as usize {
        if i % 2 == 0 {
            assert!(
                gpu_out[i] > 0.0,
                "rmsnorm_negative: output[{}] = {} should be positive",
                i,
                gpu_out[i]
            );
        } else {
            assert!(
                gpu_out[i] < 0.0,
                "rmsnorm_negative: output[{}] = {} should be negative",
                i,
                gpu_out[i]
            );
        }
    }

    // Also check magnitude: all absolute values should be equal (since |x|=1 everywhere)
    let abs_val = gpu_out[0].abs();
    for (i, &v) in gpu_out.iter().enumerate() {
        assert!(
            (v.abs() - abs_val).abs() < 1e-4,
            "rmsnorm_negative: |output[{}]| = {}, expected {}",
            i,
            v.abs(),
            abs_val
        );
    }
}

#[test]
fn test_rmsnorm_varying_input() {
    let harness = RmsNormTest::new("float", EPS, SIMDGROUPS_PER_TG);

    let hidden_size = 128u32;
    let num_rows = 1u32;

    // x = [0.1, 0.2, 0.3, ...] — sequential increasing values
    let input: Vec<f32> = (0..hidden_size as usize)
        .map(|i| (i + 1) as f32 * 0.1)
        .collect();
    let gamma = vec![1.0f32; hidden_size as usize];

    let gpu_out = harness.run_f32(num_rows, hidden_size, &input, &gamma);
    let cpu_out = cpu_rmsnorm(&input, &gamma, hidden_size as usize, EPS);

    let abs_err = max_abs_error(&gpu_out, &cpu_out);
    let rel_err = max_rel_error(&gpu_out, &cpu_out);
    assert!(
        rel_err < 1e-4,
        "rmsnorm_varying: max rel error {}, abs error {}, gpu[0]={}, cpu[0]={}",
        rel_err,
        abs_err,
        gpu_out[0],
        cpu_out[0]
    );
}

// ═══════════════════════════════════════════════════════════════════
// Half-precision tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_rmsnorm_ones_f16() {
    let harness = RmsNormTest::new("half", EPS, SIMDGROUPS_PER_TG);

    let hidden_size = 128u32;
    let num_rows = 1u32;
    let input = vec![f16::from_f32(1.0); (num_rows * hidden_size) as usize];
    let gamma = vec![f16::from_f32(1.0); hidden_size as usize];

    let gpu_out = harness.run_f16(num_rows, hidden_size, &input, &gamma);

    // CPU reference in f32 for comparison
    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    let gamma_f32: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
    let cpu_out = cpu_rmsnorm(&input_f32, &gamma_f32, hidden_size as usize, EPS);

    let err = max_abs_error(&gpu_out, &cpu_out);
    assert!(
        err < 5e-3,
        "rmsnorm_ones_f16: max abs error {}, gpu[0]={}, cpu[0]={}",
        err,
        gpu_out[0],
        cpu_out[0]
    );
}

#[test]
fn test_rmsnorm_varying_input_f16() {
    let harness = RmsNormTest::new("half", EPS, SIMDGROUPS_PER_TG);

    let hidden_size = 128u32;
    let num_rows = 1u32;

    let input: Vec<f16> = (0..hidden_size as usize)
        .map(|i| f16::from_f32((i + 1) as f32 * 0.1))
        .collect();
    let gamma = vec![f16::from_f32(1.0); hidden_size as usize];

    let gpu_out = harness.run_f16(num_rows, hidden_size, &input, &gamma);

    let input_f32: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    let gamma_f32: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
    let cpu_out = cpu_rmsnorm(&input_f32, &gamma_f32, hidden_size as usize, EPS);

    // f16 has less precision, allow larger tolerance
    let rel_err = max_rel_error(&gpu_out, &cpu_out);
    assert!(
        rel_err < 5e-3,
        "rmsnorm_varying_f16: max rel error {}, gpu[0]={}, cpu[0]={}",
        rel_err,
        gpu_out[0],
        cpu_out[0]
    );
}

// ═══════════════════════════════════════════════════════════════════
// Multi-row test
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_rmsnorm_multi_row() {
    let harness = RmsNormTest::new("float", EPS, SIMDGROUPS_PER_TG);

    let hidden_size = 128u32;
    let num_rows = 4u32;

    // Each row has different values so each row gets a different RMS
    let mut input = Vec::with_capacity((num_rows * hidden_size) as usize);
    for row in 0..num_rows {
        let scale = (row + 1) as f32;
        for i in 0..hidden_size {
            input.push(scale * ((i as f32 + 1.0) * 0.01));
        }
    }
    let gamma = vec![1.0f32; hidden_size as usize];

    let gpu_out = harness.run_f32(num_rows, hidden_size, &input, &gamma);
    let cpu_out = cpu_rmsnorm(&input, &gamma, hidden_size as usize, EPS);

    let rel_err = max_rel_error(&gpu_out, &cpu_out);
    let abs_err = max_abs_error(&gpu_out, &cpu_out);
    assert!(
        rel_err < 1e-4,
        "rmsnorm_multi_row: max rel error {}, abs error {}",
        rel_err,
        abs_err
    );
}
