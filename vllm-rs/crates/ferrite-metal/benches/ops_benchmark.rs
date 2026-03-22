/// Operation-level benchmarks — measures each Ferrite op at realistic sizes.
///
/// Run with: cargo bench -p ferrite-metal --bench ops_benchmark
///
/// Compares wall-clock GPU time for:
/// - GEMM at various sizes (square, tall-skinny for decode)
/// - FlashAttention at various seq_len × d_head
/// - RmsNorm at d_model=4096
/// - Convert f32↔f16
///
/// Each benchmark: compile once, warm up, then measure N iterations.
use ferrite_metal::atoms::*;
use ferrite_metal::attention_emitter::{AttentionConfig, build_attention_msl};
use ferrite_metal::config::{MetalGemmConfig, Precision};
use ferrite_metal::emitter::{build_gemm_msl, build_standalone_gemm};
use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const WARMUP: u32 = 5;
const ITERS: u32 = 50;

fn get_device() -> (Device, CommandQueue) {
    let d = Device::system_default().expect("No Metal device");
    let q = d.new_command_queue();
    (d, q)
}

fn compile(device: &Device, msl: &str, name: &str) -> ComputePipelineState {
    let opts = CompileOptions::new();
    opts.set_language_version(MTLLanguageVersion::V3_0);
    let lib = device
        .new_library_with_source(msl, &opts)
        .unwrap_or_else(|e| panic!("Compile {}: {}", name, e));
    let func = lib
        .get_function(name, None)
        .unwrap_or_else(|e| panic!("{}: {}", name, e));
    device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipeline")
}

fn empty_f16(d: &Device, n: usize) -> Buffer {
    d.new_buffer((n * 2) as u64, MTLResourceOptions::StorageModeShared)
}
fn empty_f32(d: &Device, n: usize) -> Buffer {
    d.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared)
}

/// Time N dispatches of a closure that encodes GPU work.
fn bench_gpu<F: Fn(&CommandBufferRef)>(queue: &CommandQueue, warmup: u32, iters: u32, f: F) -> f64 {
    // Warmup
    for _ in 0..warmup {
        let cmd = queue.new_command_buffer();
        f(cmd);
        cmd.commit();
        cmd.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let cmd = queue.new_command_buffer();
        f(cmd);
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / iters as f64
}

// ═══════════════════════════════════════════════════════════════════
// GEMM benchmarks
// ═══════════════════════════════════════════════════════════════════

fn bench_gemm_config(
    device: &Device,
    queue: &CommandQueue,
    m: u32,
    n: u32,
    k: u32,
    config: &MetalGemmConfig,
) -> f64 {
    let config = config.clone();
    let msl = build_standalone_gemm(&config);
    let pipeline = compile(device, &msl, "gemm");

    let a_buf = empty_f16(device, (m * k) as usize);
    let b_buf = empty_f16(device, (n * k) as usize);
    let c_rows = (m as u64).max(config.block_m as u64);
    let c_cols = (n as u64).max(config.block_n as u64);
    let c_buf = device.new_buffer(c_rows * c_cols * 4, MTLResourceOptions::StorageModeShared);
    let offsets: [u32; 4] = [m, n, k, 0];
    let offsets_buf = device.new_buffer_with_data(
        offsets.as_ptr() as *const c_void,
        16,
        MTLResourceOptions::StorageModeShared,
    );

    let m_group = config.block_m as u64;
    let n_group = config.block_n as u64;
    let grid = MTLSize::new(
        (n as u64 + n_group - 1) / n_group,
        (m as u64 + m_group - 1) / m_group,
        1,
    );
    let tg = MTLSize::new(config.threadgroup_size() as u64, 1, 1);

    bench_gpu(queue, WARMUP, ITERS, |cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_buffer(10, Some(&offsets_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    })
}

// ═══════════════════════════════════════════════════════════════════
// Attention benchmarks
// ═══════════════════════════════════════════════════════════════════

fn bench_attention(device: &Device, queue: &CommandQueue, seq_len: u32, d_head: u16) -> f64 {
    let block = ((seq_len as u16 + 7) / 8) * 8;
    let config = AttentionConfig {
        block_r: block.min(32),
        block_c: block.min(32),
        d_head,
        num_heads: 1,
        causal: false,
        memory_precision: Precision::FP16,
        accumulator_precision: Precision::FP32,
    };
    let msl = build_attention_msl(&config);
    let pipeline = compile(device, &msl, "attention");

    let n = (seq_len * d_head as u32) as usize;
    let q_buf = empty_f16(device, n);
    let k_buf = empty_f16(device, n);
    let v_buf = empty_f16(device, n);
    let o_buf = empty_f32(device, n);
    let params: [u32; 4] = [seq_len, d_head as u32, 1, 0];
    let params_buf = device.new_buffer_with_data(
        params.as_ptr() as *const c_void,
        16,
        MTLResourceOptions::StorageModeShared,
    );

    let grid_y = (seq_len as u64 + config.block_r as u64 - 1) / config.block_r as u64;
    let grid = MTLSize::new(1, grid_y, 1);
    let tg = MTLSize::new(32, 1, 1);

    bench_gpu(queue, WARMUP, ITERS, |cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, Some(&o_buf), 0);
        enc.set_buffer(10, Some(&params_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    })
}

// ═══════════════════════════════════════════════════════════════════
// RmsNorm benchmark
// ═══════════════════════════════════════════════════════════════════

fn bench_rmsnorm(device: &Device, queue: &CommandQueue, rows: u32, hidden: u32) -> f64 {
    let atom = RmsNormAtom::new(1e-5, "float", 4);
    let msl = atom.emit_kernel(ferrite_metal::msl_builder::MslBuilder::new());
    let pipeline = compile(device, &msl, "rmsnorm");

    let input_buf = empty_f32(device, (rows * hidden) as usize);
    let gamma_buf = empty_f32(device, hidden as usize);
    let output_buf = empty_f32(device, (rows * hidden) as usize);
    let hs_buf = device.new_buffer_with_data(
        &hidden as *const u32 as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new(rows as u64, 1, 1);
    let tg = MTLSize::new(128, 1, 1); // 4 simdgroups

    bench_gpu(queue, WARMUP, ITERS, |cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&input_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_buffer(2, Some(&output_buf), 0);
        enc.set_buffer(3, Some(&hs_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    })
}

// ═══════════════════════════════════════════════════════════════════
// Convert benchmark
// ═══════════════════════════════════════════════════════════════════

fn bench_convert(device: &Device, queue: &CommandQueue, count: u32) -> f64 {
    let msl = ConvertAtom::emit_kernel("float", "half");
    let pipeline = compile(device, &msl, "convert");

    let input_buf = empty_f32(device, count as usize);
    let output_buf = empty_f16(device, count as usize);
    let count_buf = device.new_buffer_with_data(
        &count as *const u32 as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let tg_size = 256u64;
    let grid = MTLSize::new((count as u64 + tg_size - 1) / tg_size, 1, 1);
    let tg = MTLSize::new(tg_size, 1, 1);

    bench_gpu(queue, WARMUP, ITERS, |cmd| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&input_buf), 0);
        enc.set_buffer(1, Some(&output_buf), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    })
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let (device, queue) = get_device();
    eprintln!("Device: {}", device.name());
    eprintln!("Warmup: {}, Iterations: {}", WARMUP, ITERS);
    eprintln!();

    // GEMM benchmarks — compare tile configs
    let gemm_sizes: Vec<(u32, u32, u32)> = vec![
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (1, 4096, 4096),
        (512, 4096, 4096),
    ];

    let apple9 = MetalGemmConfig::default_apple9_f16(); // 32×32×8
    let apple8 = MetalGemmConfig::default_apple8_f16(); // 48×48×32
    let mut k32 = MetalGemmConfig::default_apple9_f16();
    k32.block_k = 32;
    k32.leading_block_dims = None;

    // 4 simdgroups: 64×64 output tile, K=32, each simdgroup does 32×32
    let mut multi_sg = MetalGemmConfig::default_apple9_f16();
    multi_sg.block_m = 64;
    multi_sg.block_n = 64;
    multi_sg.block_k = 32;
    multi_sg.splits = [2, 2]; // 4 simdgroups
    multi_sg.leading_block_dims = None;

    // 16 simdgroups: 128×128 output tile, K=32
    let mut big = MetalGemmConfig::default_apple9_f16();
    big.block_m = 128;
    big.block_n = 128;
    big.block_k = 32;
    big.splits = [4, 4]; // 16 simdgroups
    big.leading_block_dims = None;

    for (label, cfg) in [
        ("32×32×8  1sg", &apple9),
        ("32×32×32 1sg", &k32),
        ("64×64×32 4sg", &multi_sg),
        ("128×128×32 16sg", &big),
    ] {
        eprintln!("=== GEMM {} ===", label);
        for &(m, n, k) in &gemm_sizes {
            let t = bench_gemm_config(&device, &queue, m, n, k, cfg);
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let tflops = flops / t / 1e12;
            eprintln!(
                "  {:>5}×{:<5} K={:<5}  {:.3} ms  ({:.1} TFLOPS)",
                m,
                n,
                k,
                t * 1e3,
                tflops
            );
        }
        eprintln!();
    }
    eprintln!();

    // Attention benchmarks (d_head=64 to fit threadgroup memory)
    eprintln!("=== FlashAttention (f16, 1 head) ===");
    for &(seq, dh) in &[(32, 64), (128, 64), (512, 64)] {
        let t = bench_attention(&device, &queue, seq, dh);
        eprintln!("  seq={:<5} d_head={:<4}  {:.3} ms", seq, dh, t * 1e3);
    }
    eprintln!();

    // RmsNorm benchmarks
    eprintln!("=== RmsNorm (f32) ===");
    for &(rows, hidden) in &[(1, 4096), (32, 4096), (512, 4096)] {
        let t = bench_rmsnorm(&device, &queue, rows, hidden);
        let gb = rows as f64 * hidden as f64 * 4.0 * 2.0 / 1e9; // read + write
        let bw = gb / t;
        eprintln!(
            "  {:>4} rows × {:<5}  {:.3} ms  ({:.1} GB/s)",
            rows,
            hidden,
            t * 1e3,
            bw
        );
    }
    eprintln!();

    // Convert benchmarks
    eprintln!("=== Convert f32→f16 ===");
    for &count in &[4096 * 4096, 512 * 4096, 1 * 4096] {
        let t = bench_convert(&device, &queue, count);
        let gb = count as f64 * 6.0 / 1e9; // 4 bytes read + 2 bytes write
        let bw = gb / t;
        eprintln!("  {:>10} elems  {:.3} ms  ({:.1} GB/s)", count, t * 1e3, bw);
    }
}
