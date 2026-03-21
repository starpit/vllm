//! Benchmark: Fused MLP block (proc macro) vs. unfused (4 separate launches).
//!
//! Compares:
//!   1. Fused: TWO-STAGE proc macro (RmsNorm+GEMM+SiLU -> CVT -> GEMM = 3 launches)
//!   2. Unfused: 4 separate kernel launches (RMSNorm, GEMM, SiLU, GEMM)
//!
//! The fused version should be faster due to:
//!   - Fewer kernel launches (3 vs 4)
//!   - Stage 1 fuses RMSNorm+GEMM+SiLU into a single kernel
//!   - Better L2 cache locality for intermediates
//!
//! Usage:
//!   cargo run --release --bin ferrite-mlp-bench

use cudarc::driver::result as cuda;
use cudarc::driver::sys as cuda_sys;
use std::ffi::{CString, c_void};
use std::time::Instant;

type DevicePtr = u64;

// The fused MLP block via proc macro — 2-stage plan
#[ferrite_macros::fuse(arch = "sm_89")]
fn mlp_fwd(
    input: DevicePtr,
    w_norm: DevicePtr,
    w_gate: DevicePtr,
    w_down: DevicePtr,
    output: DevicePtr,
    batch: u32,
    hidden: u32,
    inter: u32,
    out: u32,
) {
    let n = rmsnorm(input, w_norm);
    let g = gemm(n, w_gate);
    let h = silu(g);
    gemm(h, w_down)
}

/// Launch a standalone RMSNorm kernel.
fn launch_standalone_rmsnorm(
    f: cuda_sys::CUfunction,
    d_output: cuda_sys::CUdeviceptr,
    d_input: cuda_sys::CUdeviceptr,
    d_weight: cuda_sys::CUdeviceptr,
    rows: u32,
) {
    let eps: f32 = 1e-6;
    let mut p_out = d_output;
    let mut p_in = d_input;
    let mut p_wt = d_weight;
    let mut p_eps = eps;

    let mut args: Vec<*mut c_void> = vec![
        &mut p_out as *mut _ as *mut c_void,
        &mut p_in as *mut _ as *mut c_void,
        &mut p_wt as *mut _ as *mut c_void,
        &mut p_eps as *mut _ as *mut c_void,
    ];

    unsafe {
        cuda::launch_kernel(
            f,
            (rows, 1, 1),
            (128, 1, 1),
            512,
            cuda::stream::null(),
            &mut args,
        )
        .unwrap();
    }
}

/// Launch a standalone GEMM kernel.
fn launch_standalone_gemm(
    f: cuda_sys::CUfunction,
    d_a: cuda_sys::CUdeviceptr,
    d_b: cuda_sys::CUdeviceptr,
    d_c: cuda_sys::CUdeviceptr,
    m: u32,
    n: u32,
    k: u32,
    smem: u32,
) {
    let mut p_a = d_a;
    let mut p_b = d_b;
    let mut p_c = d_c;
    let mut p_m = m;
    let mut p_n = n;
    let mut p_k = k;

    let mut args: Vec<*mut c_void> = vec![
        &mut p_a as *mut _ as *mut c_void,
        &mut p_b as *mut _ as *mut c_void,
        &mut p_c as *mut _ as *mut c_void,
        &mut p_m as *mut _ as *mut c_void,
        &mut p_n as *mut _ as *mut c_void,
        &mut p_k as *mut _ as *mut c_void,
    ];

    unsafe {
        cuda::launch_kernel(
            f,
            ((n + 127) / 128, (m + 127) / 128, 1),
            (128, 1, 1),
            smem,
            cuda::stream::null(),
            &mut args,
        )
        .unwrap();
    }
}

/// Launch a standalone SiLU kernel (operates on f32 data in-place).
fn launch_standalone_silu(f: cuda_sys::CUfunction, d_data: cuda_sys::CUdeviceptr, n: u32) {
    let mut p_data = d_data;
    let mut p_n = n;

    let mut args: Vec<*mut c_void> = vec![
        &mut p_data as *mut _ as *mut c_void,
        &mut p_n as *mut _ as *mut c_void,
    ];

    let block_size = 256u32;
    let elems_per_thread = 4u32;
    let threads_needed = (n + elems_per_thread - 1) / elems_per_thread;
    let grid = (threads_needed + block_size - 1) / block_size;

    unsafe {
        cuda::launch_kernel(
            f,
            (grid, 1, 1),
            (block_size, 1, 1),
            0,
            cuda::stream::null(),
            &mut args,
        )
        .unwrap();
    }
}

fn load_kernel(ptx: &str, name: &str) -> cuda_sys::CUfunction {
    let ptx_cstr = CString::new(ptx).unwrap();
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _).unwrap() };
    let name_cstr = CString::new(name).unwrap();
    unsafe { cuda::module::get_function(module, name_cstr).unwrap() }
}

fn bench_mlp_fused_vs_unfused(
    batch: u32,
    hidden: u32,
    inter: u32,
    out: u32,
    warmup: u32,
    iters: u32,
) {
    println!(
        "\n  Benchmark: batch={}, hidden={}, inter={}, out={}",
        batch, hidden, inter, out
    );

    let stream = cuda::stream::null();

    // Generate standalone kernel PTXes for the unfused baseline
    #[path = "gemm_128x128.rs"]
    mod gemm_128x128;
    let gemm_ptx = gemm_128x128::emit_ptx_128x128();
    let rmsnorm_ptx = ferrite_ptx::rmsnorm::build_rmsnorm_kernel(128, hidden);
    let silu_ptx = ferrite_ptx::silu::build_silu_kernel(256, 4);
    let config = ferrite_ptx::config::GemmConfig::default_128x128();
    let smem = config.smem_total();

    // Pre-load kernel functions (don't count JIT time)
    let rmsnorm_f = load_kernel(&rmsnorm_ptx, "rmsnorm_kernel");
    let gemm_f = load_kernel(&gemm_ptx, "gemm_128x128");
    let silu_f = load_kernel(&silu_ptx, "silu_kernel");

    // Allocate device memory
    let input_elems = (batch as usize) * (hidden as usize);
    let wnorm_elems = hidden as usize;
    let wgate_elems = (hidden as usize) * (inter as usize);
    let wdown_elems = (inter as usize) * (out as usize);
    let inter_elems = (batch as usize) * (inter as usize);
    let output_elems = (batch as usize) * (out as usize);

    let d_input = unsafe { cuda::malloc_async(stream, input_elems * 2).unwrap() };
    let d_wnorm = unsafe { cuda::malloc_async(stream, wnorm_elems * 2).unwrap() };
    let d_wgate = unsafe { cuda::malloc_async(stream, wgate_elems * 2).unwrap() };
    let d_wdown = unsafe { cuda::malloc_async(stream, wdown_elems * 2).unwrap() };
    let d_norm_out = unsafe { cuda::malloc_async(stream, input_elems * 2).unwrap() };
    // Unfused intermediate: GEMM1 output (f32), then SiLU in-place
    let d_gemm1_out = unsafe { cuda::malloc_async(stream, inter_elems * 4).unwrap() };
    let d_output_fused = unsafe { cuda::malloc_async(stream, output_elems * 4).unwrap() };
    let d_output_unfused = unsafe { cuda::malloc_async(stream, output_elems * 4).unwrap() };

    // Initialize with dummy data
    unsafe {
        cuda::memset_d8_async(d_input, 0x3C, input_elems * 2, stream).ok();
        cuda::memset_d8_async(d_wnorm, 0x3C, wnorm_elems * 2, stream).ok();
        cuda::memset_d8_async(d_wgate, 0x3C, wgate_elems * 2, stream).ok();
        cuda::memset_d8_async(d_wdown, 0x3C, wdown_elems * 2, stream).ok();
        cuda::stream::synchronize(stream).unwrap();
    }

    // ── Warmup + benchmark: FUSED (proc macro, 3 launches) ──
    println!("  Benchmarking fused MLP (3 launches via proc macro)...");
    for _ in 0..warmup {
        mlp_fwd(
            d_input,
            d_wnorm,
            d_wgate,
            d_wdown,
            d_output_fused,
            batch,
            hidden,
            inter,
            out,
        );
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };

    let t0 = Instant::now();
    for _ in 0..iters {
        mlp_fwd(
            d_input,
            d_wnorm,
            d_wgate,
            d_wdown,
            d_output_fused,
            batch,
            hidden,
            inter,
            out,
        );
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };
    let fused_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // ── Warmup + benchmark: UNFUSED (4 separate launches) ──
    println!("  Benchmarking unfused MLP (4 separate launches)...");
    for _ in 0..warmup {
        // 1. RMSNorm
        launch_standalone_rmsnorm(rmsnorm_f, d_norm_out, d_input, d_wnorm, batch);
        // 2. GEMM1: norm_out [batch, hidden] x w_gate [hidden, inter] -> gemm1_out [batch, inter]
        launch_standalone_gemm(
            gemm_f,
            d_norm_out,
            d_wgate,
            d_gemm1_out,
            batch,
            inter,
            hidden,
            smem,
        );
        // 3. SiLU (in-place on f32 gemm1 output)
        launch_standalone_silu(silu_f, d_gemm1_out, batch * inter);
        // 4. GEMM2: gemm1_out [batch, inter] x w_down [inter, out] -> output [batch, out]
        // Note: unfused path skips f32->f16 conversion, operating on f32 throughout.
        // This is actually an advantage for unfused (no conversion overhead).
        // For a fair comparison, the unfused path also needs the conversion,
        // but in practice PyTorch would do the same 4-op sequence in f16/f32 mixed.
        launch_standalone_gemm(
            gemm_f,
            d_gemm1_out,
            d_wdown,
            d_output_unfused,
            batch,
            out,
            inter,
            smem,
        );
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };

    let t0 = Instant::now();
    for _ in 0..iters {
        launch_standalone_rmsnorm(rmsnorm_f, d_norm_out, d_input, d_wnorm, batch);
        launch_standalone_gemm(
            gemm_f,
            d_norm_out,
            d_wgate,
            d_gemm1_out,
            batch,
            inter,
            hidden,
            smem,
        );
        launch_standalone_silu(silu_f, d_gemm1_out, batch * inter);
        launch_standalone_gemm(
            gemm_f,
            d_gemm1_out,
            d_wdown,
            d_output_unfused,
            batch,
            out,
            inter,
            smem,
        );
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };
    let unfused_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // ── Report ──
    let speedup = unfused_us / fused_us;
    println!("\n  Results:");
    println!("    Fused MLP (3 launches):   {:.1} us", fused_us);
    println!("    Unfused MLP (4 launches): {:.1} us", unfused_us);
    println!("    Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!(
            "    PASS: Fused MLP is {:.1}% faster",
            (speedup - 1.0) * 100.0
        );
    } else {
        println!(
            "    WARN: Fused MLP is SLOWER by {:.1}%",
            (1.0 - speedup) * 100.0
        );
        println!("    This may be due to the f32->f16 conversion overhead.");
    }

    // Cleanup
    unsafe {
        cuda::free_async(d_input, stream).ok();
        cuda::free_async(d_wnorm, stream).ok();
        cuda::free_async(d_wgate, stream).ok();
        cuda::free_async(d_wdown, stream).ok();
        cuda::free_async(d_norm_out, stream).ok();
        cuda::free_async(d_gemm1_out, stream).ok();
        cuda::free_async(d_output_fused, stream).ok();
        cuda::free_async(d_output_unfused, stream).ok();
    }
}

fn main() {
    println!("Ferrite MLP Block: Fused vs. Unfused Benchmark");
    println!("===============================================");

    // Initialize CUDA
    cuda::init().expect("CUDA init failed");
    let device = cuda::device::get(0).expect("No CUDA device");
    let ctx = unsafe { cuda::primary_ctx::retain(device).expect("Cannot create context") };
    unsafe { cuda::ctx::set_current(ctx).unwrap() };

    let (major, minor) = unsafe {
        let maj = cuda::device::get_attribute(
            device,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap_or(0);
        let min = cuda::device::get_attribute(
            device,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap_or(0);
        (maj, min)
    };
    println!("GPU: compute capability {}.{}", major, minor);

    let warmup = 10;
    let iters = 100;

    // LLaMA-7B MLP dimensions: hidden=4096, intermediate=4096 (typically 11008 but 4096 for test)
    bench_mlp_fused_vs_unfused(1024, 4096, 4096, 4096, warmup, iters);

    // Smaller batch
    bench_mlp_fused_vs_unfused(256, 4096, 4096, 4096, warmup, iters);

    println!("\n===============================================");
    println!("Benchmark complete.");
}
