//! Benchmark: Fused megakernel vs. unfused (separate launches).
//!
//! Compares:
//!   1. Fused: ONE kernel doing RMSNorm -> GEMM -> SiLU (single launch)
//!   2. Unfused: 3 separate kernel launches (RMSNorm, GEMM, SiLU)
//!
//! The fused kernel MUST be faster. If it's not, the NormalizedLoader
//! or fusion strategy needs fixing.
//!
//! Usage:
//!   cargo run --release --bin ferrite-fuse-bench

use cudarc::driver::result as cuda;
use cudarc::driver::sys as cuda_sys;
use std::ffi::{CString, c_void};
use std::time::Instant;

type DevicePtr = u64;

// The fused megakernel via proc macro
#[ferrite_macros::fuse(arch = "sm_89")]
fn mlp_fwd_fused(
    input: DevicePtr,
    weight_norm: DevicePtr,
    weight_gemm: DevicePtr,
    output: DevicePtr,
    batch: u32,
    hidden: u32,
    out_feat: u32,
) {
    let n = rmsnorm(input, weight_norm);
    let g = gemm(n, weight_gemm);
    silu(g)
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

/// Launch a standalone SiLU kernel.
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

fn bench_fused_vs_unfused(batch: u32, hidden: u32, out_feat: u32, warmup: u32, iters: u32) {
    println!(
        "\n  Benchmark: batch={}, hidden={}, out_feat={}",
        batch, hidden, out_feat
    );

    let stream = cuda::stream::null();

    // Generate the standalone kernel PTXes (128×128 GEMM for fair comparison)
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
    let wgemm_elems = (hidden as usize) * (out_feat as usize);
    let output_elems = (batch as usize) * (out_feat as usize);

    let d_input = unsafe { cuda::malloc_async(stream, input_elems * 2).unwrap() };
    let d_wnorm = unsafe { cuda::malloc_async(stream, wnorm_elems * 2).unwrap() };
    let d_wgemm = unsafe { cuda::malloc_async(stream, wgemm_elems * 2).unwrap() };
    let d_norm_out = unsafe { cuda::malloc_async(stream, input_elems * 2).unwrap() };
    let d_output_fused = unsafe { cuda::malloc_async(stream, output_elems * 4).unwrap() };
    let d_output_unfused = unsafe { cuda::malloc_async(stream, output_elems * 4).unwrap() };

    // Initialize with dummy data
    unsafe {
        cuda::memset_d8_async(d_input, 0x3C, input_elems * 2, stream).ok();
        cuda::memset_d8_async(d_wnorm, 0x3C, wnorm_elems * 2, stream).ok();
        cuda::memset_d8_async(d_wgemm, 0x3C, wgemm_elems * 2, stream).ok();
        cuda::stream::synchronize(stream).unwrap();
    }

    // ── Warmup + benchmark: FUSED ──
    println!("  Benchmarking fused kernel (ONE launch)...");
    for _ in 0..warmup {
        mlp_fwd_fused(
            d_input,
            d_wnorm,
            d_wgemm,
            d_output_fused,
            batch,
            hidden,
            out_feat,
        );
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };

    let t0 = Instant::now();
    for _ in 0..iters {
        mlp_fwd_fused(
            d_input,
            d_wnorm,
            d_wgemm,
            d_output_fused,
            batch,
            hidden,
            out_feat,
        );
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };
    let fused_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // ── Warmup + benchmark: UNFUSED (3 separate launches) ──
    println!("  Benchmarking unfused (3 separate launches)...");
    for _ in 0..warmup {
        launch_standalone_rmsnorm(rmsnorm_f, d_norm_out, d_input, d_wnorm, batch);
        launch_standalone_gemm(
            gemm_f,
            d_norm_out,
            d_wgemm,
            d_output_unfused,
            batch,
            out_feat,
            hidden,
            smem,
        );
        launch_standalone_silu(silu_f, d_output_unfused, batch * out_feat);
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };

    let t0 = Instant::now();
    for _ in 0..iters {
        launch_standalone_rmsnorm(rmsnorm_f, d_norm_out, d_input, d_wnorm, batch);
        launch_standalone_gemm(
            gemm_f,
            d_norm_out,
            d_wgemm,
            d_output_unfused,
            batch,
            out_feat,
            hidden,
            smem,
        );
        launch_standalone_silu(silu_f, d_output_unfused, batch * out_feat);
    }
    unsafe { cuda::stream::synchronize(stream).unwrap() };
    let unfused_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // ── Report ──
    let speedup = unfused_us / fused_us;
    println!("\n  Results:");
    println!("    Fused (1 launch):     {:.1} us", fused_us);
    println!("    Unfused (3 launches): {:.1} us", unfused_us);
    println!("    Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!(
            "    PASS: Fused kernel is {:.1}% faster",
            (speedup - 1.0) * 100.0
        );
    } else {
        println!(
            "    FAIL: Fused kernel is SLOWER by {:.1}%",
            (1.0 - speedup) * 100.0
        );
        println!("    The NormalizedLoader needs further optimization.");
    }

    // Cleanup
    unsafe {
        cuda::free_async(d_input, stream).ok();
        cuda::free_async(d_wnorm, stream).ok();
        cuda::free_async(d_wgemm, stream).ok();
        cuda::free_async(d_norm_out, stream).ok();
        cuda::free_async(d_output_fused, stream).ok();
        cuda::free_async(d_output_unfused, stream).ok();
    }
}

fn main() {
    println!("Ferrite Fused vs. Unfused Benchmark");
    println!("===================================");

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

    // LLaMA dimensions: hidden=4096, out=4096
    bench_fused_vs_unfused(256, 4096, 4096, warmup, iters);
    bench_fused_vs_unfused(1024, 4096, 4096, warmup, iters);

    println!("\n===================================");
    println!("Benchmark complete.");
}
