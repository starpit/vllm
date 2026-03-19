// Ferrite Phase 1: rust-cuda backend proof of concept.
//
// Compiles a GEMM kernel written in Rust through libnvvm (NVIDIA's optimizer),
// then benchmarks it. This tests whether libnvvm gives us the register allocation
// quality that upstream LLVM couldn't achieve.
//
// Build & run:
//   cargo run -p ferrite-rc --release

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::result as cuda;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ferrite_kernels.ptx"));

fn main() -> Result<()> {
    println!("Ferrite Phase 1 — rust-cuda Backend");
    println!("════════════════════════════════════\n");

    cuda::init()?;
    let device = cuda::device::get(0)?;
    let ctx = unsafe { cuda::primary_ctx::retain(device)? };
    unsafe { cuda::ctx::set_current(ctx)?; }

    let name = cuda::device::get_name(device)?;
    let major = unsafe {
        cuda::device::get_attribute(device,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
    };
    let minor = unsafe {
        cuda::device::get_attribute(device,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?
    };
    println!("[cuda] GPU: {} (sm_{}{})", name, major, minor);
    println!("[ptx]  Loaded {} bytes of PTX from rust-cuda/libnvvm\n", PTX.len());

    let ptx_cstr = CString::new(PTX).context("PTX null byte")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("mma_gemm").unwrap())?
    };

    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;
    let size_a = (m * k) as usize;
    let size_b = (k * n) as usize;
    let size_c = (m * n) as usize;

    let d_a = unsafe { cuda::malloc_sync(size_a * 2)? };
    let d_b = unsafe { cuda::malloc_sync(size_b * 2)? };
    let d_c = unsafe { cuda::malloc_sync(size_c * 4)? };

    let h_a: Vec<half::f16> = (0..size_a)
        .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
        .collect();
    let h_b: Vec<half::f16> = (0..size_b)
        .map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
        .collect();
    let mut h_c: Vec<f32> = vec![0.0; size_c];

    unsafe {
        cuda::memcpy_htod_sync(d_a, &h_a)?;
        cuda::memcpy_htod_sync(d_b, &h_b)?;
    }

    let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

    let bm: c_uint = 64;
    let bn: c_uint = 64;
    let grid_x: c_uint = n / bn;
    let grid_y: c_uint = m / bm;
    let threads: c_uint = 128; // 4 warps

    let params: &mut [*mut c_void] = &mut [
        (&d_a) as *const _ as *mut c_void,
        (&d_b) as *const _ as *mut c_void,
        (&d_c) as *const _ as *mut c_void,
        (&m) as *const _ as *mut c_void,
        (&n) as *const _ as *mut c_void,
        (&k) as *const _ as *mut c_void,
    ];

    // Warmup
    for _ in 0..5 {
        unsafe {
            cuda::launch_kernel(
                func, (grid_x, grid_y, 1), (threads, 1, 1), 0, stream, params,
            )?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }

    // Benchmark
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            cuda::launch_kernel(
                func, (grid_x, grid_y, 1), (threads, 1, 1), 0, stream, params,
            )?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }
    let elapsed = start.elapsed();
    let us_per = elapsed.as_micros() as f64 / iters as f64;

    // Verify
    unsafe { cuda::memcpy_dtoh_sync(&mut h_c, d_c)?; }

    let mut max_err: f32 = 0.0;
    for row in 0..m as usize {
        for col in 0..n as usize {
            let mut expected: f32 = 0.0;
            for kk in 0..k as usize {
                expected += h_a[row * k as usize + kk].to_f32()
                    * h_b[kk * n as usize + col].to_f32();
            }
            let err = (h_c[row * n as usize + col] - expected).abs();
            if err > max_err { max_err = err; }
        }
    }

    if max_err < 1.0 {
        println!("✓ Correct! (max error: {:.4})", max_err);
    } else {
        println!("✗ Max error: {:.4}", max_err);
        bail!("Verification failed");
    }

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / (us_per * 1e-6) / 1e12;
    let pct = tflops / 181.0 * 100.0;
    println!("{:.1} μs, {:.2} TFLOPS ({:.1}% of L40S peak)", us_per, tflops, pct);

    unsafe {
        cuda::stream::destroy(stream)?;
        cuda::free_sync(d_a)?;
        cuda::free_sync(d_b)?;
        cuda::free_sync(d_c)?;
        cuda::module::unload(module)?;
    }

    Ok(())
}
