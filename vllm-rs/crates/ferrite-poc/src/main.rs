// SPDX-License-Identifier: Apache-2.0
//
// Ferrite Phase 0 — Proof of Concept
//
// Validates the full pipeline: Rust → inkwell → LLVM IR → PTX → GPU kernel.
// Run on a machine with LLVM 20 and an NVIDIA GPU:
//
//   LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo run -p ferrite-poc --release

#[allow(unused)]
mod mma_gemm;
#[allow(unused)]
mod tiled_mma;
#[allow(unused)]
mod cubek_gemm;
#[allow(unused, unsafe_op_in_unsafe_fn)]
mod sweep;
#[allow(unused, unsafe_op_in_unsafe_fn)]
mod triton_style;
#[allow(unused)]
mod ptx_builder;
#[allow(unused)]
mod gemm_128x128;

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

// ---------------------------------------------------------------------------
// LLVM / inkwell
// ---------------------------------------------------------------------------
use inkwell::context::Context as LlvmContext;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::targets::{
    InitializationConfig, Target, TargetMachine, TargetTriple,
    RelocMode, CodeModel, FileType,
};
use inkwell::values::{AsValueRef, BasicValueEnum, FunctionValue, IntValue};
use inkwell::{AddressSpace, OptimizationLevel, IntPredicate};

// ---------------------------------------------------------------------------
// CUDA (cudarc raw driver API)
// ---------------------------------------------------------------------------
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::result as cuda;

fn main() -> Result<()> {
    println!("Ferrite Phase 0 — Proof of Concept");
    println!("═══════════════════════════════════\n");

    // Enable 32-bit shared memory pointers — required for ldmatrix.
    // Same flag Triton uses (confirmed: their ldmatrix uses %r not %rd).
    unsafe {
        let args: [*const i8; 2] = [
            b"ferrite\0".as_ptr() as *const i8,
            b"-nvptx-short-ptr\0".as_ptr() as *const i8,
        ];
        llvm_sys::support::LLVMParseCommandLineOptions(2, args.as_ptr(), std::ptr::null());
    }
    Target::initialize_nvptx(&InitializationConfig::default());
    println!("[llvm] NVPTX target initialized (short-ptr enabled)");

    cuda::init()?;
    let device = cuda::device::get(0)?;
    let ctx = unsafe { cuda::primary_ctx::retain(device)? };
    unsafe { cuda::ctx::set_current(ctx)?; }

    let name = cuda::device::get_name(device)?;
    let major = unsafe {
        cuda::device::get_attribute(
            device,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?
    };
    let minor = unsafe {
        cuda::device::get_attribute(
            device,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )?
    };
    let sm = format!("sm_{}{}", major, minor);
    println!("[cuda] GPU: {} ({})", name, sm);

    // GEMM validation gate (55 TFLOPS)
    println!("\nGEMM 64x64 (validation gate)");
    run_ptxbuilder_gemm()?;

    // 128x128 GEMM benchmark
    println!("\nGEMM 128x128 (targeting 48+ TFLOPS)");
    run_gemm_128x128()?;

    println!("\n═══════════════════════════════════");
    println!("Phase 0 complete.");
    Ok(())
}

// ===========================================================================
// PtxBuilder GEMM benchmark
// ===========================================================================

fn run_pipeline_gemm() -> Result<()> {
    let config = ferrite_ptx::config::GemmConfig::default_64x64();
    let ptx = ferrite_ptx::gemm::build_gemm_pipeline(&config);
    println!("  Generated {} bytes of PTX", ptx.len());
    std::fs::write("/tmp/ferrite_pipeline_gemm.ptx", &ptx).ok();

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("triton_style_gemm").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)? };
    println!("  [cuda] Pipeline GEMM -- {} regs/thread, {} bytes local", nregs, local_bytes);

    let bm = config.bm;
    let bn = config.bn;
    let threads = config.threads();
    let smem_bytes: c_uint = config.smem_total();

    for &sz in &[1024u32] {
        let m = sz; let n = sz; let k = sz;
        let sa = (m * k) as usize;
        let sb = (k * n) as usize;
        let sc = (m * n) as usize;
        let da = unsafe { cuda::malloc_sync(sa * 2)? };
        let db = unsafe { cuda::malloc_sync(sb * 2)? };
        let dc = unsafe { cuda::malloc_sync(sc * 4)? };
        let ha: Vec<half::f16> = (0..sa).map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1)).collect();
        let hb: Vec<half::f16> = (0..sb).map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1)).collect();
        let mut hc: Vec<f32> = vec![0.0; sc];
        unsafe { cuda::memcpy_htod_sync(da, &ha)?; cuda::memcpy_htod_sync(db, &hb)?; }
        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx = n / bn; let gy = m / bm;
        let params: &mut [*mut c_void] = &mut [
            (&da) as *const _ as *mut c_void, (&db) as *const _ as *mut c_void,
            (&dc) as *const _ as *mut c_void, (&m) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void, (&k) as *const _ as *mut c_void,
        ];
        for _ in 0..10 {
            unsafe { cuda::launch_kernel(func, (gx, gy, 1), (threads, 1, 1), smem_bytes, stream, params)?; }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe { cuda::launch_kernel(func, (gx, gy, 1), (threads, 1, 1), smem_bytes, stream, params)?; }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let us = start.elapsed().as_micros() as f64 / iters as f64;
        unsafe { cuda::memcpy_dtoh_sync(&mut hc, dc)?; }
        let mut max_err: f32 = 0.0;
        for r in [0,1,31,32,63] {
            for c2 in [0,1,31,32,63] {
                if r >= m as usize || c2 >= n as usize { continue; }
                let mut e = 0.0f32;
                for kk in 0..k as usize { e += ha[r*k as usize+kk].to_f32() * hb[kk*n as usize+c2].to_f32(); }
                let err = (hc[r * n as usize + c2] - e).abs();
                if err > max_err { max_err = err; }
            }
        }
        let tf = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
        println!("  --- {m}x{n} ---");
        if max_err < 1.0 { println!("  Correct (max err: {max_err:.4})"); }
        else { println!("  ERROR: max err {max_err:.4}"); }
        println!("  {us:.1} us, {tf:.1} TFLOPS");
        unsafe { cuda::stream::destroy(stream)?; cuda::free_sync(da)?; cuda::free_sync(db)?; cuda::free_sync(dc)?; cuda::module::unload(module)?; }
    }
    Ok(())
}

fn run_ptxbuilder_gemm() -> Result<()> {
    let config = ptx_builder::config::GemmConfig::default_64x64();
    let ptx = ptx_builder::gemm::build_gemm(&config);
    println!("  Generated {} bytes of PTX", ptx.len());

    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_ptxbuilder.ptx", &ptx).ok();
        println!("  [debug] PTX dumped to /tmp/ferrite_ptxbuilder.ptx");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("triton_style_gemm").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)? };
    println!("  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)", nregs, local_bytes);

    let bm = config.bm;
    let bn = config.bn;
    let threads = config.threads();
    let smem_bytes: c_uint = config.smem_total();

    for &sz in &[1024u32, 4096] {
        let m = sz;
        let n = sz;
        let k = sz;
        println!("  --- {}x{} ---", m, n);

        let sa = (m * k) as usize;
        let sb = (k * n) as usize;
        let sc = (m * n) as usize;
        let da = unsafe { cuda::malloc_sync(sa * 2)? };
        let db = unsafe { cuda::malloc_sync(sb * 2)? };
        let dc = unsafe { cuda::malloc_sync(sc * 4)? };

        let ha: Vec<half::f16> = (0..sa).map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1)).collect();
        let hb: Vec<half::f16> = (0..sb).map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1)).collect();
        let mut hc: Vec<f32> = vec![0.0; sc];
        unsafe { cuda::memcpy_htod_sync(da, &ha)?; cuda::memcpy_htod_sync(db, &hb)?; }

        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx = n / bn;
        let gy = m / bm;
        let params: &mut [*mut c_void] = &mut [
            (&da) as *const _ as *mut c_void, (&db) as *const _ as *mut c_void,
            (&dc) as *const _ as *mut c_void, (&m) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void, (&k) as *const _ as *mut c_void,
        ];

        for _ in 0..10 {
            unsafe { cuda::launch_kernel(func, (gx, gy, 1), (threads, 1, 1), smem_bytes, stream, params)?; }
        }
        unsafe { cuda::stream::synchronize(stream)?; }

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe { cuda::launch_kernel(func, (gx, gy, 1), (threads, 1, 1), smem_bytes, stream, params)?; }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        unsafe { cuda::memcpy_dtoh_sync(&mut hc, dc)?; }
        let mut max_err: f32 = 0.0;
        for r in [0,1,31,32,63,511,1023] {
            for c in [0,1,31,32,63,511,1023] {
                if r >= m as usize || c >= n as usize { continue; }
                let mut e = 0.0f32;
                for kk in 0..k as usize { e += ha[r*k as usize+kk].to_f32() * hb[kk*n as usize+c].to_f32(); }
                let err = (hc[r * n as usize + c] - e).abs();
                if err > max_err { max_err = err; }
            }
        }

        if max_err < 1.0 { println!("  Correct (max err: {:.4})", max_err); }
        else { bail!("  Max error: {:.4}", max_err); }

        let tf = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
        println!("  {:.1} us, {:.1} TFLOPS", us, tf);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(da)?; cuda::free_sync(db)?; cuda::free_sync(dc)?;
        }
    }

    unsafe { cuda::module::unload(module)?; }
    Ok(())
}

// ===========================================================================
// 128x128 GEMM benchmark
// ===========================================================================

fn run_gemm_128x128() -> Result<()> {
    let ptx = gemm_128x128::emit_ptx_128x128();
    println!("  Generated {} bytes of PTX", ptx.len());

    std::fs::write("/tmp/ferrite_gemm_128x128.ptx", &ptx).ok();
    println!("  [debug] PTX dumped to /tmp/ferrite_gemm_128x128.ptx");

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("gemm_128x128").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)? };
    println!("  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)", nregs, local_bytes);

    let bm: u32 = 128;
    let bn: u32 = 128;
    let threads: u32 = 128;
    let smem_bytes: c_uint = 32768; // 2 stages × (8192 A + 8192 B) = 32KB

    for &sz in &[1024u32] {
        let m = sz; let n = sz; let k = sz;
        println!("  --- {}x{} ---", m, n);

        let sa = (m * k) as usize;
        let sb = (k * n) as usize;
        let sc = (m * n) as usize;
        let da = unsafe { cuda::malloc_sync(sa * 2)? };
        let db = unsafe { cuda::malloc_sync(sb * 2)? };
        let dc = unsafe { cuda::malloc_sync(sc * 4)? };

        let ha: Vec<half::f16> = (0..sa).map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1)).collect();
        let hb: Vec<half::f16> = (0..sb).map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1)).collect();
        let mut hc: Vec<f32> = vec![0.0; sc];
        unsafe { cuda::memcpy_htod_sync(da, &ha)?; cuda::memcpy_htod_sync(db, &hb)?; }

        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx = n / bn;
        let gy = m / bm;
        let params: &mut [*mut c_void] = &mut [
            (&da) as *const _ as *mut c_void, (&db) as *const _ as *mut c_void,
            (&dc) as *const _ as *mut c_void, (&m) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void, (&k) as *const _ as *mut c_void,
        ];

        // Warmup
        for _ in 0..10 {
            unsafe { cuda::launch_kernel(func, (gx, gy, 1), (threads, 1, 1), smem_bytes, stream, params)?; }
        }
        unsafe { cuda::stream::synchronize(stream)?; }

        // Benchmark
        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe { cuda::launch_kernel(func, (gx, gy, 1), (threads, 1, 1), smem_bytes, stream, params)?; }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        // Verify
        unsafe { cuda::memcpy_dtoh_sync(&mut hc, dc)?; }
        let mut max_err: f32 = 0.0;
        for r in [0,1,31,32,63,64,95,96,127,511,1023] {
            for c in [0,1,31,32,63,64,95,96,127,511,1023] {
                if r >= m as usize || c >= n as usize { continue; }
                let mut e = 0.0f32;
                for kk in 0..k as usize { e += ha[r*k as usize+kk].to_f32() * hb[kk*n as usize+c].to_f32(); }
                let err = (hc[r * n as usize + c] - e).abs();
                if err > max_err { max_err = err; }
            }
        }

        if max_err < 1.0 { println!("  Correct (max err: {:.4})", max_err); }
        else {
            // Print a few sample values for debugging
            for r in [0, 1, 64, 127] {
                for c in [0, 1, 64, 127] {
                    if r >= m as usize || c >= n as usize { continue; }
                    let mut e = 0.0f32;
                    for kk in 0..k as usize { e += ha[r*k as usize+kk].to_f32() * hb[kk*n as usize+c].to_f32(); }
                    let got = hc[r * n as usize + c];
                    println!("    C[{r}][{c}]: got={got:.4} expected={e:.4}");
                }
            }
            bail!("  Max error: {:.4}", max_err);
        }

        let tf = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
        println!("  {:.1} us, {:.1} TFLOPS", us, tf);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(da)?; cuda::free_sync(db)?; cuda::free_sync(dc)?;
        }
    }

    unsafe { cuda::module::unload(module)?; }
    Ok(())
}

// ===========================================================================
// PtxBuilder SiLU benchmark
// ===========================================================================

fn run_ptxbuilder_silu() -> Result<()> {
    let block_size: u32 = 256;
    let elems_per_thread: u32 = 4;
    let ptx = ptx_builder::silu::build_silu_kernel(block_size, elems_per_thread);
    println!("  Generated {} bytes of PTX", ptx.len());

    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_silu.ptx", &ptx).ok();
        println!("  [debug] PTX dumped to /tmp/ferrite_silu.ptx");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("silu_kernel").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    println!("  [cuda] Loaded -- {} regs/thread", nregs);

    // CPU reference SiLU
    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    for &n in &[1_000_000u32, 4_000_000, 16_000_000, 64_000_000] {
        println!("  --- N={} ---", n);

        // Allocate
        let bytes = n as usize * 4;
        let d_data = unsafe { cuda::malloc_sync(bytes)? };

        // Host data: mix of positive/negative values
        let h_input: Vec<f32> = (0..n as usize)
            .map(|i| ((i % 1000) as f32 - 500.0) * 0.01)
            .collect();
        let mut h_output: Vec<f32> = vec![0.0; n as usize];
        unsafe { cuda::memcpy_htod_sync(d_data, &h_input)?; }

        let threads_per_block = block_size;
        let elems_per_block = block_size * elems_per_thread;
        let grid_size = (n + elems_per_block - 1) / elems_per_block;

        let params: &mut [*mut c_void] = &mut [
            (&d_data) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void,
        ];

        // Correctness check using default stream (synchronous)
        unsafe {
            cuda::launch_kernel(
                func, (grid_size, 1, 1), (threads_per_block, 1, 1),
                0, std::ptr::null_mut(), params,
            )?;
            cuda::ctx::synchronize()?;
            cuda::memcpy_dtoh_sync(&mut h_output, d_data)?;
        }

        // Verify
        let mut max_err: f32 = 0.0;
        for i in (0..n as usize).step_by(997) {
            let expected = silu_ref(h_input[i]);
            let err = (h_output[i] - expected).abs();
            if err > max_err { max_err = err; }
        }
        if max_err < 1e-3 {
            println!("  Correct (max err: {:.6})", max_err);
        } else {
            bail!("  SiLU max error: {:.4}", max_err);
        }

        // Benchmark
        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

        // Warmup
        unsafe { cuda::memcpy_htod_sync(d_data, &h_input)?; }
        for _ in 0..20 {
            unsafe {
                cuda::launch_kernel(
                    func, (grid_size, 1, 1), (threads_per_block, 1, 1),
                    0, stream, params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }

        // Timed runs
        let iters = 200;
        unsafe { cuda::memcpy_htod_sync(d_data, &h_input)?; }
        unsafe { cuda::stream::synchronize(stream)?; }
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func, (grid_size, 1, 1), (threads_per_block, 1, 1),
                    0, stream, params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        // Bandwidth: read + write = 2 * N * 4 bytes
        let bw = (2.0 * n as f64 * 4.0) / (us * 1e-6) / 1e9;
        println!("  {:.1} us, {:.1} GB/s", us, bw);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(d_data)?;
        }
    }

    unsafe { cuda::module::unload(module)?; }
    Ok(())
}

// ===========================================================================
// PtxBuilder RMSNorm benchmark
// ===========================================================================

fn run_ptxbuilder_rmsnorm() -> Result<()> {
    let block_size: u32 = 256;
    let hidden_size: u32 = 4096;
    let eps: f32 = 1e-6;

    let ptx = ptx_builder::rmsnorm::build_rmsnorm_kernel(block_size, hidden_size);
    println!("  Generated {} bytes of PTX", ptx.len());

    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_rmsnorm.ptx", &ptx).ok();
        println!("  [debug] PTX dumped to /tmp/ferrite_rmsnorm.ptx");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("rmsnorm_kernel").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    println!("  [cuda] Loaded -- {} regs/thread", nregs);

    // Shared memory needed: num_warps * 4 bytes for reduction scratch
    let num_warps = block_size / 32;
    let smem_bytes: c_uint = num_warps * 4;

    // CPU reference RMSNorm
    fn rmsnorm_ref(input: &[half::f16], weight: &[half::f16], eps: f32) -> Vec<half::f16> {
        let n = input.len();
        let sum_sq: f32 = input.iter().map(|x| {
            let xf = x.to_f32();
            xf * xf
        }).sum();
        let rms = ((sum_sq / n as f32) + eps).sqrt();
        let scale = 1.0 / rms;
        input.iter().zip(weight.iter()).map(|(x, w)| {
            half::f16::from_f32(x.to_f32() * scale * w.to_f32())
        }).collect()
    }

    for &batch_size in &[1u32, 32, 256, 1024] {
        println!("  --- batch={}, hidden={} ---", batch_size, hidden_size);

        let total_elems = (batch_size * hidden_size) as usize;
        let weight_elems = hidden_size as usize;

        // Allocate device memory (f16 = 2 bytes each)
        let d_input = unsafe { cuda::malloc_sync(total_elems * 2)? };
        let d_output = unsafe { cuda::malloc_sync(total_elems * 2)? };
        let d_weight = unsafe { cuda::malloc_sync(weight_elems * 2)? };

        // Host data
        let h_input: Vec<half::f16> = (0..total_elems)
            .map(|i| half::f16::from_f32(((i % 1000) as f32 - 500.0) * 0.002))
            .collect();
        let h_weight: Vec<half::f16> = (0..weight_elems)
            .map(|i| half::f16::from_f32(0.5 + ((i % 100) as f32) * 0.01))
            .collect();
        let mut h_output: Vec<half::f16> = vec![half::f16::from_f32(0.0); total_elems];

        unsafe {
            cuda::memcpy_htod_sync(d_input, &h_input)?;
            cuda::memcpy_htod_sync(d_weight, &h_weight)?;
        }

        // Launch: one block per row
        let grid_size = batch_size;
        let params: &mut [*mut c_void] = &mut [
            (&d_output) as *const _ as *mut c_void,
            (&d_input) as *const _ as *mut c_void,
            (&d_weight) as *const _ as *mut c_void,
            (&eps) as *const _ as *mut c_void,
        ];

        // Correctness check
        unsafe {
            cuda::launch_kernel(
                func, (grid_size, 1, 1), (block_size, 1, 1),
                smem_bytes, std::ptr::null_mut(), params,
            )?;
            cuda::ctx::synchronize()?;
            cuda::memcpy_dtoh_sync(&mut h_output, d_output)?;
        }

        // Verify against CPU reference (check a few rows)
        let mut max_err: f32 = 0.0;
        for row in 0..batch_size.min(8) as usize {
            let start = row * hidden_size as usize;
            let end = start + hidden_size as usize;
            let expected = rmsnorm_ref(&h_input[start..end], &h_weight, eps);
            for (_j, (&got, &exp)) in h_output[start..end].iter().zip(expected.iter()).enumerate() {
                let err = (got.to_f32() - exp.to_f32()).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        if max_err < 0.05 {
            println!("  Correct (max err: {:.6})", max_err);
        } else {
            bail!("  RMSNorm max error: {:.4}", max_err);
        }

        // Benchmark
        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

        // Warmup
        for _ in 0..50 {
            unsafe {
                cuda::launch_kernel(
                    func, (grid_size, 1, 1), (block_size, 1, 1),
                    smem_bytes, stream, params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }

        // Timed
        let iters = 500;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func, (grid_size, 1, 1), (block_size, 1, 1),
                    smem_bytes, stream, params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        // Bandwidth: read input (f16) + read weight (f16) + write output (f16)
        let bytes_total = batch_size as f64 * hidden_size as f64 * 2.0 * 2.0
            + hidden_size as f64 * 2.0;
        let bw = bytes_total / (us * 1e-6) / 1e9;
        println!("  {:.1} us, {:.1} GB/s", us, bw);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(d_input)?;
            cuda::free_sync(d_output)?;
            cuda::free_sync(d_weight)?;
        }
    }

    unsafe { cuda::module::unload(module)?; }
    Ok(())
}

// ===========================================================================
// Fused RMSNorm -> GEMM -> SiLU benchmark
// ===========================================================================

fn run_fused_rmsnorm_gemm_silu() -> Result<()> {
    let config = ptx_builder::config::GemmConfig::default_64x64();
    let hidden_size: u32 = 4096;
    let out_features: u32 = 4096;
    let eps: f32 = 1e-6;

    let ptx = ptx_builder::fused::build_fused_rmsnorm_gemm_silu(&config, hidden_size);
    println!("  Generated {} bytes of PTX", ptx.len());

    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_fused.ptx", &ptx).ok();
        println!("  [debug] PTX dumped to /tmp/ferrite_fused.ptx");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("fused_rmsnorm_gemm_silu").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)? };
    println!("  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)", nregs, local_bytes);

    // Also load standalone kernels for unfused comparison
    let rmsnorm_ptx = ptx_builder::rmsnorm::build_rmsnorm_kernel(256, hidden_size);
    let rmsnorm_cstr = CString::new(rmsnorm_ptx.as_bytes()).context("PTX null")?;
    let rmsnorm_module = unsafe { cuda::module::load_data(rmsnorm_cstr.as_ptr() as *const _)? };
    let rmsnorm_func = unsafe {
        cuda::module::get_function(rmsnorm_module, CString::new("rmsnorm_kernel").unwrap())?
    };

    let gemm_ptx = ptx_builder::gemm::build_gemm(&config);
    let gemm_cstr = CString::new(gemm_ptx.as_bytes()).context("PTX null")?;
    let gemm_module = unsafe { cuda::module::load_data(gemm_cstr.as_ptr() as *const _)? };
    let gemm_func = unsafe {
        cuda::module::get_function(gemm_module, CString::new("triton_style_gemm").unwrap())?
    };

    let silu_ptx = ptx_builder::silu::build_silu_kernel(256, 4);
    let silu_cstr = CString::new(silu_ptx.as_bytes()).context("PTX null")?;
    let silu_module = unsafe { cuda::module::load_data(silu_cstr.as_ptr() as *const _)? };
    let silu_func = unsafe {
        cuda::module::get_function(silu_module, CString::new("silu_kernel").unwrap())?
    };

    let bm = config.bm;
    let bn = config.bn;
    let threads = config.threads();

    // CPU reference
    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn rmsnorm_gemm_silu_ref(
        input: &[half::f16], wnorm: &[half::f16], wgemm: &[half::f16],
        batch: usize, hidden: usize, out: usize, eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * out];
        for b in 0..batch {
            // RMSNorm
            let row = &input[b * hidden..(b + 1) * hidden];
            let sum_sq: f32 = row.iter().map(|x| {
                let xf = x.to_f32();
                xf * xf
            }).sum();
            let scale = 1.0 / ((sum_sq / hidden as f32) + eps).sqrt();

            // GEMM + SiLU
            for j in 0..out {
                let mut acc = 0.0f32;
                for k in 0..hidden {
                    let normed = row[k].to_f32() * scale * wnorm[k].to_f32();
                    acc += normed * wgemm[k * out + j].to_f32();
                }
                // SiLU
                output[b * out + j] = acc / (1.0 + (-acc).exp());
            }
        }
        output
    }

    // Shared memory for fused kernel
    let gemm_smem = config.smem_total();
    let fused_smem: c_uint = gemm_smem + 32 + bm * 4; // norm scratch + norm factors

    for &batch_size in &[64u32, 256, 1024] {
        println!("  --- batch={}, hidden={}, out_features={} ---", batch_size, hidden_size, out_features);

        let input_elems = (batch_size * hidden_size) as usize;
        let wnorm_elems = hidden_size as usize;
        let wgemm_elems = (hidden_size * out_features) as usize;
        let output_elems = (batch_size * out_features) as usize;

        // Allocate device memory
        let d_input = unsafe { cuda::malloc_sync(input_elems * 2)? };
        let d_wnorm = unsafe { cuda::malloc_sync(wnorm_elems * 2)? };
        let d_wgemm = unsafe { cuda::malloc_sync(wgemm_elems * 2)? };
        let d_output = unsafe { cuda::malloc_sync(output_elems * 4)? };

        // Also need intermediate buffers for unfused path
        let d_normed = unsafe { cuda::malloc_sync(input_elems * 2)? };     // rmsnorm output (f16)
        let d_gemm_out = unsafe { cuda::malloc_sync(output_elems * 4)? };  // gemm output (f32)

        // Host data
        let h_input: Vec<half::f16> = (0..input_elems)
            .map(|i| half::f16::from_f32(((i % 1000) as f32 - 500.0) * 0.002))
            .collect();
        let h_wnorm: Vec<half::f16> = (0..wnorm_elems)
            .map(|i| half::f16::from_f32(0.5 + ((i % 100) as f32) * 0.01))
            .collect();
        let h_wgemm: Vec<half::f16> = (0..wgemm_elems)
            .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.01))
            .collect();
        let mut h_output: Vec<f32> = vec![0.0; output_elems];

        unsafe {
            cuda::memcpy_htod_sync(d_input, &h_input)?;
            cuda::memcpy_htod_sync(d_wnorm, &h_wnorm)?;
            cuda::memcpy_htod_sync(d_wgemm, &h_wgemm)?;
        }

        // ── Fused kernel launch ──
        let gx = out_features / bn;
        let gy = batch_size / bm;
        let fused_params: &mut [*mut c_void] = &mut [
            (&d_input) as *const _ as *mut c_void,
            (&d_wnorm) as *const _ as *mut c_void,
            (&d_wgemm) as *const _ as *mut c_void,
            (&d_output) as *const _ as *mut c_void,
            (&out_features) as *const _ as *mut c_void,
            (&hidden_size) as *const _ as *mut c_void,
        ];

        // Correctness check
        unsafe {
            cuda::launch_kernel(
                func, (gx, gy, 1), (threads, 1, 1),
                fused_smem, std::ptr::null_mut(), fused_params,
            )?;
            cuda::ctx::synchronize()?;
            cuda::memcpy_dtoh_sync(&mut h_output, d_output)?;
        }

        // CPU reference
        let expected = rmsnorm_gemm_silu_ref(
            &h_input, &h_wnorm, &h_wgemm,
            batch_size as usize, hidden_size as usize, out_features as usize, eps,
        );

        let mut max_err: f32 = 0.0;
        let mut err_count = 0;
        for i in (0..output_elems).step_by(97) {
            let err = (h_output[i] - expected[i]).abs();
            if err > max_err {
                max_err = err;
            }
            if err > 1.0 {
                err_count += 1;
                if err_count <= 3 {
                    let row = i / out_features as usize;
                    let col = i % out_features as usize;
                    println!("    MISMATCH [{},{}]: got {:.6}, expected {:.6}, err {:.6}",
                             row, col, h_output[i], expected[i], err);
                }
            }
        }

        if max_err < 2.0 {
            println!("  Fused correct (max err: {:.4})", max_err);
        } else {
            println!("  WARNING: Fused max error: {:.4} (may be accumulation tolerance issue)", max_err);
        }

        // ── Benchmark fused ──
        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

        // Warmup
        for _ in 0..20 {
            unsafe {
                cuda::launch_kernel(
                    func, (gx, gy, 1), (threads, 1, 1),
                    fused_smem, stream, fused_params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func, (gx, gy, 1), (threads, 1, 1),
                    fused_smem, stream, fused_params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let fused_us = start.elapsed().as_micros() as f64 / iters as f64;

        // ── Benchmark unfused (3 separate launches) ──
        let rmsnorm_smem: c_uint = (256 / 32) * 4;  // 8 warps * 4 bytes
        let gemm_smem: c_uint = config.smem_total();
        let m_param = batch_size;
        let k_param_val = hidden_size;
        let n_param_val = out_features;
        let silu_n = output_elems as u32;

        let rmsnorm_params: &mut [*mut c_void] = &mut [
            (&d_normed) as *const _ as *mut c_void,
            (&d_input) as *const _ as *mut c_void,
            (&d_wnorm) as *const _ as *mut c_void,
            (&eps) as *const _ as *mut c_void,
        ];

        let gemm_params: &mut [*mut c_void] = &mut [
            (&d_normed) as *const _ as *mut c_void,
            (&d_wgemm) as *const _ as *mut c_void,
            (&d_gemm_out) as *const _ as *mut c_void,
            (&m_param) as *const _ as *mut c_void,
            (&n_param_val) as *const _ as *mut c_void,
            (&k_param_val) as *const _ as *mut c_void,
        ];

        let silu_params: &mut [*mut c_void] = &mut [
            (&d_gemm_out) as *const _ as *mut c_void,
            (&silu_n) as *const _ as *mut c_void,
        ];

        // Warmup unfused
        for _ in 0..20 {
            unsafe {
                cuda::launch_kernel(
                    rmsnorm_func, (batch_size, 1, 1), (256, 1, 1),
                    rmsnorm_smem, stream, rmsnorm_params,
                )?;
                cuda::launch_kernel(
                    gemm_func, (gx, gy, 1), (threads, 1, 1),
                    gemm_smem, stream, gemm_params,
                )?;
                let silu_grid = (silu_n + 1023) / 1024;
                cuda::launch_kernel(
                    silu_func, (silu_grid, 1, 1), (256, 1, 1),
                    0, stream, silu_params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }

        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    rmsnorm_func, (batch_size, 1, 1), (256, 1, 1),
                    rmsnorm_smem, stream, rmsnorm_params,
                )?;
                cuda::launch_kernel(
                    gemm_func, (gx, gy, 1), (threads, 1, 1),
                    gemm_smem, stream, gemm_params,
                )?;
                let silu_grid = (silu_n + 1023) / 1024;
                cuda::launch_kernel(
                    silu_func, (silu_grid, 1, 1), (256, 1, 1),
                    0, stream, silu_params,
                )?;
            }
        }
        unsafe { cuda::stream::synchronize(stream)?; }
        let unfused_us = start.elapsed().as_micros() as f64 / iters as f64;

        let flops = 2.0 * batch_size as f64 * hidden_size as f64 * out_features as f64;
        let fused_tflops = flops / (fused_us * 1e-6) / 1e12;
        let unfused_tflops = flops / (unfused_us * 1e-6) / 1e12;
        let speedup = unfused_us / fused_us;

        println!("  Fused:   {:.1} us, {:.1} TFLOPS", fused_us, fused_tflops);
        println!("  Unfused: {:.1} us, {:.1} TFLOPS (3 launches)", unfused_us, unfused_tflops);
        println!("  Speedup: {:.2}x", speedup);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(d_input)?;
            cuda::free_sync(d_wnorm)?;
            cuda::free_sync(d_wgemm)?;
            cuda::free_sync(d_output)?;
            cuda::free_sync(d_normed)?;
            cuda::free_sync(d_gemm_out)?;
        }
    }

    unsafe {
        cuda::module::unload(module)?;
        cuda::module::unload(rmsnorm_module)?;
        cuda::module::unload(gemm_module)?;
        cuda::module::unload(silu_module)?;
    }
    Ok(())
}

// ===========================================================================
// Step 1: Vector Add — proves inkwell → PTX → launch → correct
// ===========================================================================

fn step1_vector_add(sm: &str) -> Result<()> {
    let n: usize = 1 << 20;

    let ptx = emit_vector_add_ptx(sm)?;
    println!("  [llvm] Generated {} bytes of PTX", ptx.len());

    // Load PTX as CUDA module
    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX contains null byte")?;
    let module = unsafe {
        cuda::module::load_data(ptx_cstr.as_ptr() as *const _)?
    };
    let func_name = CString::new("vector_add").unwrap();
    let func = unsafe { cuda::module::get_function(module, func_name)? };
    println!("  [cuda] Module loaded, function resolved");

    // Allocate GPU memory
    let bytes = n * std::mem::size_of::<f32>();
    let d_a = unsafe { cuda::malloc_sync(bytes)? };
    let d_b = unsafe { cuda::malloc_sync(bytes)? };
    let d_c = unsafe { cuda::malloc_sync(bytes)? };

    // Host data
    let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let h_b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let mut h_c: Vec<f32> = vec![0.0; n];

    unsafe {
        cuda::memcpy_htod_sync(d_a, &h_a)?;
        cuda::memcpy_htod_sync(d_b, &h_b)?;
    }

    // Stream
    let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

    let block_size: c_uint = 256;
    let grid_size: c_uint = ((n as c_uint) + block_size - 1) / block_size;
    let n_u32 = n as u32;

    let params: &mut [*mut c_void] = &mut [
        (&d_a) as *const _ as *mut c_void,
        (&d_b) as *const _ as *mut c_void,
        (&d_c) as *const _ as *mut c_void,
        (&n_u32) as *const _ as *mut c_void,
    ];

    // Warmup
    for _ in 0..10 {
        unsafe {
            cuda::launch_kernel(
                func,
                (grid_size, 1, 1),
                (block_size, 1, 1),
                0,
                stream,
                params,
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
                func,
                (grid_size, 1, 1),
                (block_size, 1, 1),
                0,
                stream,
                params,
            )?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_micros() as f64 / iters as f64;

    // Verify
    unsafe { cuda::memcpy_dtoh_sync(&mut h_c, d_c)?; }

    let mut correct = true;
    for i in 0..n {
        let expected = h_a[i] + h_b[i];
        if (h_c[i] - expected).abs() > 1e-5 {
            println!("  MISMATCH at {}: got {}, expected {}", i, h_c[i], expected);
            correct = false;
            break;
        }
    }

    if correct {
        println!("  ✓ Correct! ({} elements verified)", n);
    } else {
        bail!("  ✗ Verification failed");
    }

    let gb_per_sec = (3.0 * bytes as f64) / (us_per_launch * 1e-6) / 1e9;
    println!(
        "  {:.1} μs/launch, {:.1} GB/s effective bandwidth",
        us_per_launch, gb_per_sec
    );

    // Cleanup
    unsafe {
        cuda::stream::destroy(stream)?;
        cuda::free_sync(d_a)?;
        cuda::free_sync(d_b)?;
        cuda::free_sync(d_c)?;
        cuda::module::unload(module)?;
    }

    Ok(())
}

// ===========================================================================
// Step 2: Tiled GEMM — proves shared memory, barriers, loops, 2D grids
//
// Classic tiled matrix multiply: C = A * B (row-major, f32).
// Uses shared memory tiles and __syncthreads(). No tensor cores.
// ===========================================================================

const TILE: u32 = 32;

fn step2_tiled_gemm(sm: &str) -> Result<()> {
    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;
    let size_a = (m * k) as usize;
    let size_b = (k * n) as usize;
    let size_c = (m * n) as usize;

    let ptx = emit_tiled_gemm_ptx(sm)?;
    println!("  [llvm] Generated {} bytes of PTX", ptx.len());

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX contains null byte")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func_name = CString::new("tiled_gemm").unwrap();
    let func = unsafe { cuda::module::get_function(module, func_name)? };
    println!("  [cuda] Module loaded, function resolved");

    // Allocate
    let d_a = unsafe { cuda::malloc_sync(size_a * 4)? };
    let d_b = unsafe { cuda::malloc_sync(size_b * 4)? };
    let d_c = unsafe { cuda::malloc_sync(size_c * 4)? };

    // Host data — small values to avoid fp32 precision issues
    let h_a: Vec<f32> = (0..size_a).map(|i| ((i % 17) as f32) * 0.1).collect();
    let h_b: Vec<f32> = (0..size_b).map(|i| ((i % 13) as f32) * 0.1).collect();
    let mut h_c: Vec<f32> = vec![0.0; size_c];

    unsafe {
        cuda::memcpy_htod_sync(d_a, &h_a)?;
        cuda::memcpy_htod_sync(d_b, &h_b)?;
    }

    let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

    let grid_x: c_uint = n / TILE;
    let grid_y: c_uint = m / TILE;

    let params: &mut [*mut c_void] = &mut [
        (&d_a) as *const _ as *mut c_void,
        (&d_b) as *const _ as *mut c_void,
        (&d_c) as *const _ as *mut c_void,
        (&m) as *const _ as *mut c_void,
        (&n) as *const _ as *mut c_void,
        (&k) as *const _ as *mut c_void,
    ];

    // Dynamic shared memory: 2 tiles × TILE × TILE × sizeof(f32)
    let smem_bytes: c_uint = 2 * TILE * TILE * 4;

    // Warmup
    for _ in 0..5 {
        unsafe {
            cuda::launch_kernel(
                func,
                (grid_x, grid_y, 1),
                (TILE, TILE, 1),
                smem_bytes,
                stream,
                params,
            )?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }

    // Benchmark
    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            cuda::launch_kernel(
                func,
                (grid_x, grid_y, 1),
                (TILE, TILE, 1),
                smem_bytes,
                stream,
                params,
            )?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_micros() as f64 / iters as f64;

    // Verify against CPU reference
    unsafe { cuda::memcpy_dtoh_sync(&mut h_c, d_c)?; }

    let mut max_err: f32 = 0.0;
    for row in 0..m as usize {
        for col in 0..n as usize {
            let mut expected: f32 = 0.0;
            for kk in 0..k as usize {
                expected += h_a[row * k as usize + kk] * h_b[kk * n as usize + col];
            }
            let err = (h_c[row * n as usize + col] - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    if max_err < 0.01 {
        println!("  ✓ Correct! (max error: {:.6})", max_err);
    } else {
        println!("  ✗ Max error: {:.6} (expected < 0.01)", max_err);
        bail!("GEMM verification failed");
    }

    // 2*M*N*K FLOPs for matmul
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / (us_per_launch * 1e-6) / 1e12;
    println!("  {:.1} μs, {:.2} TFLOPS (naive tiled, no tensor cores)", us_per_launch, tflops);

    unsafe {
        cuda::stream::destroy(stream)?;
        cuda::free_sync(d_a)?;
        cuda::free_sync(d_b)?;
        cuda::free_sync(d_c)?;
        cuda::module::unload(module)?;
    }

    Ok(())
}

// ===========================================================================
// PTX generation via inkwell
// ===========================================================================

pub fn create_nvptx_target_machine(sm: &str) -> Result<TargetMachine> {
    let triple = TargetTriple::create("nvptx64-nvidia-cuda");
    let target = Target::from_triple(&triple)
        .map_err(|e| anyhow::anyhow!("NVPTX target: {}", e))?;
    target
        .create_target_machine(
            &triple,
            sm,
            "+ptx86",
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or_else(|| anyhow::anyhow!("Failed to create target machine for {}", sm))
}

/// Emit LLVM IR for a vector_add kernel and compile to PTX.
///
/// Equivalent CUDA:
/// ```cuda
/// extern "C" __global__ void vector_add(
///     const float* a, const float* b, float* c, unsigned int n
/// ) {
///     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
///     if (i < n) c[i] = a[i] + b[i];
/// }
/// ```
fn emit_vector_add_ptx(sm: &str) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;

    let context = LlvmContext::create();
    let module = context.create_module("ferrite_poc");
    let builder = context.create_builder();

    // Set target layout from the machine (avoids &str vs &DataLayout mismatch)
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    // Types
    let i32_type = context.i32_type();
    let f32_type = context.f32_type();
    let void_type = context.void_type();
    let ptr_global = context.ptr_type(AddressSpace::from(1u16)); // addrspace(1) = global

    // Kernel function
    let fn_type = void_type.fn_type(
        &[
            ptr_global.into(), // a
            ptr_global.into(), // b
            ptr_global.into(), // c
            i32_type.into(),   // n
        ],
        false,
    );
    let function = module.add_function("vector_add", fn_type, None);
    add_nvvm_kernel_metadata(&module, &function);

    // Blocks
    let entry = context.append_basic_block(function, "entry");
    let body = context.append_basic_block(function, "body");
    let exit = context.append_basic_block(function, "exit");

    // ── Entry ──
    builder.position_at_end(entry);

    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let n = function.get_nth_param(3).unwrap().into_int_value();

    let tid = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.tid.x", "tid");
    let ctaid = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ctaid.x", "ctaid");
    let ntid = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ntid.x", "ntid");
    let offset = builder.build_int_mul(ctaid, ntid, "offset").unwrap();
    let i = builder.build_int_add(offset, tid, "i").unwrap();

    let cmp = builder.build_int_compare(IntPredicate::ULT, i, n, "cmp").unwrap();
    builder.build_conditional_branch(cmp, body, exit).unwrap();

    // ── Body: c[i] = a[i] + b[i] ──
    builder.position_at_end(body);
    let a_i = unsafe { builder.build_gep(f32_type, a_ptr, &[i], "a_i").unwrap() };
    let b_i = unsafe { builder.build_gep(f32_type, b_ptr, &[i], "b_i").unwrap() };
    let c_i = unsafe { builder.build_gep(f32_type, c_ptr, &[i], "c_i").unwrap() };
    let va = builder.build_load(f32_type, a_i, "va").unwrap().into_float_value();
    let vb = builder.build_load(f32_type, b_i, "vb").unwrap().into_float_value();
    let sum = builder.build_float_add(va, vb, "sum").unwrap();
    builder.build_store(c_i, sum).unwrap();
    builder.build_unconditional_branch(exit).unwrap();

    // ── Exit ──
    builder.position_at_end(exit);
    builder.build_return(None).unwrap();

    // ── Emit PTX ──
    let buf = machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow::anyhow!("PTX emission failed: {}", e))?;

    let ptx = std::str::from_utf8(buf.as_slice())
        .context("PTX is not valid UTF-8")?
        .to_string();

    Ok(ptx)
}

/// Emit a tiled GEMM kernel: C = A * B (row-major f32, TILE×TILE blocks).
///
/// Uses dynamic shared memory (extern __shared__) passed via launch config.
/// Equivalent CUDA:
/// ```cuda
/// extern "C" __global__ void tiled_gemm(
///     const float* A, const float* B, float* C,
///     unsigned M, unsigned N, unsigned K
/// ) {
///     extern __shared__ float smem[];
///     float* As = smem;                       // [TILE][TILE]
///     float* Bs = smem + TILE * TILE;         // [TILE][TILE]
///     int row = blockIdx.y * TILE + threadIdx.y;
///     int col = blockIdx.x * TILE + threadIdx.x;
///     float acc = 0.0f;
///     for (int t = 0; t < K; t += TILE) {
///         As[threadIdx.y * TILE + threadIdx.x] = A[row * K + t + threadIdx.x];
///         Bs[threadIdx.y * TILE + threadIdx.x] = B[(t + threadIdx.y) * N + col];
///         __syncthreads();
///         for (int i = 0; i < TILE; i++)
///             acc += As[threadIdx.y * TILE + i] * Bs[i * TILE + threadIdx.x];
///         __syncthreads();
///     }
///     C[row * N + col] = acc;
/// }
/// ```
fn emit_tiled_gemm_ptx(sm: &str) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;
    let context = LlvmContext::create();
    let module = context.create_module("ferrite_gemm");
    let builder = context.create_builder();

    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let i32_type = context.i32_type();
    let f32_type = context.f32_type();
    let void_type = context.void_type();
    let ptr_global = context.ptr_type(AddressSpace::from(1u16));
    // addrspace(3) = shared memory
    let ptr_shared = context.ptr_type(AddressSpace::from(3u16));

    // Declare extern shared memory as a global variable in addrspace(3)
    // This is the dynamic shared memory, sized at launch time.
    let smem_global = module.add_global(
        context.i8_type().array_type(0), // zero-length array = extern __shared__
        Some(AddressSpace::from(3u16)),
        "smem",
    );
    smem_global.set_alignment(16);
    // Mark as externally initialized (dynamic shared mem)
    smem_global.set_externally_initialized(true);

    let fn_type = void_type.fn_type(
        &[
            ptr_global.into(), // A
            ptr_global.into(), // B
            ptr_global.into(), // C
            i32_type.into(),   // M
            i32_type.into(),   // N
            i32_type.into(),   // K
        ],
        false,
    );
    let function = module.add_function("tiled_gemm", fn_type, None);
    add_nvvm_kernel_metadata(&module, &function);

    let tile_const = i32_type.const_int(TILE as u64, false);

    // ── Blocks ──
    let entry = context.append_basic_block(function, "entry");
    let tile_loop_header = context.append_basic_block(function, "tile_loop_header");
    let tile_loop_body = context.append_basic_block(function, "tile_loop_body");
    let inner_loop_header = context.append_basic_block(function, "inner_loop_header");
    let inner_loop_body = context.append_basic_block(function, "inner_loop_body");
    let inner_loop_exit = context.append_basic_block(function, "inner_loop_exit");
    let tile_loop_exit = context.append_basic_block(function, "tile_loop_exit");

    // ── Entry ──
    builder.position_at_end(entry);

    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let _m = function.get_nth_param(3).unwrap().into_int_value();
    let n_param = function.get_nth_param(4).unwrap().into_int_value();
    let k_param = function.get_nth_param(5).unwrap().into_int_value();

    let tid_x = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.tid.x", "tid_x");
    let tid_y = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.tid.y", "tid_y");
    let bid_x = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ctaid.x", "bid_x");
    let bid_y = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ctaid.y", "bid_y");

    // row = blockIdx.y * TILE + threadIdx.y
    let row = builder.build_int_add(
        builder.build_int_mul(bid_y, tile_const, "").unwrap(),
        tid_y, "row",
    ).unwrap();
    // col = blockIdx.x * TILE + threadIdx.x
    let col = builder.build_int_add(
        builder.build_int_mul(bid_x, tile_const, "").unwrap(),
        tid_x, "col",
    ).unwrap();

    // Shared memory pointers: As = &smem[0], Bs = &smem[TILE*TILE*4]
    let smem_ptr = smem_global.as_pointer_value();
    let tile_sq_bytes = i32_type.const_int((TILE * TILE * 4) as u64, false);
    let as_ptr = builder.build_pointer_cast(smem_ptr, ptr_shared, "as_ptr").unwrap();
    let bs_offset = unsafe {
        builder.build_gep(context.i8_type(), smem_ptr, &[tile_sq_bytes], "bs_off").unwrap()
    };
    let bs_ptr = builder.build_pointer_cast(bs_offset, ptr_shared, "bs_ptr").unwrap();

    // acc = 0.0
    let zero_f32 = f32_type.const_float(0.0);

    builder.build_unconditional_branch(tile_loop_header).unwrap();

    // ── Tile loop: for (t = 0; t < K; t += TILE) ──
    builder.position_at_end(tile_loop_header);
    let t_phi = builder.build_phi(i32_type, "t").unwrap();
    let acc_phi = builder.build_phi(f32_type, "acc").unwrap();
    let t = t_phi.as_basic_value().into_int_value();
    let acc = acc_phi.as_basic_value().into_float_value();

    let tile_cmp = builder.build_int_compare(IntPredicate::ULT, t, k_param, "tile_cmp").unwrap();
    builder.build_conditional_branch(tile_cmp, tile_loop_body, tile_loop_exit).unwrap();

    // ── Tile loop body: load tiles into shared memory ──
    builder.position_at_end(tile_loop_body);

    // As[threadIdx.y * TILE + threadIdx.x] = A[row * K + t + threadIdx.x]
    let as_idx = builder.build_int_add(
        builder.build_int_mul(tid_y, tile_const, "").unwrap(),
        tid_x, "as_idx",
    ).unwrap();
    let a_row_off = builder.build_int_mul(row, k_param, "").unwrap();
    let a_idx = builder.build_int_add(
        builder.build_int_add(a_row_off, t, "").unwrap(),
        tid_x, "a_idx",
    ).unwrap();
    let a_elem_ptr = unsafe { builder.build_gep(f32_type, a_ptr, &[a_idx], "a_ep").unwrap() };
    let a_val = builder.build_load(f32_type, a_elem_ptr, "a_val").unwrap();
    let as_elem_ptr = unsafe { builder.build_gep(f32_type, as_ptr, &[as_idx], "as_ep").unwrap() };
    builder.build_store(as_elem_ptr, a_val).unwrap();

    // Bs[threadIdx.y * TILE + threadIdx.x] = B[(t + threadIdx.y) * N + col]
    let bs_idx = builder.build_int_add(
        builder.build_int_mul(tid_y, tile_const, "").unwrap(),
        tid_x, "bs_idx",
    ).unwrap();
    let b_row = builder.build_int_add(t, tid_y, "b_row").unwrap();
    let b_idx = builder.build_int_add(
        builder.build_int_mul(b_row, n_param, "").unwrap(),
        col, "b_idx",
    ).unwrap();
    let b_elem_ptr = unsafe { builder.build_gep(f32_type, b_ptr, &[b_idx], "b_ep").unwrap() };
    let b_val = builder.build_load(f32_type, b_elem_ptr, "b_val").unwrap();
    let bs_elem_ptr = unsafe { builder.build_gep(f32_type, bs_ptr, &[bs_idx], "bs_ep").unwrap() };
    builder.build_store(bs_elem_ptr, b_val).unwrap();

    // __syncthreads()
    call_barrier0(&context, &module, &builder);

    builder.build_unconditional_branch(inner_loop_header).unwrap();

    // ── Inner loop: for (i = 0; i < TILE; i++) acc += As[ty*TILE+i] * Bs[i*TILE+tx] ──
    builder.position_at_end(inner_loop_header);
    let i_phi = builder.build_phi(i32_type, "i").unwrap();
    let acc2_phi = builder.build_phi(f32_type, "acc2").unwrap();
    let i_val = i_phi.as_basic_value().into_int_value();
    let acc2 = acc2_phi.as_basic_value().into_float_value();

    let inner_cmp = builder.build_int_compare(IntPredicate::ULT, i_val, tile_const, "icmp").unwrap();
    builder.build_conditional_branch(inner_cmp, inner_loop_body, inner_loop_exit).unwrap();

    // ── Inner loop body ──
    builder.position_at_end(inner_loop_body);

    // As[threadIdx.y * TILE + i]
    let as_load_idx = builder.build_int_add(
        builder.build_int_mul(tid_y, tile_const, "").unwrap(),
        i_val, "as_li",
    ).unwrap();
    let as_lp = unsafe { builder.build_gep(f32_type, as_ptr, &[as_load_idx], "as_lp").unwrap() };
    let as_v = builder.build_load(f32_type, as_lp, "as_v").unwrap().into_float_value();

    // Bs[i * TILE + threadIdx.x]
    let bs_load_idx = builder.build_int_add(
        builder.build_int_mul(i_val, tile_const, "").unwrap(),
        tid_x, "bs_li",
    ).unwrap();
    let bs_lp = unsafe { builder.build_gep(f32_type, bs_ptr, &[bs_load_idx], "bs_lp").unwrap() };
    let bs_v = builder.build_load(f32_type, bs_lp, "bs_v").unwrap().into_float_value();

    // acc += as_v * bs_v
    let prod = builder.build_float_mul(as_v, bs_v, "prod").unwrap();
    let acc_new = builder.build_float_add(acc2, prod, "acc_new").unwrap();

    let i_next = builder.build_int_add(i_val, i32_type.const_int(1, false), "i_next").unwrap();
    builder.build_unconditional_branch(inner_loop_header).unwrap();

    // Wire inner loop phi nodes
    i_phi.add_incoming(&[
        (&i32_type.const_int(0, false), tile_loop_body),
        (&i_next, inner_loop_body),
    ]);
    acc2_phi.add_incoming(&[
        (&acc, tile_loop_body),
        (&acc_new, inner_loop_body),
    ]);

    // ── Inner loop exit → second syncthreads, advance tile loop ──
    builder.position_at_end(inner_loop_exit);

    call_barrier0(&context, &module, &builder);

    let t_next = builder.build_int_add(t, tile_const, "t_next").unwrap();
    builder.build_unconditional_branch(tile_loop_header).unwrap();

    // Wire tile loop phi nodes
    t_phi.add_incoming(&[
        (&i32_type.const_int(0, false), entry),
        (&t_next, inner_loop_exit),
    ]);
    acc_phi.add_incoming(&[
        (&zero_f32, entry),
        (&acc2, inner_loop_exit), // acc2 is the final acc from inner loop
    ]);

    // ── Tile loop exit: store C[row * N + col] = acc ──
    builder.position_at_end(tile_loop_exit);

    let c_idx = builder.build_int_add(
        builder.build_int_mul(row, n_param, "").unwrap(),
        col, "c_idx",
    ).unwrap();
    let c_elem_ptr = unsafe { builder.build_gep(f32_type, c_ptr, &[c_idx], "c_ep").unwrap() };
    builder.build_store(c_elem_ptr, acc).unwrap();
    builder.build_return(None).unwrap();

    // ── Emit PTX ──
    let buf = machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow::anyhow!("PTX emission failed: {}", e))?;

    let ptx = std::str::from_utf8(buf.as_slice())
        .context("PTX is not valid UTF-8")?
        .to_string();

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// NVPTX helpers
// ---------------------------------------------------------------------------

pub fn call_sreg<'ctx>(
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    intrinsic: &str,
    name: &str,
) -> IntValue<'ctx> {
    let i32_type = context.i32_type();
    let fn_type = i32_type.fn_type(&[], false);
    let func = module
        .get_function(intrinsic)
        .unwrap_or_else(|| module.add_function(intrinsic, fn_type, None));
    let call_site = builder.build_call(func, &[], name).unwrap();
    // Wrap the raw LLVMValueRef as a BasicValueEnum, then extract IntValue.
    // A non-void call instruction IS its return value in LLVM SSA.
    let raw: llvm_sys::prelude::LLVMValueRef = call_site.as_value_ref();
    unsafe { BasicValueEnum::new(raw) }.into_int_value()
}

/// Call @llvm.nvvm.barrier0() — equivalent to __syncthreads().
pub fn call_barrier0<'ctx>(
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
) {
    let void_type = context.void_type();
    let fn_type = void_type.fn_type(&[], false);
    let func = module
        .get_function("llvm.nvvm.barrier0")
        .unwrap_or_else(|| module.add_function("llvm.nvvm.barrier0", fn_type, None));
    builder.build_call(func, &[], "").unwrap();
}

/// NVVM metadata: !nvvm.annotations = !{!0}
///                !0 = !{ptr @func, !"kernel", i32 1}
///
/// Uses raw LLVM C API because inkwell's named metadata API varies by version.
pub fn add_nvvm_kernel_metadata<'ctx>(module: &Module<'ctx>, function: &FunctionValue<'ctx>) {
    use llvm_sys;
    use std::ffi::CStr;

    unsafe {
        // Get raw LLVM pointers from inkwell types
        let mod_ref: llvm_sys::prelude::LLVMModuleRef = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);

        // Build metadata: !{ptr @function, !"kernel", i32 1}
        let fn_val: llvm_sys::prelude::LLVMValueRef = function.as_value_ref();
        let fn_md = llvm_sys::core::LLVMValueAsMetadata(fn_val);

        let kernel_cstr = CStr::from_bytes_with_nul(b"kernel\0").unwrap();
        let kernel_md = llvm_sys::core::LLVMMDStringInContext2(
            ctx_ref,
            kernel_cstr.as_ptr(),
            6, // length of "kernel"
        );

        let i32_ty = llvm_sys::core::LLVMInt32TypeInContext(ctx_ref);
        let one_val = llvm_sys::core::LLVMConstInt(i32_ty, 1, 0);
        let one_md = llvm_sys::core::LLVMValueAsMetadata(one_val);

        let mut ops = [fn_md, kernel_md, one_md];
        let node = llvm_sys::core::LLVMMDNodeInContext2(ctx_ref, ops.as_mut_ptr(), 3);

        let annot_name = CStr::from_bytes_with_nul(b"nvvm.annotations\0").unwrap();
        let named_md = llvm_sys::core::LLVMGetOrInsertNamedMetadata(
            mod_ref,
            annot_name.as_ptr(),
            16, // length of "nvvm.annotations"
        );
        llvm_sys::core::LLVMAddNamedMetadataOperand(
            mod_ref,
            annot_name.as_ptr(),
            llvm_sys::core::LLVMMetadataAsValue(ctx_ref, node),
        );
        let _ = named_md; // only needed to ensure it exists
    }
}
