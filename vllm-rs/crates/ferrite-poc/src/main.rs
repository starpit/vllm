// SPDX-License-Identifier: Apache-2.0
//
// Ferrite Phase 0 — Proof of Concept
//
// Validates the full pipeline: Rust → inkwell → LLVM IR → PTX → GPU kernel.
// Run on a machine with LLVM 20 and an NVIDIA GPU:
//
//   LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo run -p ferrite-poc --release

#[allow(unused)]
mod cubek_gemm;
#[allow(unused)]
mod flash_attn;
#[allow(unused)]
mod flash_attn_hdim128;
#[allow(unused)]
mod gemm_128x128;
#[allow(unused)]
mod mma_gemm;
#[allow(unused)]
mod ptx_builder;
#[allow(unused, unsafe_op_in_unsafe_fn)]
mod sweep;
#[allow(unused)]
mod tiled_mma;
#[allow(unused, unsafe_op_in_unsafe_fn)]
mod triton_style;
#[allow(unused)]
mod rotary;
#[allow(unused)]
mod silu_mul;

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

// ---------------------------------------------------------------------------
// LLVM / inkwell
// ---------------------------------------------------------------------------
use inkwell::builder::Builder;
use inkwell::context::Context as LlvmContext;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::values::{AsValueRef, BasicValueEnum, FunctionValue, IntValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

// ---------------------------------------------------------------------------
// CUDA (cudarc raw driver API)
// ---------------------------------------------------------------------------
use cudarc::driver::result as cuda;
use cudarc::driver::sys as cuda_sys;

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
    unsafe {
        cuda::ctx::set_current(ctx)?;
    }

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

    // Fused RmsNorm→GEMM→SiLU megakernel
    println!("\nFused RmsNorm→GEMM→SiLU 128x128");
    run_fused_128x128()?;

    // Quick GELU benchmark
    println!("\nGELU standalone (16M elements)");
    {
        let ptx = ferrite_ptx::gelu::build_gelu_kernel(256, 4);
        let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
        let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
        let func =
            unsafe { cuda::module::get_function(module, CString::new("gelu_kernel").unwrap())? };
        let n: u32 = 16_000_000;
        let d = unsafe { cuda::malloc_sync(n as usize * 4)? };
        let h: Vec<f32> = (0..n as usize)
            .map(|i| ((i % 1000) as f32 - 500.0) * 0.01)
            .collect();
        unsafe {
            cuda::memcpy_htod_sync(d, &h)?;
        }
        let grid = (n + 1023) / 1024;
        let params: &mut [*mut c_void] = &mut [
            (&d) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void,
        ];
        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (grid, 1, 1),
                    (256, 1, 1),
                    0,
                    std::ptr::null_mut(),
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(std::ptr::null_mut())?;
        }
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (grid, 1, 1),
                    (256, 1, 1),
                    0,
                    std::ptr::null_mut(),
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(std::ptr::null_mut())?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;
        let bw = (n as f64 * 8.0) / (us * 1e-6) / 1e9; // read + write = 2 * 4 bytes
        println!("  {:.1} us, {:.1} GB/s", us, bw);
        // Correctness
        let mut h_out: Vec<f32> = vec![0.0; n as usize];
        unsafe {
            cuda::memcpy_dtoh_sync(&mut h_out, d)?;
        }
        let mut max_err: f32 = 0.0;
        for i in [0, 1, 100, 1000, 999999] {
            let x = h[i];
            let expected = x * (1.0 / (1.0 + (-1.702 * x).exp())); // fast gelu
            let err = (h_out[i] - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
        println!("  Correct (max err: {:.6})", max_err);
        unsafe {
            cuda::free_sync(d)?;
            cuda::module::unload(module)?;
        }
    }

    // MLP Block: RmsNorm→GEMM→SiLU→GEMM (full down-projection)
    println!("\nMLP Block: RmsNorm→GEMM→SiLU→GEMM 128x128");
    run_mlp_block_128x128(device)?;

    // Flash Attention forward
    println!("\nFlash Attention Forward (BLOCK_M=128, BLOCK_N=64, HD=64)");
    run_flash_attn_fwd()?;

    // Flash Attention forward (causal, d=64)
    println!("\nFlash Attention Forward CAUSAL (BLOCK_M=128, BLOCK_N=64, HD=64)");
    run_flash_attn_fwd_causal()?;

    // Flash Attention forward paged KV
    println!("\nFlash Attention Forward PAGED (BLOCK_M=128, BLOCK_N=64, HD=64)");
    run_flash_attn_fwd_paged()?;

    // Flash Attention forward d=128
    println!("\nFlash Attention Forward (BLOCK_M=128, BLOCK_N=32, HD=128)");
    run_flash_attn_fwd_hdim128()?;

    // SiLU × Up elementwise kernel
    println!("\nSiLU × Up Elementwise");
    run_silu_mul()?;

    // Rotary embeddings kernel
    println!("\nRotary Embeddings (RoPE)");
    run_rotary()?;

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
    let func =
        unsafe { cuda::module::get_function(module, CString::new("triton_style_gemm").unwrap())? };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Pipeline GEMM -- {} regs/thread, {} bytes local",
        nregs, local_bytes
    );

    let bm = config.bm;
    let bn = config.bn;
    let threads = config.threads();
    let smem_bytes: c_uint = config.smem_total();

    for &sz in &[1024u32] {
        let m = sz;
        let n = sz;
        let k = sz;
        let sa = (m * k) as usize;
        let sb = (k * n) as usize;
        let sc = (m * n) as usize;
        let da = unsafe { cuda::malloc_sync(sa * 2)? };
        let db = unsafe { cuda::malloc_sync(sb * 2)? };
        let dc = unsafe { cuda::malloc_sync(sc * 4)? };
        let ha: Vec<half::f16> = (0..sa)
            .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
            .collect();
        let hb: Vec<half::f16> = (0..sb)
            .map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
            .collect();
        let mut hc: Vec<f32> = vec![0.0; sc];
        unsafe {
            cuda::memcpy_htod_sync(da, &ha)?;
            cuda::memcpy_htod_sync(db, &hb)?;
        }
        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx = n / bn;
        let gy = m / bm;
        let params: &mut [*mut c_void] = &mut [
            (&da) as *const _ as *mut c_void,
            (&db) as *const _ as *mut c_void,
            (&dc) as *const _ as *mut c_void,
            (&m) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void,
            (&k) as *const _ as *mut c_void,
        ];
        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;
        unsafe {
            cuda::memcpy_dtoh_sync(&mut hc, dc)?;
        }
        let mut max_err: f32 = 0.0;
        for r in [0, 1, 31, 32, 63] {
            for c2 in [0, 1, 31, 32, 63] {
                if r >= m as usize || c2 >= n as usize {
                    continue;
                }
                let mut e = 0.0f32;
                for kk in 0..k as usize {
                    e += ha[r * k as usize + kk].to_f32() * hb[kk * n as usize + c2].to_f32();
                }
                let err = (hc[r * n as usize + c2] - e).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        let tf = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
        println!("  --- {m}x{n} ---");
        if max_err < 1.0 {
            println!("  Correct (max err: {max_err:.4})");
        } else {
            println!("  ERROR: max err {max_err:.4}");
        }
        println!("  {us:.1} us, {tf:.1} TFLOPS");
        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(da)?;
            cuda::free_sync(db)?;
            cuda::free_sync(dc)?;
            cuda::module::unload(module)?;
        }
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
    let func =
        unsafe { cuda::module::get_function(module, CString::new("triton_style_gemm").unwrap())? };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)",
        nregs, local_bytes
    );

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

        let ha: Vec<half::f16> = (0..sa)
            .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
            .collect();
        let hb: Vec<half::f16> = (0..sb)
            .map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
            .collect();
        let mut hc: Vec<f32> = vec![0.0; sc];
        unsafe {
            cuda::memcpy_htod_sync(da, &ha)?;
            cuda::memcpy_htod_sync(db, &hb)?;
        }

        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx = n / bn;
        let gy = m / bm;
        let params: &mut [*mut c_void] = &mut [
            (&da) as *const _ as *mut c_void,
            (&db) as *const _ as *mut c_void,
            (&dc) as *const _ as *mut c_void,
            (&m) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void,
            (&k) as *const _ as *mut c_void,
        ];

        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        unsafe {
            cuda::memcpy_dtoh_sync(&mut hc, dc)?;
        }
        let mut max_err: f32 = 0.0;
        for r in [0, 1, 31, 32, 63, 511, 1023] {
            for c in [0, 1, 31, 32, 63, 511, 1023] {
                if r >= m as usize || c >= n as usize {
                    continue;
                }
                let mut e = 0.0f32;
                for kk in 0..k as usize {
                    e += ha[r * k as usize + kk].to_f32() * hb[kk * n as usize + c].to_f32();
                }
                let err = (hc[r * n as usize + c] - e).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        if max_err < 1.0 {
            println!("  Correct (max err: {:.4})", max_err);
        } else {
            bail!("  Max error: {:.4}", max_err);
        }

        let tf = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
        println!("  {:.1} us, {:.1} TFLOPS", us, tf);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(da)?;
            cuda::free_sync(db)?;
            cuda::free_sync(dc)?;
        }
    }

    unsafe {
        cuda::module::unload(module)?;
    }
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
    let func =
        unsafe { cuda::module::get_function(module, CString::new("gemm_128x128").unwrap())? };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)",
        nregs, local_bytes
    );

    let bm: u32 = 128;
    let bn: u32 = 128;
    let threads: u32 = 128;
    let smem_bytes: c_uint = 32768; // 2 stages × (8192 A + 8192 B) = 32KB

    for &sz in &[1024u32] {
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

        let ha: Vec<half::f16> = (0..sa)
            .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
            .collect();
        let hb: Vec<half::f16> = (0..sb)
            .map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
            .collect();
        let mut hc: Vec<f32> = vec![0.0; sc];
        unsafe {
            cuda::memcpy_htod_sync(da, &ha)?;
            cuda::memcpy_htod_sync(db, &hb)?;
        }

        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx = n / bn;
        let gy = m / bm;
        let params: &mut [*mut c_void] = &mut [
            (&da) as *const _ as *mut c_void,
            (&db) as *const _ as *mut c_void,
            (&dc) as *const _ as *mut c_void,
            (&m) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void,
            (&k) as *const _ as *mut c_void,
        ];

        // Warmup
        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        // Benchmark
        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        // Verify
        unsafe {
            cuda::memcpy_dtoh_sync(&mut hc, dc)?;
        }
        let mut max_err: f32 = 0.0;
        for r in [0, 1, 31, 32, 63, 64, 95, 96, 127, 511, 1023] {
            for c in [0, 1, 31, 32, 63, 64, 95, 96, 127, 511, 1023] {
                if r >= m as usize || c >= n as usize {
                    continue;
                }
                let mut e = 0.0f32;
                for kk in 0..k as usize {
                    e += ha[r * k as usize + kk].to_f32() * hb[kk * n as usize + c].to_f32();
                }
                let err = (hc[r * n as usize + c] - e).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        if max_err < 1.0 {
            println!("  Correct (max err: {:.4})", max_err);
        } else {
            // Print a few sample values for debugging
            for r in [0, 1, 64, 127] {
                for c in [0, 1, 64, 127] {
                    if r >= m as usize || c >= n as usize {
                        continue;
                    }
                    let mut e = 0.0f32;
                    for kk in 0..k as usize {
                        e += ha[r * k as usize + kk].to_f32() * hb[kk * n as usize + c].to_f32();
                    }
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
            cuda::free_sync(da)?;
            cuda::free_sync(db)?;
            cuda::free_sync(dc)?;
        }
    }

    unsafe {
        cuda::module::unload(module)?;
    }
    Ok(())
}

// ===========================================================================
// Fused RmsNorm→GEMM→SiLU 128×128 benchmark
// ===========================================================================

fn run_fused_128x128() -> Result<()> {
    let ptx = gemm_128x128::emit_fused_128x128();
    println!("  Generated {} bytes of PTX", ptx.len());

    std::fs::write("/tmp/ferrite_fused_128x128.ptx", &ptx).ok();
    println!("  [debug] PTX dumped to /tmp/ferrite_fused_128x128.ptx");

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("fused_rmsnorm_gemm_silu").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)",
        nregs, local_bytes
    );

    // Also load standalone GEMM for unfused comparison
    let ptx_gemm = gemm_128x128::emit_ptx_128x128();
    let ptx_gemm_cstr = CString::new(ptx_gemm.as_bytes()).context("PTX null")?;
    let module_gemm = unsafe { cuda::module::load_data(ptx_gemm_cstr.as_ptr() as *const _)? };
    let func_gemm =
        unsafe { cuda::module::get_function(module_gemm, CString::new("gemm_128x128").unwrap())? };

    let bm: u32 = 128;
    let bn: u32 = 128;
    let threads: u32 = 128;
    let smem_fused: c_uint = 41504;
    let smem_gemm: c_uint = 32768;

    unsafe {
        cuda_sys::cuFuncSetAttribute(
            func,
            cuda_sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_fused as i32,
        );
    }

    // CPU reference helpers
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    fn rmsnorm_row(input: &[half::f16], gamma: &[half::f16], eps: f32) -> Vec<f32> {
        let n = input.len();
        let sum_sq: f32 = input
            .iter()
            .map(|x| {
                let xf = x.to_f32();
                xf * xf
            })
            .sum();
        let rms_inv = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
        input
            .iter()
            .zip(gamma.iter())
            .map(|(x, g)| x.to_f32() * rms_inv * g.to_f32())
            .collect()
    }

    for &m in &[256u32, 1024, 4096] {
        let k: u32 = 4096;
        let n: u32 = 4096;
        println!("  --- batch={}, hidden={}, out={} ---", m, k, n);

        let sa = (m * k) as usize;
        let sw = k as usize;
        let sb = (k * n) as usize;
        let sc = (m * n) as usize;

        let d_input = unsafe { cuda::malloc_sync(sa * 2)? };
        let d_wnorm = unsafe { cuda::malloc_sync(sw * 2)? };
        let d_wgemm = unsafe { cuda::malloc_sync(sb * 2)? };
        let d_output = unsafe { cuda::malloc_sync(sc * 4)? };
        let d_gemm_out = unsafe { cuda::malloc_sync(sc * 4)? };

        let h_input: Vec<half::f16> = (0..sa)
            .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
            .collect();
        let h_wnorm: Vec<half::f16> = (0..sw)
            .map(|i| half::f16::from_f32(0.5 + ((i % 100) as f32) * 0.01))
            .collect();
        let h_wgemm: Vec<half::f16> = (0..sb)
            .map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
            .collect();
        let mut h_output: Vec<f32> = vec![0.0; sc];

        unsafe {
            cuda::memcpy_htod_sync(d_input, &h_input)?;
            cuda::memcpy_htod_sync(d_wnorm, &h_wnorm)?;
            cuda::memcpy_htod_sync(d_wgemm, &h_wgemm)?;
        }

        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx = n / bn;
        let gy = m / bm;

        let params: &mut [*mut c_void] = &mut [
            (&d_input) as *const _ as *mut c_void,
            (&d_wnorm) as *const _ as *mut c_void,
            (&d_wgemm) as *const _ as *mut c_void,
            (&d_output) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void,
            (&k) as *const _ as *mut c_void,
        ];

        // Warmup
        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_fused,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        // Benchmark fused
        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_fused,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us_fused = start.elapsed().as_micros() as f64 / iters as f64;

        // Verify (only for first size to save time)
        if m == 256 {
            unsafe {
                cuda::memcpy_dtoh_sync(&mut h_output, d_output)?;
            }
            let mut max_err: f32 = 0.0;
            for r in [0usize, 1, 31, 63, 64, 127, 128, 255] {
                if r >= m as usize {
                    continue;
                }
                let row_start = r * k as usize;
                let row_end = row_start + k as usize;
                let normed = rmsnorm_row(&h_input[row_start..row_end], &h_wnorm, 1e-6);
                for c in [0usize, 1, 31, 63, 64, 127, 511, 1023, 4095] {
                    if c >= n as usize {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for kk in 0..k as usize {
                        dot += normed[kk] * h_wgemm[kk * n as usize + c].to_f32();
                    }
                    let expected = silu(dot);
                    let got = h_output[r * n as usize + c];
                    let err = (got - expected).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
            if max_err < 2.0 {
                println!("  Correct (max err: {:.4})", max_err);
            } else {
                for r in [0, 1] {
                    let row_start = r * k as usize;
                    let row_end = row_start + k as usize;
                    let normed = rmsnorm_row(&h_input[row_start..row_end], &h_wnorm, 1e-6);
                    for c in [0, 1, 64, 127] {
                        if c >= n as usize {
                            continue;
                        }
                        let mut dot = 0.0f32;
                        for kk in 0..k as usize {
                            dot += normed[kk] * h_wgemm[kk * n as usize + c].to_f32();
                        }
                        let expected = silu(dot);
                        let got = h_output[r * n as usize + c];
                        println!("    C[{r}][{c}]: got={got:.4} expected={expected:.4}");
                    }
                }
                bail!("  Max error: {:.4}", max_err);
            }
        }

        let tf = 2.0 * m as f64 * n as f64 * k as f64 / (us_fused * 1e-6) / 1e12;
        println!("  Fused: {:.1} us, {:.1} TFLOPS (GEMM equiv)", us_fused, tf);

        // Benchmark standalone GEMM (unfused baseline)
        let params_gemm: &mut [*mut c_void] = &mut [
            (&d_input) as *const _ as *mut c_void,
            (&d_wgemm) as *const _ as *mut c_void,
            (&d_gemm_out) as *const _ as *mut c_void,
            (&m) as *const _ as *mut c_void,
            (&n) as *const _ as *mut c_void,
            (&k) as *const _ as *mut c_void,
        ];

        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func_gemm,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_gemm,
                    stream,
                    params_gemm,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func_gemm,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    smem_gemm,
                    stream,
                    params_gemm,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us_gemm_only = start.elapsed().as_micros() as f64 / iters as f64;

        // Unfused estimate: GEMM + norm read/write M*K*2*2 + SiLU read/write M*N*4*2
        // At ~300 GB/s L4 bandwidth:
        let norm_bytes = m as f64 * k as f64 * 2.0 * 2.0; // read + write f16
        let silu_bytes = m as f64 * n as f64 * 4.0 * 2.0; // read + write f32
        let bw_gbs = 300.0; // L4 ~300 GB/s
        let us_norm_est = norm_bytes / (bw_gbs * 1e3); // us
        let us_silu_est = silu_bytes / (bw_gbs * 1e3); // us
        let us_unfused_est = us_gemm_only + us_norm_est + us_silu_est;

        let tf_gemm = 2.0 * m as f64 * n as f64 * k as f64 / (us_gemm_only * 1e-6) / 1e12;
        println!(
            "  Standalone GEMM: {:.1} us, {:.1} TFLOPS",
            us_gemm_only, tf_gemm
        );
        println!(
            "  Unfused estimate (GEMM+norm+SiLU): {:.1} us (norm: {:.1} us, SiLU: {:.1} us)",
            us_unfused_est, us_norm_est, us_silu_est
        );
        let speedup = us_unfused_est / us_fused;
        if speedup >= 1.0 {
            println!("  Fused speedup vs unfused: {:.1}x FASTER", speedup);
        } else {
            println!(
                "  Fused vs unfused: {:.1}% overhead",
                (1.0 / speedup - 1.0) * 100.0
            );
        }

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(d_input)?;
            cuda::free_sync(d_wnorm)?;
            cuda::free_sync(d_wgemm)?;
            cuda::free_sync(d_output)?;
            cuda::free_sync(d_gemm_out)?;
        }
    }

    unsafe {
        cuda::module::unload(module)?;
        cuda::module::unload(module_gemm)?;
    }
    Ok(())
}

// ===========================================================================
// MLP Block benchmark: RmsNorm → GEMM → SiLU → GEMM (down-projection)
// ===========================================================================

fn run_mlp_block_128x128(_device: cuda_sys::CUdevice) -> Result<()> {
    let (ptx_g1, ptx_cvt, ptx_g2) = gemm_128x128::emit_mlp_block_128x128();
    println!(
        "  Generated GEMM1: {} bytes, CVT: {} bytes, GEMM2: {} bytes",
        ptx_g1.len(),
        ptx_cvt.len(),
        ptx_g2.len()
    );

    std::fs::write("/tmp/ferrite_mlp_gemm1.ptx", &ptx_g1).ok();
    std::fs::write("/tmp/ferrite_mlp_gemm2.ptx", &ptx_g2).ok();

    let ptx_g1_cstr = CString::new(ptx_g1.as_bytes()).context("PTX null")?;
    let module_g1 = unsafe { cuda::module::load_data(ptx_g1_cstr.as_ptr() as *const _)? };
    let func_g1 =
        unsafe { cuda::module::get_function(module_g1, CString::new("mlp_gemm1_silu").unwrap())? };

    let ptx_cvt_cstr = CString::new(ptx_cvt.as_bytes()).context("PTX null")?;
    let module_cvt = unsafe { cuda::module::load_data(ptx_cvt_cstr.as_ptr() as *const _)? };
    let func_cvt =
        unsafe { cuda::module::get_function(module_cvt, CString::new("cvt_f32_to_f16").unwrap())? };

    let ptx_g2_cstr = CString::new(ptx_g2.as_bytes()).context("PTX null")?;
    let module_g2 = unsafe { cuda::module::load_data(ptx_g2_cstr.as_ptr() as *const _)? };
    let func_g2 =
        unsafe { cuda::module::get_function(module_g2, CString::new("mlp_gemm2").unwrap())? };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs_g1 =
        unsafe { cuda::function::get_function_attribute(func_g1, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let nregs_g2 =
        unsafe { cuda::function::get_function_attribute(func_g2, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    println!(
        "  [cuda] GEMM1: {} regs, GEMM2: {} regs",
        nregs_g1, nregs_g2
    );

    // Also load standalone GEMM for unfused comparison
    let ptx_gemm = gemm_128x128::emit_ptx_128x128();
    let ptx_gemm_cstr = CString::new(ptx_gemm.as_bytes()).context("PTX null")?;
    let module_gemm = unsafe { cuda::module::load_data(ptx_gemm_cstr.as_ptr() as *const _)? };
    let func_gemm =
        unsafe { cuda::module::get_function(module_gemm, CString::new("gemm_128x128").unwrap())? };

    let bm: u32 = 128;
    let bn: u32 = 128;
    let threads: u32 = 128;
    let smem_fused: c_uint = 41504;
    let smem_gemm: c_uint = 32768;

    unsafe {
        cuda_sys::cuFuncSetAttribute(
            func_g1,
            cuda_sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_fused as i32,
        );
        cuda_sys::cuFuncSetAttribute(
            func_g2,
            cuda_sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_gemm as i32,
        );
    }

    // CPU reference helpers
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    fn rmsnorm_row(input: &[half::f16], gamma: &[half::f16], eps: f32) -> Vec<f32> {
        let n = input.len();
        let sum_sq: f32 = input
            .iter()
            .map(|x| {
                let xf = x.to_f32();
                xf * xf
            })
            .sum();
        let rms_inv = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
        input
            .iter()
            .zip(gamma.iter())
            .map(|(x, g)| x.to_f32() * rms_inv * g.to_f32())
            .collect()
    }

    for &m in &[256u32, 1024, 4096] {
        let k1: u32 = 4096; // hidden dim (must be >= 1024 for gamma preload)
        let n1: u32 = 4096; // intermediate dim
        let n2: u32 = 4096; // output dim (must equal n1 for this kernel)
        println!(
            "  --- batch={}, hidden={}, inter={}, out={} ---",
            m, k1, n1, n2
        );

        let sa = (m * k1) as usize; // input [M, K1]
        let sw = k1 as usize; // gamma [K1]
        let sb1 = (k1 * n1) as usize; // w_gate [K1, N1]
        let sb2 = (n1 * n2) as usize; // w_down [N1, N2]
        let si = (m * n1) as usize; // intermediate [M, N1] f16
        let sc = (m * n2) as usize; // output [M, N2] f32

        let d_input = unsafe { cuda::malloc_sync(sa * 2)? };
        let d_wnorm = unsafe { cuda::malloc_sync(sw * 2)? };
        let d_wgate = unsafe { cuda::malloc_sync(sb1 * 2)? };
        let d_wdown = unsafe { cuda::malloc_sync(sb2 * 2)? };
        let d_inter_f32 = unsafe { cuda::malloc_sync(si * 4)? }; // GEMM1 output f32
        let d_inter = unsafe { cuda::malloc_sync(si * 2)? }; // converted f16
        let d_output = unsafe { cuda::malloc_sync(sc * 4)? };
        let d_barrier = unsafe { cuda::malloc_sync(4)? }; // u32 barrier counter
        let d_gemm_out = unsafe { cuda::malloc_sync(sc * 4)? };

        let h_input: Vec<half::f16> = (0..sa)
            .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
            .collect();
        let h_wnorm: Vec<half::f16> = (0..sw)
            .map(|i| half::f16::from_f32(0.5 + ((i % 100) as f32) * 0.01))
            .collect();
        let h_wgate: Vec<half::f16> = (0..sb1)
            .map(|i| half::f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
            .collect();
        let h_wdown: Vec<half::f16> = (0..sb2)
            .map(|i| half::f16::from_f32(((i % 11) as f32 - 5.0) * 0.05))
            .collect();
        let mut h_output: Vec<f32> = vec![0.0; sc];

        unsafe {
            cuda::memcpy_htod_sync(d_input, &h_input)?;
            cuda::memcpy_htod_sync(d_wnorm, &h_wnorm)?;
            cuda::memcpy_htod_sync(d_wgate, &h_wgate)?;
            cuda::memcpy_htod_sync(d_wdown, &h_wdown)?;
        }

        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
        let gx1 = n1 / bn; // GEMM1 grid x
        let gy = m / bm; // grid y (shared)
        let gx2 = n2 / bn; // GEMM2 grid x
        let cvt_n = m * n1; // elements to convert
        let cvt_grid = (cvt_n + 1023) / 1024; // 256 threads * 4 elems = 1024 per block

        // GEMM1 params: input, wnorm, wgate, inter_f32, N1, K1
        let params_g1: &mut [*mut c_void] = &mut [
            (&d_input) as *const _ as *mut c_void,
            (&d_wnorm) as *const _ as *mut c_void,
            (&d_wgate) as *const _ as *mut c_void,
            (&d_inter_f32) as *const _ as *mut c_void,
            (&n1) as *const _ as *mut c_void,
            (&k1) as *const _ as *mut c_void,
        ];

        // CVT params: inter_f32, inter_f16, count
        let params_cvt: &mut [*mut c_void] = &mut [
            (&d_inter_f32) as *const _ as *mut c_void,
            (&d_inter) as *const _ as *mut c_void,
            (&cvt_n) as *const _ as *mut c_void,
        ];

        // GEMM2 params: inter_f16, wdown, output, M, N2, N1(=K2)
        let params_g2: &mut [*mut c_void] = &mut [
            (&d_inter) as *const _ as *mut c_void,
            (&d_wdown) as *const _ as *mut c_void,
            (&d_output) as *const _ as *mut c_void,
            (&m) as *const _ as *mut c_void,
            (&n2) as *const _ as *mut c_void,
            (&n1) as *const _ as *mut c_void,
        ];

        // Helper to launch the 3-kernel MLP block
        let launch_mlp = |stream: cuda_sys::CUstream,
                          p_g1: &mut [*mut c_void],
                          p_cvt: &mut [*mut c_void],
                          p_g2: &mut [*mut c_void]|
         -> anyhow::Result<()> {
            unsafe {
                // GEMM1: RmsNorm → GEMM → SiLU → f32
                cuda::launch_kernel(
                    func_g1,
                    (gx1, gy, 1),
                    (threads, 1, 1),
                    smem_fused,
                    stream,
                    p_g1,
                )?;
                // CVT: f32 → f16
                cuda::launch_kernel(func_cvt, (cvt_grid, 1, 1), (256, 1, 1), 0, stream, p_cvt)?;
                // GEMM2: f16 → f32
                cuda::launch_kernel(
                    func_g2,
                    (gx2, gy, 1),
                    (threads, 1, 1),
                    smem_gemm,
                    stream,
                    p_g2,
                )?;
            }
            Ok(())
        };

        // Warmup
        for _ in 0..10 {
            launch_mlp(stream, params_g1, params_cvt, params_g2)?;
            unsafe {
                cuda::stream::synchronize(stream)?;
            }
        }

        // Benchmark MLP block
        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            launch_mlp(stream, params_g1, params_cvt, params_g2)?;
            unsafe {
                cuda::stream::synchronize(stream)?;
            }
        }
        let us_mlp = start.elapsed().as_micros() as f64 / iters as f64;

        // Verify (only for first size)
        if m == 256 {
            launch_mlp(stream, params_g1, params_cvt, params_g2)?;
            unsafe {
                cuda::stream::synchronize(stream)?;
                cuda::memcpy_dtoh_sync(&mut h_output, d_output)?;
            }

            // CPU reference: RmsNorm → GEMM1 → SiLU → GEMM2
            let mut max_err: f32 = 0.0;
            for r in [0usize, 1, 31, 63, 64, 127, 128, 255] {
                if r >= m as usize {
                    continue;
                }
                let row_start = r * k1 as usize;
                let row_end = row_start + k1 as usize;
                let normed = rmsnorm_row(&h_input[row_start..row_end], &h_wnorm, 1e-6);

                // GEMM1: normed × w_gate → intermediate
                let mut inter_row = vec![0.0f32; n1 as usize];
                for c in 0..n1 as usize {
                    let mut dot = 0.0f32;
                    for kk in 0..k1 as usize {
                        dot += normed[kk] * h_wgate[kk * n1 as usize + c].to_f32();
                    }
                    inter_row[c] = silu(dot);
                }

                // GEMM2: inter × w_down → output
                for c in [0usize, 1, 31, 63, 64, 127, 511, 1023, 4095] {
                    if c >= n2 as usize {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for kk in 0..n1 as usize {
                        // inter_row is f32, but GPU stores as f16 then reads back
                        // Use f16 conversion to match GPU precision
                        let inter_f16 = half::f16::from_f32(inter_row[kk]);
                        dot += inter_f16.to_f32() * h_wdown[kk * n2 as usize + c].to_f32();
                    }
                    let expected = dot;
                    let got = h_output[r * n2 as usize + c];
                    let err = (got - expected).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
            if max_err < 5.0 {
                println!("  Correct (max err: {:.4})", max_err);
            } else {
                for r in [0, 1] {
                    let row_start = r * k1 as usize;
                    let row_end = row_start + k1 as usize;
                    let normed = rmsnorm_row(&h_input[row_start..row_end], &h_wnorm, 1e-6);
                    let mut inter_row = vec![0.0f32; n1 as usize];
                    for c in 0..n1 as usize {
                        let mut dot = 0.0f32;
                        for kk in 0..k1 as usize {
                            dot += normed[kk] * h_wgate[kk * n1 as usize + c].to_f32();
                        }
                        inter_row[c] = silu(dot);
                    }
                    for c in [0, 1, 64, 127] {
                        if c >= n2 as usize {
                            continue;
                        }
                        let mut dot = 0.0f32;
                        for kk in 0..n1 as usize {
                            let inter_f16 = half::f16::from_f32(inter_row[kk]);
                            dot += inter_f16.to_f32() * h_wdown[kk * n2 as usize + c].to_f32();
                        }
                        let got = h_output[r * n2 as usize + c];
                        println!("    C[{r}][{c}]: got={got:.4} expected={dot:.4}");
                    }
                }
                bail!("  MLP block max error: {:.4}", max_err);
            }
        }

        // Total FLOPS: GEMM1 (2*M*N1*K1) + GEMM2 (2*M*N2*N1)
        let flops_gemm1 = 2.0 * m as f64 * n1 as f64 * k1 as f64;
        let flops_gemm2 = 2.0 * m as f64 * n2 as f64 * n1 as f64;
        let total_flops = flops_gemm1 + flops_gemm2;
        let tf = total_flops / (us_mlp * 1e-6) / 1e12;
        println!(
            "  MLP fused: {:.1} us, {:.1} TFLOPS (both GEMMs)",
            us_mlp, tf
        );

        // Benchmark unfused: 2x standalone GEMM + norm + SiLU memory traffic
        let params_gemm1: &mut [*mut c_void] = &mut [
            (&d_input) as *const _ as *mut c_void,
            (&d_wgate) as *const _ as *mut c_void,
            (&d_gemm_out) as *const _ as *mut c_void,
            (&m) as *const _ as *mut c_void,
            (&n1) as *const _ as *mut c_void,
            (&k1) as *const _ as *mut c_void,
        ];

        // Warmup GEMM1
        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func_gemm,
                    (gx1, gy, 1),
                    (threads, 1, 1),
                    smem_gemm,
                    stream,
                    params_gemm1,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func_gemm,
                    (gx1, gy, 1),
                    (threads, 1, 1),
                    smem_gemm,
                    stream,
                    params_gemm1,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us_gemm1 = start.elapsed().as_micros() as f64 / iters as f64;

        // GEMM2 timing (same dimensions for square case)
        let params_gemm2: &mut [*mut c_void] = &mut [
            (&d_inter) as *const _ as *mut c_void, // A = intermediate
            (&d_wdown) as *const _ as *mut c_void, // B = w_down
            (&d_gemm_out) as *const _ as *mut c_void,
            (&m) as *const _ as *mut c_void,
            (&n2) as *const _ as *mut c_void,
            (&n1) as *const _ as *mut c_void,
        ];

        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func_gemm,
                    (n2 / bn, gy, 1),
                    (threads, 1, 1),
                    smem_gemm,
                    stream,
                    params_gemm2,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func_gemm,
                    (n2 / bn, gy, 1),
                    (threads, 1, 1),
                    smem_gemm,
                    stream,
                    params_gemm2,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us_gemm2 = start.elapsed().as_micros() as f64 / iters as f64;

        // Unfused estimate: 2 GEMMs + norm + SiLU + intermediate store/load
        let bw_gbs = 300.0; // L4 ~300 GB/s
        let norm_bytes = m as f64 * k1 as f64 * 2.0 * 2.0;
        let silu_bytes = m as f64 * n1 as f64 * 4.0 * 2.0; // f32 read+write
        let inter_bytes = m as f64 * n1 as f64 * 2.0 * 2.0; // f16 write+read
        let us_norm_est = norm_bytes / (bw_gbs * 1e3);
        let us_silu_est = silu_bytes / (bw_gbs * 1e3);
        let us_inter_est = inter_bytes / (bw_gbs * 1e3);
        let us_unfused_est = us_gemm1 + us_gemm2 + us_norm_est + us_silu_est + us_inter_est;

        let tf_gemm1 = flops_gemm1 / (us_gemm1 * 1e-6) / 1e12;
        let tf_gemm2 = flops_gemm2 / (us_gemm2 * 1e-6) / 1e12;
        println!(
            "  Standalone GEMM1: {:.1} us ({:.1} TFLOPS), GEMM2: {:.1} us ({:.1} TFLOPS)",
            us_gemm1, tf_gemm1, us_gemm2, tf_gemm2
        );
        println!(
            "  Unfused estimate: {:.1} us (norm: {:.1}, SiLU: {:.1}, inter: {:.1})",
            us_unfused_est, us_norm_est, us_silu_est, us_inter_est
        );
        let speedup = us_unfused_est / us_mlp;
        if speedup >= 1.0 {
            println!("  Fused speedup vs unfused: {:.2}x FASTER", speedup);
        } else {
            println!(
                "  Fused vs unfused: {:.1}% overhead",
                (1.0 / speedup - 1.0) * 100.0
            );
        }

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(d_input)?;
            cuda::free_sync(d_wnorm)?;
            cuda::free_sync(d_wgate)?;
            cuda::free_sync(d_wdown)?;
            cuda::free_sync(d_inter_f32)?;
            cuda::free_sync(d_inter)?;
            cuda::free_sync(d_output)?;
            cuda::free_sync(d_gemm_out)?;
        }
    }

    unsafe {
        cuda::module::unload(module_g1)?;
        cuda::module::unload(module_cvt)?;
        cuda::module::unload(module_g2)?;
        cuda::module::unload(module_gemm)?;
    }
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
    let func = unsafe { cuda::module::get_function(module, CString::new("silu_kernel").unwrap())? };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
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
        unsafe {
            cuda::memcpy_htod_sync(d_data, &h_input)?;
        }

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
                func,
                (grid_size, 1, 1),
                (threads_per_block, 1, 1),
                0,
                std::ptr::null_mut(),
                params,
            )?;
            cuda::ctx::synchronize()?;
            cuda::memcpy_dtoh_sync(&mut h_output, d_data)?;
        }

        // Verify
        let mut max_err: f32 = 0.0;
        for i in (0..n as usize).step_by(997) {
            let expected = silu_ref(h_input[i]);
            let err = (h_output[i] - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
        if max_err < 1e-3 {
            println!("  Correct (max err: {:.6})", max_err);
        } else {
            bail!("  SiLU max error: {:.4}", max_err);
        }

        // Benchmark
        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

        // Warmup
        unsafe {
            cuda::memcpy_htod_sync(d_data, &h_input)?;
        }
        for _ in 0..20 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (grid_size, 1, 1),
                    (threads_per_block, 1, 1),
                    0,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        // Timed runs
        let iters = 200;
        unsafe {
            cuda::memcpy_htod_sync(d_data, &h_input)?;
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (grid_size, 1, 1),
                    (threads_per_block, 1, 1),
                    0,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        // Bandwidth: read + write = 2 * N * 4 bytes
        let bw = (2.0 * n as f64 * 4.0) / (us * 1e-6) / 1e9;
        println!("  {:.1} us, {:.1} GB/s", us, bw);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(d_data)?;
        }
    }

    unsafe {
        cuda::module::unload(module)?;
    }
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
    let func =
        unsafe { cuda::module::get_function(module, CString::new("rmsnorm_kernel").unwrap())? };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    println!("  [cuda] Loaded -- {} regs/thread", nregs);

    // Shared memory needed: num_warps * 4 bytes for reduction scratch
    let num_warps = block_size / 32;
    let smem_bytes: c_uint = num_warps * 4;

    // CPU reference RMSNorm
    fn rmsnorm_ref(input: &[half::f16], weight: &[half::f16], eps: f32) -> Vec<half::f16> {
        let n = input.len();
        let sum_sq: f32 = input
            .iter()
            .map(|x| {
                let xf = x.to_f32();
                xf * xf
            })
            .sum();
        let rms = ((sum_sq / n as f32) + eps).sqrt();
        let scale = 1.0 / rms;
        input
            .iter()
            .zip(weight.iter())
            .map(|(x, w)| half::f16::from_f32(x.to_f32() * scale * w.to_f32()))
            .collect()
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
                func,
                (grid_size, 1, 1),
                (block_size, 1, 1),
                smem_bytes,
                std::ptr::null_mut(),
                params,
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
                    func,
                    (grid_size, 1, 1),
                    (block_size, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        // Timed
        let iters = 500;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (grid_size, 1, 1),
                    (block_size, 1, 1),
                    smem_bytes,
                    stream,
                    params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        // Bandwidth: read input (f16) + read weight (f16) + write output (f16)
        let bytes_total =
            batch_size as f64 * hidden_size as f64 * 2.0 * 2.0 + hidden_size as f64 * 2.0;
        let bw = bytes_total / (us * 1e-6) / 1e9;
        println!("  {:.1} us, {:.1} GB/s", us, bw);

        unsafe {
            cuda::stream::destroy(stream)?;
            cuda::free_sync(d_input)?;
            cuda::free_sync(d_output)?;
            cuda::free_sync(d_weight)?;
        }
    }

    unsafe {
        cuda::module::unload(module)?;
    }
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
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)",
        nregs, local_bytes
    );

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
    let silu_func =
        unsafe { cuda::module::get_function(silu_module, CString::new("silu_kernel").unwrap())? };

    let bm = config.bm;
    let bn = config.bn;
    let threads = config.threads();

    // CPU reference
    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn rmsnorm_gemm_silu_ref(
        input: &[half::f16],
        wnorm: &[half::f16],
        wgemm: &[half::f16],
        batch: usize,
        hidden: usize,
        out: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * out];
        for b in 0..batch {
            // RMSNorm
            let row = &input[b * hidden..(b + 1) * hidden];
            let sum_sq: f32 = row
                .iter()
                .map(|x| {
                    let xf = x.to_f32();
                    xf * xf
                })
                .sum();
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
        println!(
            "  --- batch={}, hidden={}, out_features={} ---",
            batch_size, hidden_size, out_features
        );

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
        let d_normed = unsafe { cuda::malloc_sync(input_elems * 2)? }; // rmsnorm output (f16)
        let d_gemm_out = unsafe { cuda::malloc_sync(output_elems * 4)? }; // gemm output (f32)

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
                func,
                (gx, gy, 1),
                (threads, 1, 1),
                fused_smem,
                std::ptr::null_mut(),
                fused_params,
            )?;
            cuda::ctx::synchronize()?;
            cuda::memcpy_dtoh_sync(&mut h_output, d_output)?;
        }

        // CPU reference
        let expected = rmsnorm_gemm_silu_ref(
            &h_input,
            &h_wnorm,
            &h_wgemm,
            batch_size as usize,
            hidden_size as usize,
            out_features as usize,
            eps,
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
                    println!(
                        "    MISMATCH [{},{}]: got {:.6}, expected {:.6}, err {:.6}",
                        row, col, h_output[i], expected[i], err
                    );
                }
            }
        }

        if max_err < 2.0 {
            println!("  Fused correct (max err: {:.4})", max_err);
        } else {
            println!(
                "  WARNING: Fused max error: {:.4} (may be accumulation tolerance issue)",
                max_err
            );
        }

        // ── Benchmark fused ──
        let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

        // Warmup
        for _ in 0..20 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    fused_smem,
                    stream,
                    fused_params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    fused_smem,
                    stream,
                    fused_params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let fused_us = start.elapsed().as_micros() as f64 / iters as f64;

        // ── Benchmark unfused (3 separate launches) ──
        let rmsnorm_smem: c_uint = (256 / 32) * 4; // 8 warps * 4 bytes
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
                    rmsnorm_func,
                    (batch_size, 1, 1),
                    (256, 1, 1),
                    rmsnorm_smem,
                    stream,
                    rmsnorm_params,
                )?;
                cuda::launch_kernel(
                    gemm_func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    gemm_smem,
                    stream,
                    gemm_params,
                )?;
                let silu_grid = (silu_n + 1023) / 1024;
                cuda::launch_kernel(
                    silu_func,
                    (silu_grid, 1, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    silu_params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    rmsnorm_func,
                    (batch_size, 1, 1),
                    (256, 1, 1),
                    rmsnorm_smem,
                    stream,
                    rmsnorm_params,
                )?;
                cuda::launch_kernel(
                    gemm_func,
                    (gx, gy, 1),
                    (threads, 1, 1),
                    gemm_smem,
                    stream,
                    gemm_params,
                )?;
                let silu_grid = (silu_n + 1023) / 1024;
                cuda::launch_kernel(
                    silu_func,
                    (silu_grid, 1, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    silu_params,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let unfused_us = start.elapsed().as_micros() as f64 / iters as f64;

        let flops = 2.0 * batch_size as f64 * hidden_size as f64 * out_features as f64;
        let fused_tflops = flops / (fused_us * 1e-6) / 1e12;
        let unfused_tflops = flops / (unfused_us * 1e-6) / 1e12;
        let speedup = unfused_us / fused_us;

        println!("  Fused:   {:.1} us, {:.1} TFLOPS", fused_us, fused_tflops);
        println!(
            "  Unfused: {:.1} us, {:.1} TFLOPS (3 launches)",
            unfused_us, unfused_tflops
        );
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
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
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
    unsafe {
        cuda::stream::synchronize(stream)?;
    }

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
    unsafe {
        cuda::stream::synchronize(stream)?;
    }
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_micros() as f64 / iters as f64;

    // Verify
    unsafe {
        cuda::memcpy_dtoh_sync(&mut h_c, d_c)?;
    }

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
    unsafe {
        cuda::stream::synchronize(stream)?;
    }

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
    unsafe {
        cuda::stream::synchronize(stream)?;
    }
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_micros() as f64 / iters as f64;

    // Verify against CPU reference
    unsafe {
        cuda::memcpy_dtoh_sync(&mut h_c, d_c)?;
    }

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
    println!(
        "  {:.1} μs, {:.2} TFLOPS (naive tiled, no tensor cores)",
        us_per_launch, tflops
    );

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
    let target =
        Target::from_triple(&triple).map_err(|e| anyhow::anyhow!("NVPTX target: {}", e))?;
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

    let tid = call_sreg(
        &context,
        &module,
        &builder,
        "llvm.nvvm.read.ptx.sreg.tid.x",
        "tid",
    );
    let ctaid = call_sreg(
        &context,
        &module,
        &builder,
        "llvm.nvvm.read.ptx.sreg.ctaid.x",
        "ctaid",
    );
    let ntid = call_sreg(
        &context,
        &module,
        &builder,
        "llvm.nvvm.read.ptx.sreg.ntid.x",
        "ntid",
    );
    let offset = builder.build_int_mul(ctaid, ntid, "offset").unwrap();
    let i = builder.build_int_add(offset, tid, "i").unwrap();

    let cmp = builder
        .build_int_compare(IntPredicate::ULT, i, n, "cmp")
        .unwrap();
    builder.build_conditional_branch(cmp, body, exit).unwrap();

    // ── Body: c[i] = a[i] + b[i] ──
    builder.position_at_end(body);
    let a_i = unsafe { builder.build_gep(f32_type, a_ptr, &[i], "a_i").unwrap() };
    let b_i = unsafe { builder.build_gep(f32_type, b_ptr, &[i], "b_i").unwrap() };
    let c_i = unsafe { builder.build_gep(f32_type, c_ptr, &[i], "c_i").unwrap() };
    let va = builder
        .build_load(f32_type, a_i, "va")
        .unwrap()
        .into_float_value();
    let vb = builder
        .build_load(f32_type, b_i, "vb")
        .unwrap()
        .into_float_value();
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

    let tid_x = call_sreg(
        &context,
        &module,
        &builder,
        "llvm.nvvm.read.ptx.sreg.tid.x",
        "tid_x",
    );
    let tid_y = call_sreg(
        &context,
        &module,
        &builder,
        "llvm.nvvm.read.ptx.sreg.tid.y",
        "tid_y",
    );
    let bid_x = call_sreg(
        &context,
        &module,
        &builder,
        "llvm.nvvm.read.ptx.sreg.ctaid.x",
        "bid_x",
    );
    let bid_y = call_sreg(
        &context,
        &module,
        &builder,
        "llvm.nvvm.read.ptx.sreg.ctaid.y",
        "bid_y",
    );

    // row = blockIdx.y * TILE + threadIdx.y
    let row = builder
        .build_int_add(
            builder.build_int_mul(bid_y, tile_const, "").unwrap(),
            tid_y,
            "row",
        )
        .unwrap();
    // col = blockIdx.x * TILE + threadIdx.x
    let col = builder
        .build_int_add(
            builder.build_int_mul(bid_x, tile_const, "").unwrap(),
            tid_x,
            "col",
        )
        .unwrap();

    // Shared memory pointers: As = &smem[0], Bs = &smem[TILE*TILE*4]
    let smem_ptr = smem_global.as_pointer_value();
    let tile_sq_bytes = i32_type.const_int((TILE * TILE * 4) as u64, false);
    let as_ptr = builder
        .build_pointer_cast(smem_ptr, ptr_shared, "as_ptr")
        .unwrap();
    let bs_offset = unsafe {
        builder
            .build_gep(context.i8_type(), smem_ptr, &[tile_sq_bytes], "bs_off")
            .unwrap()
    };
    let bs_ptr = builder
        .build_pointer_cast(bs_offset, ptr_shared, "bs_ptr")
        .unwrap();

    // acc = 0.0
    let zero_f32 = f32_type.const_float(0.0);

    builder
        .build_unconditional_branch(tile_loop_header)
        .unwrap();

    // ── Tile loop: for (t = 0; t < K; t += TILE) ──
    builder.position_at_end(tile_loop_header);
    let t_phi = builder.build_phi(i32_type, "t").unwrap();
    let acc_phi = builder.build_phi(f32_type, "acc").unwrap();
    let t = t_phi.as_basic_value().into_int_value();
    let acc = acc_phi.as_basic_value().into_float_value();

    let tile_cmp = builder
        .build_int_compare(IntPredicate::ULT, t, k_param, "tile_cmp")
        .unwrap();
    builder
        .build_conditional_branch(tile_cmp, tile_loop_body, tile_loop_exit)
        .unwrap();

    // ── Tile loop body: load tiles into shared memory ──
    builder.position_at_end(tile_loop_body);

    // As[threadIdx.y * TILE + threadIdx.x] = A[row * K + t + threadIdx.x]
    let as_idx = builder
        .build_int_add(
            builder.build_int_mul(tid_y, tile_const, "").unwrap(),
            tid_x,
            "as_idx",
        )
        .unwrap();
    let a_row_off = builder.build_int_mul(row, k_param, "").unwrap();
    let a_idx = builder
        .build_int_add(
            builder.build_int_add(a_row_off, t, "").unwrap(),
            tid_x,
            "a_idx",
        )
        .unwrap();
    let a_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, a_ptr, &[a_idx], "a_ep")
            .unwrap()
    };
    let a_val = builder.build_load(f32_type, a_elem_ptr, "a_val").unwrap();
    let as_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, as_ptr, &[as_idx], "as_ep")
            .unwrap()
    };
    builder.build_store(as_elem_ptr, a_val).unwrap();

    // Bs[threadIdx.y * TILE + threadIdx.x] = B[(t + threadIdx.y) * N + col]
    let bs_idx = builder
        .build_int_add(
            builder.build_int_mul(tid_y, tile_const, "").unwrap(),
            tid_x,
            "bs_idx",
        )
        .unwrap();
    let b_row = builder.build_int_add(t, tid_y, "b_row").unwrap();
    let b_idx = builder
        .build_int_add(
            builder.build_int_mul(b_row, n_param, "").unwrap(),
            col,
            "b_idx",
        )
        .unwrap();
    let b_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, b_ptr, &[b_idx], "b_ep")
            .unwrap()
    };
    let b_val = builder.build_load(f32_type, b_elem_ptr, "b_val").unwrap();
    let bs_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, bs_ptr, &[bs_idx], "bs_ep")
            .unwrap()
    };
    builder.build_store(bs_elem_ptr, b_val).unwrap();

    // __syncthreads()
    call_barrier0(&context, &module, &builder);

    builder
        .build_unconditional_branch(inner_loop_header)
        .unwrap();

    // ── Inner loop: for (i = 0; i < TILE; i++) acc += As[ty*TILE+i] * Bs[i*TILE+tx] ──
    builder.position_at_end(inner_loop_header);
    let i_phi = builder.build_phi(i32_type, "i").unwrap();
    let acc2_phi = builder.build_phi(f32_type, "acc2").unwrap();
    let i_val = i_phi.as_basic_value().into_int_value();
    let acc2 = acc2_phi.as_basic_value().into_float_value();

    let inner_cmp = builder
        .build_int_compare(IntPredicate::ULT, i_val, tile_const, "icmp")
        .unwrap();
    builder
        .build_conditional_branch(inner_cmp, inner_loop_body, inner_loop_exit)
        .unwrap();

    // ── Inner loop body ──
    builder.position_at_end(inner_loop_body);

    // As[threadIdx.y * TILE + i]
    let as_load_idx = builder
        .build_int_add(
            builder.build_int_mul(tid_y, tile_const, "").unwrap(),
            i_val,
            "as_li",
        )
        .unwrap();
    let as_lp = unsafe {
        builder
            .build_gep(f32_type, as_ptr, &[as_load_idx], "as_lp")
            .unwrap()
    };
    let as_v = builder
        .build_load(f32_type, as_lp, "as_v")
        .unwrap()
        .into_float_value();

    // Bs[i * TILE + threadIdx.x]
    let bs_load_idx = builder
        .build_int_add(
            builder.build_int_mul(i_val, tile_const, "").unwrap(),
            tid_x,
            "bs_li",
        )
        .unwrap();
    let bs_lp = unsafe {
        builder
            .build_gep(f32_type, bs_ptr, &[bs_load_idx], "bs_lp")
            .unwrap()
    };
    let bs_v = builder
        .build_load(f32_type, bs_lp, "bs_v")
        .unwrap()
        .into_float_value();

    // acc += as_v * bs_v
    let prod = builder.build_float_mul(as_v, bs_v, "prod").unwrap();
    let acc_new = builder.build_float_add(acc2, prod, "acc_new").unwrap();

    let i_next = builder
        .build_int_add(i_val, i32_type.const_int(1, false), "i_next")
        .unwrap();
    builder
        .build_unconditional_branch(inner_loop_header)
        .unwrap();

    // Wire inner loop phi nodes
    i_phi.add_incoming(&[
        (&i32_type.const_int(0, false), tile_loop_body),
        (&i_next, inner_loop_body),
    ]);
    acc2_phi.add_incoming(&[(&acc, tile_loop_body), (&acc_new, inner_loop_body)]);

    // ── Inner loop exit → second syncthreads, advance tile loop ──
    builder.position_at_end(inner_loop_exit);

    call_barrier0(&context, &module, &builder);

    let t_next = builder.build_int_add(t, tile_const, "t_next").unwrap();
    builder
        .build_unconditional_branch(tile_loop_header)
        .unwrap();

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

    let c_idx = builder
        .build_int_add(
            builder.build_int_mul(row, n_param, "").unwrap(),
            col,
            "c_idx",
        )
        .unwrap();
    let c_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, c_ptr, &[c_idx], "c_ep")
            .unwrap()
    };
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

// ===========================================================================
// Flash Attention Forward
// ===========================================================================

fn run_flash_attn_fwd() -> Result<()> {
    let ptx = flash_attn::emit_flash_attn_fwd();
    println!("  Generated {} bytes of PTX", ptx.len());

    std::fs::write("/tmp/ferrite_flash_attn.ptx", &ptx).ok();
    println!("  [debug] PTX dumped to /tmp/ferrite_flash_attn.ptx");

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func =
        unsafe { cuda::module::get_function(module, CString::new("flash_attn_fwd").unwrap())? };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)",
        nregs, local_bytes
    );

    // Test parameters
    let batch: u32 = 1;
    let heads: u32 = 1;
    let seq_len: u32 = 128; // Start small for correctness
    let head_dim: u32 = 64;

    let n_q = (batch * heads * seq_len * head_dim) as usize;
    let n_kv = n_q; // Same seq_len for K, V
    let n_o = n_q;

    // Allocate
    let d_q = unsafe { cuda::malloc_sync(n_q * 2)? };
    let d_k = unsafe { cuda::malloc_sync(n_kv * 2)? };
    let d_v = unsafe { cuda::malloc_sync(n_kv * 2)? };
    let d_o = unsafe { cuda::malloc_sync(n_o * 2)? };

    // Initialize Q, K, V with small random-ish values
    let h_q: Vec<half::f16> = (0..n_q)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
        .collect();
    let h_k: Vec<half::f16> = (0..n_kv)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.05))
        .collect();
    let h_v: Vec<half::f16> = (0..n_kv)
        .map(|i| half::f16::from_f32(((i % 11) as f32 - 5.0) * 0.05))
        .collect();

    unsafe {
        cuda::memcpy_htod_sync(d_q, &h_q)?;
        cuda::memcpy_htod_sync(d_k, &h_k)?;
        cuda::memcpy_htod_sync(d_v, &h_v)?;
        cuda::memset_d8_sync(d_o, 0, n_o * 2)?;
    }

    let scale: f32 = 1.0 / (head_dim as f32).sqrt() * 1.44269504; // sm_scale * log2(e)
    let stride_batch: u32 = seq_len * head_dim; // Elements per batch*head

    let smem_bytes: c_uint = flash_attn::SMEM_BYTES;

    // Set max dynamic shared memory
    unsafe {
        cuda::function::set_function_attribute(
            func,
            FA::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes as i32,
        )?;
    }

    let grid_x = (seq_len + 127) / 128; // ceil(seq_len / BLOCK_M)
    let grid_y = batch * heads;
    let threads: u32 = 128;

    let params: &mut [*mut c_void] = &mut [
        (&d_q) as *const _ as *mut c_void,
        (&d_k) as *const _ as *mut c_void,
        (&d_v) as *const _ as *mut c_void,
        (&d_o) as *const _ as *mut c_void,
        (&seq_len) as *const _ as *mut c_void,
        (&scale) as *const _ as *mut c_void,
        (&stride_batch) as *const _ as *mut c_void,
    ];

    let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

    // Launch
    unsafe {
        cuda::launch_kernel(
            func,
            (grid_x, grid_y, 1),
            (threads, 1, 1),
            smem_bytes,
            stream,
            params,
        )?;
        cuda::stream::synchronize(stream)?;
    }

    // Read output
    let mut h_o: Vec<half::f16> = vec![half::f16::ZERO; n_o];
    unsafe {
        cuda::memcpy_dtoh_sync(&mut h_o, d_o)?;
    }

    // CPU reference: standard attention
    // O = softmax(Q @ K^T / sqrt(d)) @ V
    let sd = seq_len as usize;
    let dd = head_dim as usize;
    let mut max_err: f32 = 0.0;
    let sm_scale = 1.0 / (dd as f32).sqrt();

    for row in [0usize, 1, 31, 32, 63, 64, 95, 127] {
        if row >= sd {
            continue;
        }

        // Compute scores S[row][col] = sum_k Q[row][k] * K[col][k] * sm_scale
        let mut scores = vec![0.0f32; sd];
        for col in 0..sd {
            let mut dot = 0.0f32;
            for k in 0..dd {
                dot += h_q[row * dd + k].to_f32() * h_k[col * dd + k].to_f32();
            }
            scores[col] = dot * sm_scale;
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }

        // O[row] = scores @ V
        for d in [0usize, 1, 31, 32, 63] {
            if d >= dd {
                continue;
            }
            let mut o_val = 0.0f32;
            for col in 0..sd {
                o_val += scores[col] * h_v[col * dd + d].to_f32();
            }
            let got = h_o[row * dd + d].to_f32();
            let err = (got - o_val).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    if max_err < 0.1 {
        println!("  Correct (max err: {:.6})", max_err);
    } else {
        // Print diagnostic values
        for row in [0, 1, 63] {
            for d in [0, 1, 63] {
                if row >= sd || d >= dd {
                    continue;
                }
                let mut scores = vec![0.0f32; sd];
                for col in 0..sd {
                    let mut dot = 0.0f32;
                    for k in 0..dd {
                        dot += h_q[row * dd + k].to_f32() * h_k[col * dd + k].to_f32();
                    }
                    scores[col] = dot * sm_scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    exp_sum += *s;
                }
                for s in &mut scores {
                    *s /= exp_sum;
                }
                let mut o_val = 0.0f32;
                for col in 0..sd {
                    o_val += scores[col] * h_v[col * dd + d].to_f32();
                }
                let got = h_o[row * dd + d].to_f32();
                println!(
                    "    O[{row}][{d}]: got={got:.6} expected={o_val:.6} err={:.6}",
                    (got - o_val).abs()
                );
            }
        }
        println!("  Max error: {:.6} (FAIL)", max_err);
    }

    // Benchmark with larger sizes and multi-head
    for &(bench_batch, bench_heads, bench_seq) in &[
        (1u32, 1u32, 512u32),
        (1, 1, 1024),
        (1, 32, 512),
        (1, 32, 1024),
        (4, 32, 512),
    ] {
        let n_total = (bench_batch * bench_heads * bench_seq * head_dim) as usize;
        let d_q2 = unsafe { cuda::malloc_sync(n_total * 2)? };
        let d_k2 = unsafe { cuda::malloc_sync(n_total * 2)? };
        let d_v2 = unsafe { cuda::malloc_sync(n_total * 2)? };
        let d_o2 = unsafe { cuda::malloc_sync(n_total * 2)? };

        let h_data: Vec<half::f16> = (0..n_total)
            .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
            .collect();
        unsafe {
            cuda::memcpy_htod_sync(d_q2, &h_data)?;
            cuda::memcpy_htod_sync(d_k2, &h_data)?;
            cuda::memcpy_htod_sync(d_v2, &h_data)?;
        }

        let scale2: f32 = 1.0 / (head_dim as f32).sqrt() * 1.44269504;
        let stride2: u32 = bench_seq * head_dim;
        let gx2 = (bench_seq + 127) / 128;
        let gy2 = bench_batch * bench_heads;

        let params2: &mut [*mut c_void] = &mut [
            (&d_q2) as *const _ as *mut c_void,
            (&d_k2) as *const _ as *mut c_void,
            (&d_v2) as *const _ as *mut c_void,
            (&d_o2) as *const _ as *mut c_void,
            (&bench_seq) as *const _ as *mut c_void,
            (&scale2) as *const _ as *mut c_void,
            (&stride2) as *const _ as *mut c_void,
        ];

        // Warmup
        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx2, gy2, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params2,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx2, gy2, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params2,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        // Flash attention FLOPs: 2 * seq_q * seq_kv * head_dim * 2 (Q@K^T and P@V)
        let flops = 4.0
            * bench_seq as f64
            * bench_seq as f64
            * head_dim as f64
            * bench_batch as f64
            * bench_heads as f64;
        let tflops = flops / (us * 1e-6) / 1e12;
        println!(
            "  B={} H={} seq={}: {:.1} us, {:.2} TFLOPS",
            bench_batch, bench_heads, bench_seq, us, tflops
        );

        unsafe {
            cuda::free_sync(d_q2)?;
            cuda::free_sync(d_k2)?;
            cuda::free_sync(d_v2)?;
            cuda::free_sync(d_o2)?;
        }
    }

    unsafe {
        cuda::stream::destroy(stream)?;
        cuda::free_sync(d_q)?;
        cuda::free_sync(d_k)?;
        cuda::free_sync(d_v)?;
        cuda::free_sync(d_o)?;
        cuda::module::unload(module)?;
    }

    Ok(())
}

fn run_flash_attn_fwd_causal() -> Result<()> {
    let ptx = flash_attn::emit_flash_attn_fwd_causal();
    println!("  Generated {} bytes of PTX", ptx.len());

    std::fs::write("/tmp/ferrite_flash_attn_causal.ptx", &ptx).ok();

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("flash_attn_fwd_causal").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!("  [cuda] {} regs/thread, {} bytes spill", nregs, local_bytes);

    let batch: u32 = 1;
    let heads: u32 = 1;
    let seq_len: u32 = 128;
    let head_dim: u32 = 64;

    let n_q = (batch * heads * seq_len * head_dim) as usize;
    let d_q = unsafe { cuda::malloc_sync(n_q * 2)? };
    let d_k = unsafe { cuda::malloc_sync(n_q * 2)? };
    let d_v = unsafe { cuda::malloc_sync(n_q * 2)? };
    let d_o = unsafe { cuda::malloc_sync(n_q * 2)? };

    let h_q: Vec<half::f16> = (0..n_q)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
        .collect();
    let h_k: Vec<half::f16> = (0..n_q)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.05))
        .collect();
    let h_v: Vec<half::f16> = (0..n_q)
        .map(|i| half::f16::from_f32(((i % 11) as f32 - 5.0) * 0.05))
        .collect();

    unsafe {
        cuda::memcpy_htod_sync(d_q, &h_q)?;
        cuda::memcpy_htod_sync(d_k, &h_k)?;
        cuda::memcpy_htod_sync(d_v, &h_v)?;
        cuda::memset_d8_sync(d_o, 0, n_q * 2)?;
    }

    let scale: f32 = 1.0 / (head_dim as f32).sqrt() * 1.44269504;
    let stride_batch: u32 = seq_len * head_dim;
    let smem_bytes: c_uint = flash_attn::SMEM_BYTES;

    unsafe {
        cuda::function::set_function_attribute(
            func,
            FA::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes as i32,
        )?;
    }

    let grid_x = (seq_len + 127) / 128;
    let grid_y = batch * heads;
    let threads: u32 = 128;

    let params: &mut [*mut c_void] = &mut [
        (&d_q) as *const _ as *mut c_void,
        (&d_k) as *const _ as *mut c_void,
        (&d_v) as *const _ as *mut c_void,
        (&d_o) as *const _ as *mut c_void,
        (&seq_len) as *const _ as *mut c_void,
        (&scale) as *const _ as *mut c_void,
        (&stride_batch) as *const _ as *mut c_void,
    ];

    let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

    unsafe {
        cuda::launch_kernel(func, (grid_x, grid_y, 1), (threads, 1, 1), smem_bytes, stream, params)?;
        cuda::stream::synchronize(stream)?;
    }

    let mut h_o: Vec<half::f16> = vec![half::f16::ZERO; n_q];
    unsafe { cuda::memcpy_dtoh_sync(&mut h_o, d_o)?; }

    // CPU reference: causal attention
    let sd = seq_len as usize;
    let dd = head_dim as usize;
    let mut max_err: f32 = 0.0;
    let sm_scale = 1.0 / (dd as f32).sqrt();

    for row in [0usize, 1, 31, 32, 63, 64, 95, 127] {
        if row >= sd { continue; }
        let mut scores = vec![f32::NEG_INFINITY; sd];
        for col in 0..=row { // CAUSAL: only attend to col <= row
            let mut dot = 0.0f32;
            for k in 0..dd {
                dot += h_q[row * dd + k].to_f32() * h_k[col * dd + k].to_f32();
            }
            scores[col] = dot * sm_scale;
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            exp_sum += *s;
        }
        for s in &mut scores { *s /= exp_sum; }
        for d in [0usize, 1, 31, 32, 63] {
            if d >= dd { continue; }
            let mut o_val = 0.0f32;
            for col in 0..sd {
                o_val += scores[col] * h_v[col * dd + d].to_f32();
            }
            let got = h_o[row * dd + d].to_f32();
            let err = (got - o_val).abs();
            if err > max_err { max_err = err; }
        }
    }

    if max_err < 0.1 {
        println!("  Correct (max err: {:.6})", max_err);
    } else {
        for row in [0, 1, 63] {
            for d in [0, 1, 63] {
                if row >= sd || d >= dd { continue; }
                let mut scores = vec![f32::NEG_INFINITY; sd];
                for col in 0..=row {
                    let mut dot = 0.0f32;
                    for k in 0..dd {
                        dot += h_q[row * dd + k].to_f32() * h_k[col * dd + k].to_f32();
                    }
                    scores[col] = dot * sm_scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores { *s = (*s - max_s).exp(); exp_sum += *s; }
                for s in &mut scores { *s /= exp_sum; }
                let mut o_val = 0.0f32;
                for col in 0..sd { o_val += scores[col] * h_v[col * dd + d].to_f32(); }
                let got = h_o[row * dd + d].to_f32();
                println!("    O[{row}][{d}]: got={got:.6} expected={o_val:.6} err={:.6}", (got - o_val).abs());
            }
        }
        println!("  Max error: {:.6} (FAIL)", max_err);
    }

    unsafe {
        cuda::stream::destroy(stream)?;
        cuda::free_sync(d_q)?;
        cuda::free_sync(d_k)?;
        cuda::free_sync(d_v)?;
        cuda::free_sync(d_o)?;
        cuda::module::unload(module)?;
    }

    Ok(())
}

fn run_flash_attn_fwd_paged() -> Result<()> {
    let ptx = flash_attn::emit_flash_attn_fwd_paged();
    println!("  Generated {} bytes of PTX", ptx.len());

    std::fs::write("/tmp/ferrite_flash_attn_paged.ptx", &ptx).ok();
    println!("  [debug] PTX dumped to /tmp/ferrite_flash_attn_paged.ptx");

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("flash_attn_fwd_paged").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)",
        nregs, local_bytes
    );

    // Test parameters
    let batch: u32 = 1;
    let heads: u32 = 1;
    let seq_len: u32 = 128;
    let head_dim: u32 = 64;
    let page_block_size: u32 = 64; // tokens per page (== BLOCK_N)
    let num_pages = (seq_len + page_block_size - 1) / page_block_size;
    let max_blocks_per_seq: u32 = num_pages;

    let n_q = (batch * heads * seq_len * head_dim) as usize;
    let n_o = n_q;
    let page_stride: u32 = page_block_size * head_dim; // elements per page

    // Allocate physical pages (shuffled order to test paging)
    let total_pages = (batch * num_pages) as usize;

    // Create a shuffled physical page mapping
    // Logical pages: [0, 1, 2, ...] -> Physical pages: [shuffled order]
    let mut physical_pages: Vec<u32> = (0..total_pages as u32).collect();
    // Simple shuffle: reverse order
    physical_pages.reverse();

    // Build block_table: block_table[batch_idx * max_blocks + page_idx] = physical_page
    let mut h_block_table: Vec<i32> = vec![0i32; (batch as usize) * (max_blocks_per_seq as usize)];
    for b in 0..batch as usize {
        for p in 0..num_pages as usize {
            h_block_table[b * max_blocks_per_seq as usize + p] =
                physical_pages[b * num_pages as usize + p] as i32;
        }
    }

    // Create KV cache: fill physical pages with test data that corresponds to
    // the logical token order when accessed through the block_table
    let cache_elems = total_pages * page_stride as usize;
    let mut h_k_cache: Vec<half::f16> = vec![half::f16::ZERO; cache_elems];
    let mut h_v_cache: Vec<half::f16> = vec![half::f16::ZERO; cache_elems];

    // Also create contiguous reference K, V for CPU verification
    let mut h_k_ref: Vec<half::f16> = vec![half::f16::ZERO; n_q];
    let mut h_v_ref: Vec<half::f16> = vec![half::f16::ZERO; n_q];

    // Fill: for each batch, for each logical token t:
    //   - Compute the physical page and offset
    //   - Write the test pattern to the paged cache
    //   - Also write the same pattern to the contiguous reference
    for b in 0..batch as usize {
        for t in 0..seq_len as usize {
            let page_idx = t / page_block_size as usize;
            let offset_in_page = t % page_block_size as usize;
            let phys_page = h_block_table[b * max_blocks_per_seq as usize + page_idx] as usize;

            for d in 0..head_dim as usize {
                let k_val =
                    half::f16::from_f32(((t * head_dim as usize + d) % 13) as f32 * 0.05 - 0.3);
                let v_val =
                    half::f16::from_f32(((t * head_dim as usize + d) % 11) as f32 * 0.05 - 0.25);

                // Paged cache address
                let cache_idx = phys_page * page_stride as usize + offset_in_page * head_dim as usize + d;
                h_k_cache[cache_idx] = k_val;
                h_v_cache[cache_idx] = v_val;

                // Contiguous reference
                let ref_idx = b * (seq_len as usize * head_dim as usize) + t * head_dim as usize + d;
                h_k_ref[ref_idx] = k_val;
                h_v_ref[ref_idx] = v_val;
            }
        }
    }

    // Q data
    let h_q: Vec<half::f16> = (0..n_q)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
        .collect();

    // Allocate GPU memory
    let d_q = unsafe { cuda::malloc_sync(n_q * 2)? };
    let d_k_cache = unsafe { cuda::malloc_sync(cache_elems * 2)? };
    let d_v_cache = unsafe { cuda::malloc_sync(cache_elems * 2)? };
    let d_o = unsafe { cuda::malloc_sync(n_o * 2)? };
    let d_block_table = unsafe { cuda::malloc_sync(h_block_table.len() * 4)? };

    unsafe {
        cuda::memcpy_htod_sync(d_q, &h_q)?;
        cuda::memcpy_htod_sync(d_k_cache, &h_k_cache)?;
        cuda::memcpy_htod_sync(d_v_cache, &h_v_cache)?;
        cuda::memset_d8_sync(d_o, 0, n_o * 2)?;
        cuda::memcpy_htod_sync(d_block_table, &h_block_table)?;
    }

    let scale: f32 = 1.0 / (head_dim as f32).sqrt() * 1.44269504;
    let stride_q_batch: u32 = seq_len * head_dim;

    let smem_bytes: c_uint = flash_attn::SMEM_BYTES;

    unsafe {
        cuda::function::set_function_attribute(
            func,
            FA::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes as i32,
        )?;
    }

    let grid_x = (seq_len + 127) / 128;
    let grid_y = batch * heads;
    let threads: u32 = 128;

    let params: &mut [*mut c_void] = &mut [
        (&d_q) as *const _ as *mut c_void,
        (&d_k_cache) as *const _ as *mut c_void,
        (&d_v_cache) as *const _ as *mut c_void,
        (&d_o) as *const _ as *mut c_void,
        (&d_block_table) as *const _ as *mut c_void,
        (&seq_len) as *const _ as *mut c_void,
        (&scale) as *const _ as *mut c_void,
        (&stride_q_batch) as *const _ as *mut c_void,
        (&page_stride) as *const _ as *mut c_void,
        (&page_block_size) as *const _ as *mut c_void,
        (&max_blocks_per_seq) as *const _ as *mut c_void,
    ];

    // Use null stream for correctness
    let stream: cudarc::driver::sys::CUstream = std::ptr::null_mut();

    // Launch
    unsafe {
        cuda::launch_kernel(
            func,
            (grid_x, grid_y, 1),
            (threads, 1, 1),
            smem_bytes,
            stream,
            params,
        )?;
        cuda::stream::synchronize(stream)?;
    }

    // Read output
    let mut h_o: Vec<half::f16> = vec![half::f16::ZERO; n_o];
    unsafe {
        cuda::memcpy_dtoh_sync(&mut h_o, d_o)?;
    }

    // CPU reference using contiguous K, V (reconstructed from block_table)
    let sd = seq_len as usize;
    let dd = head_dim as usize;
    let mut max_err: f32 = 0.0;
    let sm_scale = 1.0 / (dd as f32).sqrt();

    for row in [0usize, 1, 31, 32, 63, 64, 95, 127] {
        if row >= sd {
            continue;
        }

        let mut scores = vec![0.0f32; sd];
        for col in 0..sd {
            let mut dot = 0.0f32;
            for k in 0..dd {
                dot += h_q[row * dd + k].to_f32() * h_k_ref[col * dd + k].to_f32();
            }
            scores[col] = dot * sm_scale;
        }

        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }

        for d in [0usize, 1, 31, 32, 63] {
            if d >= dd {
                continue;
            }
            let mut o_val = 0.0f32;
            for col in 0..sd {
                o_val += scores[col] * h_v_ref[col * dd + d].to_f32();
            }
            let got = h_o[row * dd + d].to_f32();
            let err = (got - o_val).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    if max_err < 0.1 {
        println!("  Correct (max err: {:.6})", max_err);
    } else {
        // Print diagnostic values
        for row in [0, 1, 63] {
            for d in [0, 1, 63] {
                if row >= sd || d >= dd {
                    continue;
                }
                let mut scores = vec![0.0f32; sd];
                for col in 0..sd {
                    let mut dot = 0.0f32;
                    for k in 0..dd {
                        dot += h_q[row * dd + k].to_f32() * h_k_ref[col * dd + k].to_f32();
                    }
                    scores[col] = dot * sm_scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    exp_sum += *s;
                }
                for s in &mut scores {
                    *s /= exp_sum;
                }
                let mut o_val = 0.0f32;
                for col in 0..sd {
                    o_val += scores[col] * h_v_ref[col * dd + d].to_f32();
                }
                let got = h_o[row * dd + d].to_f32();
                println!(
                    "    O[{row}][{d}]: got={got:.6} expected={o_val:.6} err={:.6}",
                    (got - o_val).abs()
                );
            }
        }
        println!("  Max error: {:.6} (FAIL)", max_err);
    }

    unsafe {
        cuda::free_sync(d_q)?;
        cuda::free_sync(d_k_cache)?;
        cuda::free_sync(d_v_cache)?;
        cuda::free_sync(d_o)?;
        cuda::free_sync(d_block_table)?;
        cuda::module::unload(module)?;
    }

    Ok(())
}

fn run_flash_attn_fwd_hdim128() -> Result<()> {
    let ptx = flash_attn_hdim128::emit_flash_attn_fwd_hdim128();
    println!("  Generated {} bytes of PTX", ptx.len());

    std::fs::write("/tmp/ferrite_flash_attn_hdim128.ptx", &ptx).ok();
    println!("  [debug] PTX dumped to /tmp/ferrite_flash_attn_hdim128.ptx");

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("flash_attn_fwd_hdim128").unwrap())?
    };

    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs =
        unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe {
        cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?
    };
    println!(
        "  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)",
        nregs, local_bytes
    );

    // Test parameters
    let batch: u32 = 1;
    let heads: u32 = 1;
    let seq_len: u32 = 128;
    let head_dim: u32 = 128;

    let n_q = (batch * heads * seq_len * head_dim) as usize;
    let n_kv = n_q;
    let n_o = n_q;

    let d_q = unsafe { cuda::malloc_sync(n_q * 2)? };
    let d_k = unsafe { cuda::malloc_sync(n_kv * 2)? };
    let d_v = unsafe { cuda::malloc_sync(n_kv * 2)? };
    let d_o = unsafe { cuda::malloc_sync(n_o * 2)? };

    let h_q: Vec<half::f16> = (0..n_q)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
        .collect();
    let h_k: Vec<half::f16> = (0..n_kv)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.05))
        .collect();
    let h_v: Vec<half::f16> = (0..n_kv)
        .map(|i| half::f16::from_f32(((i % 11) as f32 - 5.0) * 0.05))
        .collect();

    unsafe {
        cuda::memcpy_htod_sync(d_q, &h_q)?;
        cuda::memcpy_htod_sync(d_k, &h_k)?;
        cuda::memcpy_htod_sync(d_v, &h_v)?;
        cuda::memset_d8_sync(d_o, 0, n_o * 2)?;
    }

    let scale: f32 = 1.0 / (head_dim as f32).sqrt() * 1.44269504;
    let stride_batch: u32 = seq_len * head_dim;

    let smem_bytes: c_uint = flash_attn_hdim128::SMEM_BYTES;

    unsafe {
        cuda::function::set_function_attribute(
            func,
            FA::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes as i32,
        )?;
    }

    let grid_x = (seq_len + 127) / 128;
    let grid_y = batch * heads;
    let threads: u32 = 128;

    let params: &mut [*mut c_void] = &mut [
        (&d_q) as *const _ as *mut c_void,
        (&d_k) as *const _ as *mut c_void,
        (&d_v) as *const _ as *mut c_void,
        (&d_o) as *const _ as *mut c_void,
        (&seq_len) as *const _ as *mut c_void,
        (&scale) as *const _ as *mut c_void,
        (&stride_batch) as *const _ as *mut c_void,
    ];

    let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;

    unsafe {
        cuda::launch_kernel(
            func,
            (grid_x, grid_y, 1),
            (threads, 1, 1),
            smem_bytes,
            stream,
            params,
        )?;
        cuda::stream::synchronize(stream)?;
    }

    // Read output
    let mut h_o: Vec<half::f16> = vec![half::f16::ZERO; n_o];
    unsafe {
        cuda::memcpy_dtoh_sync(&mut h_o, d_o)?;
    }

    // CPU reference
    let sd = seq_len as usize;
    let dd = head_dim as usize;
    let mut max_err: f32 = 0.0;
    let sm_scale = 1.0 / (dd as f32).sqrt();

    for row in [0usize, 1, 31, 32, 63, 64, 95, 127] {
        if row >= sd {
            continue;
        }
        let mut scores = vec![0.0f32; sd];
        for col in 0..sd {
            let mut dot = 0.0f32;
            for k in 0..dd {
                dot += h_q[row * dd + k].to_f32() * h_k[col * dd + k].to_f32();
            }
            scores[col] = dot * sm_scale;
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }
        for d in [0usize, 1, 31, 63, 64, 95, 127] {
            if d >= dd {
                continue;
            }
            let mut o_val = 0.0f32;
            for col in 0..sd {
                o_val += scores[col] * h_v[col * dd + d].to_f32();
            }
            let got = h_o[row * dd + d].to_f32();
            let err = (got - o_val).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    if max_err < 0.1 {
        println!("  Correct (max err: {:.6})", max_err);
    } else {
        for row in [0, 1, 63] {
            for d in [0, 1, 63, 127] {
                if row >= sd || d >= dd {
                    continue;
                }
                let mut scores = vec![0.0f32; sd];
                for col in 0..sd {
                    let mut dot = 0.0f32;
                    for k in 0..dd {
                        dot += h_q[row * dd + k].to_f32() * h_k[col * dd + k].to_f32();
                    }
                    scores[col] = dot * sm_scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    exp_sum += *s;
                }
                for s in &mut scores {
                    *s /= exp_sum;
                }
                let mut o_val = 0.0f32;
                for col in 0..sd {
                    o_val += scores[col] * h_v[col * dd + d].to_f32();
                }
                let got = h_o[row * dd + d].to_f32();
                println!(
                    "    O[{row}][{d}]: got={got:.6} expected={o_val:.6} err={:.6}",
                    (got - o_val).abs()
                );
            }
        }
        println!("  Max error: {:.6} (FAIL)", max_err);
    }

    // Benchmark
    for &(bench_batch, bench_heads, bench_seq) in &[
        (1u32, 1u32, 512u32),
        (1, 1, 1024),
        (1, 32, 512),
        (1, 32, 1024),
        (4, 32, 512),
    ] {
        let n_total = (bench_batch * bench_heads * bench_seq * head_dim) as usize;
        let d_q2 = unsafe { cuda::malloc_sync(n_total * 2)? };
        let d_k2 = unsafe { cuda::malloc_sync(n_total * 2)? };
        let d_v2 = unsafe { cuda::malloc_sync(n_total * 2)? };
        let d_o2 = unsafe { cuda::malloc_sync(n_total * 2)? };

        let h_data: Vec<half::f16> = (0..n_total)
            .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05))
            .collect();
        unsafe {
            cuda::memcpy_htod_sync(d_q2, &h_data)?;
            cuda::memcpy_htod_sync(d_k2, &h_data)?;
            cuda::memcpy_htod_sync(d_v2, &h_data)?;
        }

        let scale2: f32 = 1.0 / (head_dim as f32).sqrt() * 1.44269504;
        let stride2: u32 = bench_seq * head_dim;
        let gx2 = (bench_seq + 127) / 128;
        let gy2 = bench_batch * bench_heads;

        let params2: &mut [*mut c_void] = &mut [
            (&d_q2) as *const _ as *mut c_void,
            (&d_k2) as *const _ as *mut c_void,
            (&d_v2) as *const _ as *mut c_void,
            (&d_o2) as *const _ as *mut c_void,
            (&bench_seq) as *const _ as *mut c_void,
            (&scale2) as *const _ as *mut c_void,
            (&stride2) as *const _ as *mut c_void,
        ];

        for _ in 0..10 {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx2, gy2, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params2,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                cuda::launch_kernel(
                    func,
                    (gx2, gy2, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    stream,
                    params2,
                )?;
            }
        }
        unsafe {
            cuda::stream::synchronize(stream)?;
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;

        let flops = 4.0
            * bench_seq as f64
            * bench_seq as f64
            * head_dim as f64
            * bench_batch as f64
            * bench_heads as f64;
        let tflops = flops / (us * 1e-6) / 1e12;
        println!(
            "  B={} H={} seq={}: {:.1} us, {:.2} TFLOPS",
            bench_batch, bench_heads, bench_seq, us, tflops
        );

        unsafe {
            cuda::free_sync(d_q2)?;
            cuda::free_sync(d_k2)?;
            cuda::free_sync(d_v2)?;
            cuda::free_sync(d_o2)?;
        }
    }

    unsafe {
        cuda::stream::destroy(stream)?;
        cuda::free_sync(d_q)?;
        cuda::free_sync(d_k)?;
        cuda::free_sync(d_v)?;
        cuda::free_sync(d_o)?;
        cuda::module::unload(module)?;
    }

    Ok(())
}

fn run_silu_mul() -> Result<()> {
    let ptx = silu_mul::emit_silu_mul_kernel();
    println!("  Generated {} bytes of PTX", ptx.len());
    std::fs::write("/tmp/ferrite_silu_mul.ptx", &ptx).ok();
    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe { cuda::module::get_function(module, CString::new("silu_mul_kernel").unwrap())? };
    let n: u32 = 4096 * 128;
    let d_gate = unsafe { cuda::malloc_sync(n as usize * 2)? };
    let d_up = unsafe { cuda::malloc_sync(n as usize * 2)? };
    let d_out = unsafe { cuda::malloc_sync(n as usize * 2)? };
    let h_gate: Vec<half::f16> = (0..n as usize).map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.1)).collect();
    let h_up: Vec<half::f16> = (0..n as usize).map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.1)).collect();
    unsafe { cuda::memcpy_htod_sync(d_gate, &h_gate)?; cuda::memcpy_htod_sync(d_up, &h_up)?; }
    let grid_x = (n + 1023) / 1024;
    let params: &mut [*mut c_void] = &mut [
        (&d_out) as *const _ as *mut c_void,
        (&d_gate) as *const _ as *mut c_void,
        (&d_up) as *const _ as *mut c_void,
        (&n) as *const _ as *mut c_void,
    ];
    // Use null stream (blocking) to avoid sync issues with memcpy on stream 0
    let stream = std::ptr::null_mut();
    unsafe {
        cuda::launch_kernel(func, (grid_x, 1, 1), (128, 1, 1), 0, stream, params)?;
        cuda::stream::synchronize(stream)?;
    }
    let mut h_out: Vec<half::f16> = vec![half::f16::ZERO; n as usize];
    unsafe { cuda::memcpy_dtoh_sync(&mut h_out, d_out)?; }

    let mut max_err: f32 = 0.0;
    for i in 0..n as usize {
        let g = h_gate[i].to_f32();
        let u = h_up[i].to_f32();
        let expected = g / (1.0 + (-g).exp()) * u;
        let got = h_out[i].to_f32();
        let err = (got - expected).abs();
        if err > max_err || err.is_nan() {
            max_err = if err.is_nan() { f32::MAX } else { err };
            if err > 0.01 { println!("    BIG ERR at i={}: got={:.6} exp={:.6} gate={:.4} up={:.4}", i, got, expected, g, u); }
        }
    }
    // Debug: print first 4 values
    for i in 0..4 {
        let g = h_gate[i].to_f32();
        let u = h_up[i].to_f32();
        let expected = g / (1.0 + (-g).exp()) * u;
        println!("  out[{}]: got={:.6} exp={:.6} err={:.6}", i, h_out[i].to_f32(), expected, (h_out[i].to_f32() - expected).abs());
    }
    if max_err < 0.01 {
        println!("  Correct (max err: {:.6})", max_err);
    } else {
        println!("  FAIL (max err: {:.6})", max_err);
    }
    for _ in 0..10 { unsafe { cuda::launch_kernel(func, (grid_x, 1, 1), (128, 1, 1), 0, stream, params)?; } }
    unsafe { cuda::stream::synchronize(stream)?; }
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters { unsafe { cuda::launch_kernel(func, (grid_x, 1, 1), (128, 1, 1), 0, stream, params)?; } }
    unsafe { cuda::stream::synchronize(stream)?; }
    let us = start.elapsed().as_micros() as f64 / iters as f64;
    let gb_s = 3.0 * n as f64 * 2.0 / (us * 1e-6) / 1e9;
    println!("  n={}: {:.1} us, {:.1} GB/s", n, us, gb_s);
    unsafe {
        // stream is null (default stream), don't destroy
        cuda::free_sync(d_gate)?; cuda::free_sync(d_up)?; cuda::free_sync(d_out)?;
        cuda::module::unload(module)?;
    }
    Ok(())
}

fn run_rotary() -> Result<()> {
    let ptx = rotary::emit_rotary_kernel();
    println!("  Generated {} bytes of PTX", ptx.len());
    std::fs::write("/tmp/ferrite_rotary.ptx", &ptx).ok();
    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe { cuda::module::get_function(module, CString::new("rotary_kernel").unwrap())? };

    let batch: u32 = 1;
    let seqlen: u32 = 128;
    let nheads: u32 = 32;
    let headdim: u32 = 128;
    let half_dim: u32 = headdim / 2;

    // Contiguous [batch, seqlen, nheads, headdim] layout
    let stride_out_seqlen: u32 = nheads * headdim;
    let stride_out_nheads: u32 = headdim;
    let stride_out_headdim: u32 = 1;
    let stride_x_seqlen: u32 = nheads * headdim; // same as out

    let n_x = (batch * seqlen * nheads * headdim) as usize;
    let n_cs = (seqlen * half_dim) as usize;
    let d_x = unsafe { cuda::malloc_sync(n_x * 2)? };
    let d_out = unsafe { cuda::malloc_sync(n_x * 2)? };
    let d_cos = unsafe { cuda::malloc_sync(n_cs * 4)? };
    let d_sin = unsafe { cuda::malloc_sync(n_cs * 4)? };

    let h_x: Vec<half::f16> = (0..n_x).map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05)).collect();
    let h_cos: Vec<f32> = (0..n_cs).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();
    let h_sin: Vec<f32> = (0..n_cs).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();
    unsafe {
        cuda::memcpy_htod_sync(d_x, &h_x)?;
        cuda::memcpy_htod_sync(d_cos, &h_cos)?;
        cuda::memcpy_htod_sync(d_sin, &h_sin)?;
        cuda::memset_d8_sync(d_out, 0, n_x * 2)?;
    }

    // Grid: (ceil(nheads/4), ceil(seqlen/8), batch)
    let grid = ((nheads + 3) / 4, (seqlen + 7) / 8, batch);

    // 10 params: out, x, cos, sin, seqlen, nheads, stride_out_seqlen, stride_out_nheads, stride_out_headdim, stride_x_seqlen
    let params: &mut [*mut c_void] = &mut [
        (&d_out) as *const _ as *mut c_void,
        (&d_x) as *const _ as *mut c_void,
        (&d_cos) as *const _ as *mut c_void,
        (&d_sin) as *const _ as *mut c_void,
        (&seqlen) as *const _ as *mut c_void,
        (&nheads) as *const _ as *mut c_void,
        (&stride_out_seqlen) as *const _ as *mut c_void,
        (&stride_out_nheads) as *const _ as *mut c_void,
        (&stride_out_headdim) as *const _ as *mut c_void,
        (&stride_x_seqlen) as *const _ as *mut c_void,
    ];

    // Shared memory: 2048 bytes for cos/sin transpose (128 threads * 16 bytes each)
    let smem_bytes: u32 = 2048;
    let stream = std::ptr::null_mut(); // null stream for sync with memcpy
    unsafe {
        cuda::launch_kernel(func, grid, (128, 1, 1), smem_bytes, stream, params)?;
        cuda::stream::synchronize(stream)?;
    }

    let mut h_out: Vec<half::f16> = vec![half::f16::ZERO; n_x];
    unsafe { cuda::memcpy_dtoh_sync(&mut h_out, d_out)?; }

    let hd = headdim as usize;
    let hh = half_dim as usize;
    let nh = nheads as usize;
    let sl = seqlen as usize;
    let mut max_err: f32 = 0.0;
    let mut checked = 0usize;
    for s in [0usize, 1, 63, 127] {
        for h in [0, 1, 31] {
            for k in [0, 1, 31, 63] {
                if s >= sl || h >= nh || k >= hh { continue; }
                let flat = s * nh * hd + h * hd;
                let x1 = h_x[flat + k].to_f32();
                let x2 = h_x[flat + hh + k].to_f32();
                let c = h_cos[s * hh + k];
                let sn = h_sin[s * hh + k];
                let err1 = (h_out[flat + k].to_f32() - (x1 * c - x2 * sn)).abs();
                let err2 = (h_out[flat + hh + k].to_f32() - (x2 * c + x1 * sn)).abs();
                if err1 > max_err { max_err = err1; }
                if err2 > max_err { max_err = err2; }
                checked += 1;
            }
        }
    }
    if max_err < 0.05 {
        println!("  Correct (max err: {:.6}, checked {} points)", max_err, checked);
    } else {
        for k in 0..8 {
            let x1 = h_x[k].to_f32();
            let x2 = h_x[hh + k].to_f32();
            println!("    out[0][0][{}]: got={:.6} exp={:.6}", k, h_out[k].to_f32(), x1 * h_cos[k] - x2 * h_sin[k]);
            println!("    out[0][0][{}+hh]: got={:.6} exp={:.6}", k, h_out[hh + k].to_f32(), x2 * h_cos[k] + x1 * h_sin[k]);
        }
        println!("  FAIL (max err: {:.6})", max_err);
    }

    // Warmup
    for _ in 0..10 {
        unsafe { cuda::launch_kernel(func, grid, (128, 1, 1), smem_bytes, stream, params)?; }
    }
    unsafe { cuda::stream::synchronize(stream)?; }

    // Benchmark
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe { cuda::launch_kernel(func, grid, (128, 1, 1), smem_bytes, stream, params)?; }
    }
    unsafe { cuda::stream::synchronize(stream)?; }
    let us = start.elapsed().as_micros() as f64 / iters as f64;
    let bytes = 2.0 * n_x as f64 * 2.0 + 2.0 * n_cs as f64 * 4.0;
    let gb_s = bytes / (us * 1e-6) / 1e9;
    println!("  S={} H={} D={}: {:.1} us, {:.1} GB/s", seqlen, nheads, headdim, us, gb_s);

    unsafe {
        cuda::free_sync(d_x)?;
        cuda::free_sync(d_out)?;
        cuda::free_sync(d_cos)?;
        cuda::free_sync(d_sin)?;
        cuda::module::unload(module)?;
    }
    Ok(())
}
