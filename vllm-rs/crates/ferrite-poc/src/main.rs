// SPDX-License-Identifier: Apache-2.0
//
// Ferrite Phase 0 — Proof of Concept
//
// Validates the full pipeline: Rust → inkwell → LLVM IR → PTX → GPU kernel.
// Run on a machine with LLVM 20 and an NVIDIA GPU:
//
//   LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo run -p ferrite-poc --release

mod mma_gemm;
mod tiled_mma;
mod cubek_gemm;

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

    // Set -nvptx-short-ptr BEFORE initializing NVPTX target.
    // This makes addrspace(3) shared memory pointers 32-bit instead of 64-bit.
    // Critical for register usage — same trick Triton uses.
    unsafe {
        let args: [*const i8; 2] = [
            b"ferrite\0".as_ptr() as *const i8,
            b"-nvptx-short-ptr\0".as_ptr() as *const i8,
        ];
        llvm_sys::support::LLVMParseCommandLineOptions(
            2,
            args.as_ptr(),
            std::ptr::null(),
        );
    }

    Target::initialize_nvptx(&InitializationConfig::default());
    println!("[llvm] NVPTX target initialized (with -nvptx-short-ptr)");

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

    println!("\n[1/4] Vector add (1M f32 elements)");
    step1_vector_add(&sm)?;

    println!("\n[2/4] Tiled GEMM (M=N=K=1024, f32, shared memory, no tensor cores)");
    step2_tiled_gemm(&sm)?;

    println!("\n[3/4] MMA GEMM (M=N=K=1024, f16→f32, mma.sync tensor cores)");
    mma_gemm::step3_mma_gemm(&sm)?;

    println!("\n[3b/4] Multi-warp MMA GEMM (4 warps, 64×64 tile, register tiling)");
    tiled_mma::step3b_multiwarp_gemm(&sm)?;

    println!("\n[4/4] CubeK-style GEMM (128×128, K=32, 4 warps×32 MMAs, B128 swizzle)");
    cubek_gemm::step_cubek_gemm(&sm)?;

    println!("\n═══════════════════════════════════");
    println!("Phase 0 complete.");
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
            "+ptx83",
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
