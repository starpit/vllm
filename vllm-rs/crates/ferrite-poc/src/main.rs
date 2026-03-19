// SPDX-License-Identifier: Apache-2.0
//
// Ferrite Phase 0 — Proof of Concept
//
// Validates the full pipeline: Rust → inkwell → LLVM IR → PTX → GPU kernel.
// Run on a machine with LLVM 20 and an NVIDIA GPU:
//
//   LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo run -p ferrite-poc --release

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
use inkwell::values::{FunctionValue, IntValue};
use inkwell::{AddressSpace, OptimizationLevel, IntPredicate};

// ---------------------------------------------------------------------------
// CUDA (cudarc raw driver API)
// ---------------------------------------------------------------------------
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::result as cuda;

fn main() -> Result<()> {
    println!("Ferrite Phase 0 — Proof of Concept");
    println!("═══════════════════════════════════\n");

    Target::initialize_nvptx(&InitializationConfig::default());
    println!("[llvm] NVPTX target initialized");

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
// PTX generation via inkwell
// ===========================================================================

fn create_nvptx_target_machine(sm: &str) -> Result<TargetMachine> {
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

// ---------------------------------------------------------------------------
// NVPTX helpers
// ---------------------------------------------------------------------------

fn call_sreg<'ctx>(
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
    let call = builder.build_call(func, &[], name).unwrap();
    // try_as_basic_value returns Either<BasicValueEnum, InstructionValue> or similar.
    // We need the left (value) side.
    let val = call.try_as_basic_value();
    match val {
        either::Either::Left(v) => v.into_int_value(),
        _ => panic!("NVPTX sreg intrinsic {} returned void", intrinsic),
    }
}

/// NVVM metadata: !nvvm.annotations = !{!0}
///                !0 = !{ptr @func, !"kernel", i32 1}
///
/// Uses raw LLVM C API because inkwell's named metadata API varies by version.
fn add_nvvm_kernel_metadata<'ctx>(module: &Module<'ctx>, function: &FunctionValue<'ctx>) {
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
