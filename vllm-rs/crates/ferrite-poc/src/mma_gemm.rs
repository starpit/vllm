// SPDX-License-Identifier: Apache-2.0
//
// Step 3: MMA GEMM — tensor core mma.sync via inline PTX asm in LLVM IR.
//
// Proves: inline asm through inkwell/LLVM works, tensor cores can be driven
// from Rust-generated LLVM IR.
//
// Kernel: C(f32) = A(f16) × B(f16), using mma.sync.aligned.m16n8k16.row.col
// Block: 32 threads (1 warp), each block computes 16×8 of C
// Grid: (N/8, M/16)

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

use inkwell::context::Context as LlvmContext;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::targets::{TargetMachine, TargetTriple, FileType};
use inkwell::types::AsTypeRef;
use inkwell::values::{AsValueRef, BasicValueEnum, FunctionValue, IntValue, FloatValue};
use inkwell::{AddressSpace, IntPredicate};

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::result as cuda;

use crate::{create_nvptx_target_machine, add_nvvm_kernel_metadata, call_sreg, call_barrier0};

pub fn step3_mma_gemm(sm: &str) -> Result<()> {
    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;

    let ptx = emit_mma_gemm_ptx(sm)?;
    println!("  [llvm] Generated {} bytes of PTX", ptx.len());

    // Optionally dump PTX for debugging
    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_mma_gemm.ptx", &ptx)?;
        println!("  [debug] PTX written to /tmp/ferrite_mma_gemm.ptx");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null byte")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("mma_gemm").unwrap())?
    };
    println!("  [cuda] Module loaded, function resolved");

    // Allocate — A,B are f16 (2 bytes), C is f32 (4 bytes)
    let size_a = (m * k) as usize;
    let size_b = (k * n) as usize;
    let size_c = (m * n) as usize;

    let d_a = unsafe { cuda::malloc_sync(size_a * 2)? };
    let d_b = unsafe { cuda::malloc_sync(size_b * 2)? };
    let d_c = unsafe { cuda::malloc_sync(size_c * 4)? };

    // Host data (f16 stored as u16 for half crate)
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

    let grid_x: c_uint = n / 8;
    let grid_y: c_uint = m / 16;
    let smem_bytes: c_uint = (16 * 16 + 8 * 16) * 2; // A tile + B transposed tile, f16

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
            cuda::launch_kernel(func, (grid_x, grid_y, 1), (32, 1, 1), smem_bytes, stream, params)?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }

    // Benchmark
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            cuda::launch_kernel(func, (grid_x, grid_y, 1), (32, 1, 1), smem_bytes, stream, params)?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_micros() as f64 / iters as f64;

    // Verify
    unsafe { cuda::memcpy_dtoh_sync(&mut h_c, d_c)?; }

    let mut max_err: f32 = 0.0;
    for row in 0..m as usize {
        for col in 0..n as usize {
            let mut expected: f32 = 0.0;
            for kk in 0..k as usize {
                expected += h_a[row * k as usize + kk].to_f32() * h_b[kk * n as usize + col].to_f32();
            }
            let err = (h_c[row * n as usize + col] - expected).abs();
            if err > max_err { max_err = err; }
        }
    }

    // f16 has ~3 decimal digits of precision, so allow more error than f32
    if max_err < 1.0 {
        println!("  ✓ Correct! (max error: {:.4})", max_err);
    } else {
        println!("  ✗ Max error: {:.4} (expected < 1.0)", max_err);
        bail!("MMA GEMM verification failed");
    }

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / (us_per_launch * 1e-6) / 1e12;
    println!("  {:.1} μs, {:.2} TFLOPS (mma.sync tensor cores)", us_per_launch, tflops);

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
// PTX emission
// ===========================================================================

/// Emit a tensor-core GEMM kernel using mma.sync.aligned.m16n8k16.
///
/// Block: 32 threads (1 warp), computes 16×8 of C.
/// Shared memory: A tile [16×16 f16] + B transposed tile [8×16 f16].
/// B is transposed during load so both A and B fragments can be loaded as i32.
fn emit_mma_gemm_ptx(sm: &str) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;
    let context = LlvmContext::create();
    let module = context.create_module("ferrite_mma_gemm");
    let builder = context.create_builder();

    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let i32_type = context.i32_type();
    let f16_type = context.f16_type();
    let f32_type = context.f32_type();
    let void_type = context.void_type();
    let ptr_global = context.ptr_type(AddressSpace::from(1u16));
    let ptr_shared = context.ptr_type(AddressSpace::from(3u16));

    // Dynamic shared memory (extern __shared__)
    let smem_global = module.add_global(
        context.i8_type().array_type(0),
        Some(AddressSpace::from(3u16)),
        "smem",
    );
    smem_global.set_alignment(16);
    smem_global.set_externally_initialized(true);

    // Kernel: void mma_gemm(f16* A, f16* B, f32* C, u32 M, u32 N, u32 K)
    let fn_type = void_type.fn_type(
        &[
            ptr_global.into(), ptr_global.into(), ptr_global.into(),
            i32_type.into(), i32_type.into(), i32_type.into(),
        ],
        false,
    );
    let function = module.add_function("mma_gemm", fn_type, None);
    add_nvvm_kernel_metadata(&module, &function);

    // Constants
    let c0 = i32_type.const_int(0, false);
    let c1 = i32_type.const_int(1, false);
    let c2 = i32_type.const_int(2, false);
    let c4 = i32_type.const_int(4, false);
    let c8 = i32_type.const_int(8, false);
    let c16 = i32_type.const_int(16, false);
    let c32 = i32_type.const_int(32, false);
    let f32_zero = f32_type.const_float(0.0);

    // Basic blocks
    let entry = context.append_basic_block(function, "entry");
    let tile_header = context.append_basic_block(function, "tile_header");
    let tile_body = context.append_basic_block(function, "tile_body");
    let tile_exit = context.append_basic_block(function, "tile_exit");

    // ── Entry ──
    builder.position_at_end(entry);

    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let _m_param = function.get_nth_param(3).unwrap().into_int_value();
    let n_param = function.get_nth_param(4).unwrap().into_int_value();
    let k_param = function.get_nth_param(5).unwrap().into_int_value();

    let lane = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.tid.x", "lane");
    let bid_x = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ctaid.x", "bid_x");
    let bid_y = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ctaid.y", "bid_y");

    // block_row = bid_y * 16, block_col = bid_x * 8
    let block_row = builder.build_int_mul(bid_y, c16, "block_row").unwrap();
    let block_col = builder.build_int_mul(bid_x, c8, "block_col").unwrap();

    // group = lane / 4, tg = lane % 4 (for MMA fragment mapping)
    let group = builder.build_int_unsigned_div(lane, c4, "group").unwrap();
    let tg = builder.build_int_unsigned_rem(lane, c4, "tg").unwrap();

    // Shared memory pointers
    let smem_base = smem_global.as_pointer_value();
    // smem_a: [16×16 f16] at offset 0 (512 bytes)
    // smem_bt: [8×16 f16] at offset 512 (256 bytes) — B transposed
    let smem_a_base = builder.build_pointer_cast(smem_base, ptr_shared, "smem_a").unwrap();
    let smem_bt_off = unsafe {
        builder.build_gep(context.i8_type(), smem_base,
            &[i32_type.const_int(512, false)], "bt_off").unwrap()
    };
    let smem_bt_base = builder.build_pointer_cast(smem_bt_off, ptr_shared, "smem_bt").unwrap();

    builder.build_unconditional_branch(tile_header).unwrap();

    // ── Tile loop header ──
    builder.position_at_end(tile_header);
    let t_phi = builder.build_phi(i32_type, "t").unwrap();
    let acc0_phi = builder.build_phi(f32_type, "acc0").unwrap();
    let acc1_phi = builder.build_phi(f32_type, "acc1").unwrap();
    let acc2_phi = builder.build_phi(f32_type, "acc2").unwrap();
    let acc3_phi = builder.build_phi(f32_type, "acc3").unwrap();
    let t = t_phi.as_basic_value().into_int_value();

    let tile_cmp = builder.build_int_compare(IntPredicate::ULT, t, k_param, "tile_cmp").unwrap();
    builder.build_conditional_branch(tile_cmp, tile_body, tile_exit).unwrap();

    // ── Tile body ──
    builder.position_at_end(tile_body);

    // Load A tile [16×16 f16] into shared memory.
    // 32 threads, 256 elements → 8 elements per thread.
    // Thread `lane` loads elements lane*8 .. lane*8+7 (linear in tile).
    let lane_x8 = builder.build_int_mul(lane, c8, "lane_x8").unwrap();
    for j in 0..8u64 {
        let j_val = i32_type.const_int(j, false);
        let lin = builder.build_int_add(lane_x8, j_val, "").unwrap();
        // row_in_tile = lin / 16, col_in_tile = lin % 16
        let row_t = builder.build_int_unsigned_div(lin, c16, "").unwrap();
        let col_t = builder.build_int_unsigned_rem(lin, c16, "").unwrap();
        // Global: A[(block_row + row_t) * K + (t + col_t)]
        let a_row = builder.build_int_add(block_row, row_t, "").unwrap();
        let a_col = builder.build_int_add(t, col_t, "").unwrap();
        let a_idx = builder.build_int_add(
            builder.build_int_mul(a_row, k_param, "").unwrap(),
            a_col, "",
        ).unwrap();
        let a_gep = unsafe { builder.build_gep(f16_type, a_ptr, &[a_idx], "").unwrap() };
        let a_val = builder.build_load(f16_type, a_gep, "").unwrap();
        // Store to smem_a[lin]
        let sa_gep = unsafe { builder.build_gep(f16_type, smem_a_base, &[lin], "").unwrap() };
        builder.build_store(sa_gep, a_val).unwrap();
    }

    // Load B tile [16×8 f16] into shared memory TRANSPOSED as [8×16 f16].
    // 32 threads, 128 elements → 4 elements per thread.
    let lane_x4 = builder.build_int_mul(lane, c4, "lane_x4").unwrap();
    for j in 0..4u64 {
        let j_val = i32_type.const_int(j, false);
        let lin = builder.build_int_add(lane_x4, j_val, "").unwrap();
        // Original B index: row_b = lin / 8, col_b = lin % 8
        let row_b = builder.build_int_unsigned_div(lin, c8, "").unwrap();
        let col_b = builder.build_int_unsigned_rem(lin, c8, "").unwrap();
        // Global: B[(t + row_b) * N + (block_col + col_b)]
        let b_row = builder.build_int_add(t, row_b, "").unwrap();
        let b_col = builder.build_int_add(block_col, col_b, "").unwrap();
        let b_idx = builder.build_int_add(
            builder.build_int_mul(b_row, n_param, "").unwrap(),
            b_col, "",
        ).unwrap();
        let b_gep = unsafe { builder.build_gep(f16_type, b_ptr, &[b_idx], "").unwrap() };
        let b_val = builder.build_load(f16_type, b_gep, "").unwrap();
        // Transposed store: smem_bt[col_b * 16 + row_b]
        let bt_idx = builder.build_int_add(
            builder.build_int_mul(col_b, c16, "").unwrap(),
            row_b, "",
        ).unwrap();
        let sbt_gep = unsafe { builder.build_gep(f16_type, smem_bt_base, &[bt_idx], "").unwrap() };
        builder.build_store(sbt_gep, b_val).unwrap();
    }

    call_barrier0(&context, &module, &builder);

    // ── Load MMA fragments from shared memory ──
    // A fragment: 4 i32 registers (each packing 2 adjacent f16)
    // a[i] = *(i32*)&smem_a[a_frag_idx[i]]
    // a[0]: smem_a[group * 16 + tg * 2]
    // a[1]: smem_a[(group + 8) * 16 + tg * 2]
    // a[2]: smem_a[group * 16 + tg * 2 + 8]
    // a[3]: smem_a[(group + 8) * 16 + tg * 2 + 8]
    let tg_x2 = builder.build_int_mul(tg, c2, "tg_x2").unwrap();
    let group_x16 = builder.build_int_mul(group, c16, "g_x16").unwrap();
    let group8_x16 = builder.build_int_mul(
        builder.build_int_add(group, c8, "").unwrap(),
        c16, "g8_x16",
    ).unwrap();

    let a_frag_indices = [
        builder.build_int_add(group_x16, tg_x2, "af0").unwrap(),
        builder.build_int_add(group8_x16, tg_x2, "af1").unwrap(),
        builder.build_int_add(
            group_x16,
            builder.build_int_add(tg_x2, c8, "").unwrap(),
            "af2",
        ).unwrap(),
        builder.build_int_add(
            group8_x16,
            builder.build_int_add(tg_x2, c8, "").unwrap(),
            "af3",
        ).unwrap(),
    ];

    let mut a_regs = Vec::new();
    for (i, idx) in a_frag_indices.iter().enumerate() {
        let ptr = unsafe {
            builder.build_gep(f16_type, smem_a_base, &[*idx], &format!("a_fp{i}")).unwrap()
        };
        // Load i32 (2 packed f16) from this address
        let val = builder.build_load(i32_type, ptr, &format!("a{i}")).unwrap().into_int_value();
        a_regs.push(val);
    }

    // B fragment: 2 i32 registers from transposed smem_bt[8×16]
    // b[0]: smem_bt[group * 16 + tg * 2]
    // b[1]: smem_bt[group * 16 + tg * 2 + 8]
    let bt_base_idx = builder.build_int_add(group_x16, tg_x2, "bt_bi").unwrap();
    let b_frag_indices = [
        bt_base_idx,
        builder.build_int_add(bt_base_idx, c8, "bf1").unwrap(),
    ];

    let mut b_regs = Vec::new();
    for (i, idx) in b_frag_indices.iter().enumerate() {
        let ptr = unsafe {
            builder.build_gep(f16_type, smem_bt_base, &[*idx], &format!("b_fp{i}")).unwrap()
        };
        let val = builder.build_load(i32_type, ptr, &format!("b{i}")).unwrap().into_int_value();
        b_regs.push(val);
    }

    // ── Inline asm: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ──
    let acc_in = [
        acc0_phi.as_basic_value().into_float_value(),
        acc1_phi.as_basic_value().into_float_value(),
        acc2_phi.as_basic_value().into_float_value(),
        acc3_phi.as_basic_value().into_float_value(),
    ];

    let [d0, d1, d2, d3] = build_mma_sync_asm(
        &builder, &context, &module,
        &a_regs, &b_regs, &acc_in,
    );

    call_barrier0(&context, &module, &builder);

    // Advance tile loop
    let t_next = builder.build_int_add(t, c16, "t_next").unwrap();
    builder.build_unconditional_branch(tile_header).unwrap();

    // Wire phi nodes
    t_phi.add_incoming(&[(&c0, entry), (&t_next, tile_body)]);
    acc0_phi.add_incoming(&[(&f32_zero, entry), (&d0, tile_body)]);
    acc1_phi.add_incoming(&[(&f32_zero, entry), (&d1, tile_body)]);
    acc2_phi.add_incoming(&[(&f32_zero, entry), (&d2, tile_body)]);
    acc3_phi.add_incoming(&[(&f32_zero, entry), (&d3, tile_body)]);

    // ── Tile exit: store C fragment to global memory ──
    builder.position_at_end(tile_exit);

    // C fragment layout for m16n8k16:
    // d[0] → C[block_row + group][block_col + tg*2]
    // d[1] → C[block_row + group][block_col + tg*2 + 1]
    // d[2] → C[block_row + group + 8][block_col + tg*2]
    // d[3] → C[block_row + group + 8][block_col + tg*2 + 1]
    let out_col_base = builder.build_int_add(block_col, tg_x2, "oc_base").unwrap();

    let c_rows = [
        builder.build_int_add(block_row, group, "cr0").unwrap(),
        builder.build_int_add(block_row, group, "cr1").unwrap(),
        builder.build_int_add(
            block_row,
            builder.build_int_add(group, c8, "").unwrap(),
            "cr2",
        ).unwrap(),
        builder.build_int_add(
            block_row,
            builder.build_int_add(group, c8, "").unwrap(),
            "cr3",
        ).unwrap(),
    ];
    let c_cols = [
        out_col_base,
        builder.build_int_add(out_col_base, c1, "oc1").unwrap(),
        out_col_base,
        builder.build_int_add(out_col_base, c1, "oc3").unwrap(),
    ];
    let accs = [
        acc0_phi.as_basic_value().into_float_value(),
        acc1_phi.as_basic_value().into_float_value(),
        acc2_phi.as_basic_value().into_float_value(),
        acc3_phi.as_basic_value().into_float_value(),
    ];

    for i in 0..4 {
        let idx = builder.build_int_add(
            builder.build_int_mul(c_rows[i], n_param, "").unwrap(),
            c_cols[i], &format!("ci{i}"),
        ).unwrap();
        let gep = unsafe {
            builder.build_gep(f32_type, c_ptr, &[idx], &format!("cep{i}")).unwrap()
        };
        builder.build_store(gep, accs[i]).unwrap();
    }

    builder.build_return(None).unwrap();

    // Emit PTX
    let buf = machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow::anyhow!("PTX emission failed: {}", e))?;
    let ptx = std::str::from_utf8(buf.as_slice())
        .context("PTX not UTF-8")?
        .to_string();
    Ok(ptx)
}

// ===========================================================================
// Inline asm for mma.sync
// ===========================================================================

/// Emit inline PTX asm for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
///
/// Returns 4 f32 output values (the D fragment).
fn build_mma_sync_asm<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    a_regs: &[IntValue<'ctx>],   // 4 × i32 (packed f16x2)
    b_regs: &[IntValue<'ctx>],   // 2 × i32 (packed f16x2)
    c_regs: &[FloatValue<'ctx>], // 4 × f32 (accumulator in)
) -> [FloatValue<'ctx>; 4] {
    let i32_type = context.i32_type();
    let f32_type = context.f32_type();

    // Build the inline asm via llvm-sys (inkwell doesn't expose inline asm API directly)
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        // Get builder ref — inkwell's Builder wraps LLVMBuilderRef.
        // The builder's as_mut_ptr() may or may not be public. Use transmute as escape hatch.
        let builder_ref: llvm_sys::prelude::LLVMBuilderRef = builder.as_mut_ptr();

        // Return type: {f32, f32, f32, f32}
        let mut ret_members = [
            f32_type.as_type_ref(),
            f32_type.as_type_ref(),
            f32_type.as_type_ref(),
            f32_type.as_type_ref(),
        ];
        let ret_struct = llvm_sys::core::LLVMStructTypeInContext(
            ctx_ref,
            ret_members.as_mut_ptr(),
            4,
            0, // not packed
        );

        // Parameter types: 4 × i32, 2 × i32, 4 × f32 = 10 params
        let mut param_types = [
            i32_type.as_type_ref(), // a0
            i32_type.as_type_ref(), // a1
            i32_type.as_type_ref(), // a2
            i32_type.as_type_ref(), // a3
            i32_type.as_type_ref(), // b0
            i32_type.as_type_ref(), // b1
            f32_type.as_type_ref(), // c0
            f32_type.as_type_ref(), // c1
            f32_type.as_type_ref(), // c2
            f32_type.as_type_ref(), // c3
        ];
        let fn_type = llvm_sys::core::LLVMFunctionType(
            ret_struct,
            param_types.as_mut_ptr(),
            10,
            0,
        );

        // PTX inline asm string
        let asm_str = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
            {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0";
        let constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f\0";

        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type,
            asm_str.as_ptr() as *const _,
            asm_str.len() - 1, // exclude null terminator from length
            constraints.as_ptr() as *const _,
            constraints.len() - 1,
            1, // has_side_effects
            0, // is_align_stack
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0, // can_throw
        );

        // Build the arguments array
        let mut args: [llvm_sys::prelude::LLVMValueRef; 10] = [
            a_regs[0].as_value_ref(),
            a_regs[1].as_value_ref(),
            a_regs[2].as_value_ref(),
            a_regs[3].as_value_ref(),
            b_regs[0].as_value_ref(),
            b_regs[1].as_value_ref(),
            c_regs[0].as_value_ref(),
            c_regs[1].as_value_ref(),
            c_regs[2].as_value_ref(),
            c_regs[3].as_value_ref(),
        ];

        let name = b"mma\0";
        let call = llvm_sys::core::LLVMBuildCall2(
            builder_ref,
            fn_type,
            asm_val,
            args.as_mut_ptr(),
            10,
            name.as_ptr() as *const _,
        );

        // Extract the 4 f32 results from the struct
        let d0_name = b"d0\0";
        let d1_name = b"d1\0";
        let d2_name = b"d2\0";
        let d3_name = b"d3\0";

        let d0_raw = llvm_sys::core::LLVMBuildExtractValue(
            builder_ref, call, 0, d0_name.as_ptr() as *const _,
        );
        let d1_raw = llvm_sys::core::LLVMBuildExtractValue(
            builder_ref, call, 1, d1_name.as_ptr() as *const _,
        );
        let d2_raw = llvm_sys::core::LLVMBuildExtractValue(
            builder_ref, call, 2, d2_name.as_ptr() as *const _,
        );
        let d3_raw = llvm_sys::core::LLVMBuildExtractValue(
            builder_ref, call, 3, d3_name.as_ptr() as *const _,
        );

        [
            FloatValue::new(d0_raw),
            FloatValue::new(d1_raw),
            FloatValue::new(d2_raw),
            FloatValue::new(d3_raw),
        ]
    }
}
