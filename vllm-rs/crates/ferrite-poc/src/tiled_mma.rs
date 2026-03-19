// SPDX-License-Identifier: Apache-2.0
//
// Multi-warp MMA GEMM with register tiling and vectorized loads.
//
// 4 warps (128 threads), 2×2 warp layout.
// Each warp: 32×32 of C via 2×4 register tiling of m16n8k16 MMAs.
// Block tile: 64×64. K-step: 16.
//
// Vectorized 128-bit global loads (8 f16 per load instruction).
// B transposed during shared memory store for contiguous MMA fragment loads.

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

use inkwell::context::Context as LlvmContext;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::targets::{TargetTriple, FileType};
use inkwell::types::AsTypeRef;
use inkwell::values::{AsValueRef, BasicValueEnum, IntValue, FloatValue, VectorValue};
use inkwell::{AddressSpace, IntPredicate};

use cudarc::driver::result as cuda;

use crate::{create_nvptx_target_machine, add_nvvm_kernel_metadata, call_sreg, call_barrier0};

const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 16;
const WM: u32 = 32;
const WN: u32 = 32;
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const WARPS: u32 = 4;
const THREADS: u32 = WARPS * 32;

pub fn step3b_multiwarp_gemm(sm: &str) -> Result<()> {
    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;

    let ptx = emit_multiwarp_gemm_ptx(sm)?;
    println!("  [llvm] Generated {} bytes of PTX", ptx.len());

    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_multiwarp_gemm.ptx", &ptx)?;
        println!("  [debug] PTX written to /tmp/ferrite_multiwarp_gemm.ptx");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("multiwarp_gemm").unwrap())?
    };
    println!("  [cuda] Module loaded, function resolved");

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
    let grid_x: c_uint = n / BN;
    let grid_y: c_uint = m / BM;
    let smem_bytes: c_uint = (BM * BK + BN * BK) * 2;

    let params: &mut [*mut c_void] = &mut [
        (&d_a) as *const _ as *mut c_void,
        (&d_b) as *const _ as *mut c_void,
        (&d_c) as *const _ as *mut c_void,
        (&m) as *const _ as *mut c_void,
        (&n) as *const _ as *mut c_void,
        (&k) as *const _ as *mut c_void,
    ];

    for _ in 0..5 {
        unsafe {
            cuda::launch_kernel(
                func, (grid_x, grid_y, 1), (THREADS, 1, 1), smem_bytes, stream, params,
            )?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            cuda::launch_kernel(
                func, (grid_x, grid_y, 1), (THREADS, 1, 1), smem_bytes, stream, params,
            )?;
        }
    }
    unsafe { cuda::stream::synchronize(stream)?; }
    let elapsed = start.elapsed();
    let us_per = elapsed.as_micros() as f64 / iters as f64;

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
        println!("  ✓ Correct! (max error: {:.4})", max_err);
    } else {
        println!("  ✗ Max error: {:.4}", max_err);
        bail!("Multi-warp GEMM verification failed");
    }

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / (us_per * 1e-6) / 1e12;
    let pct = tflops / 181.0 * 100.0;
    println!("  {:.1} μs, {:.2} TFLOPS ({:.1}% of L40S peak)", us_per, tflops, pct);

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

fn emit_multiwarp_gemm_ptx(sm: &str) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;
    let context = LlvmContext::create();
    let module = context.create_module("ferrite_mw_gemm");
    let builder = context.create_builder();

    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let i32_ty = context.i32_type();
    let f16_ty = context.f16_type();
    let f32_ty = context.f32_type();
    let void_ty = context.void_type();
    let ptr_g = context.ptr_type(AddressSpace::from(1u16));
    let ptr_s = context.ptr_type(AddressSpace::from(3u16));
    // Vector type: <8 x f16> for 128-bit loads
    let v8f16_ty = f16_ty.vec_type(8);

    // Dynamic shared memory
    let smem_g = module.add_global(
        context.i8_type().array_type(0),
        Some(AddressSpace::from(3u16)),
        "smem",
    );
    smem_g.set_alignment(16);
    smem_g.set_externally_initialized(true);

    let fn_type = void_ty.fn_type(
        &[ptr_g.into(), ptr_g.into(), ptr_g.into(),
          i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );
    let function = module.add_function("multiwarp_gemm", fn_type, None);
    add_nvvm_kernel_metadata(&module, &function);

    let ci = |v: u64| i32_ty.const_int(v, false);

    let entry = context.append_basic_block(function, "entry");
    let kloop_hdr = context.append_basic_block(function, "kloop_hdr");
    let kloop_body = context.append_basic_block(function, "kloop_body");
    let kloop_exit = context.append_basic_block(function, "kloop_exit");

    // ── Entry ──
    builder.position_at_end(entry);

    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let _m_p = function.get_nth_param(3).unwrap().into_int_value();
    let n_p = function.get_nth_param(4).unwrap().into_int_value();
    let k_p = function.get_nth_param(5).unwrap().into_int_value();

    let tid = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.tid.x", "tid");
    let bid_x = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ctaid.x", "bx");
    let bid_y = call_sreg(&context, &module, &builder, "llvm.nvvm.read.ptx.sreg.ctaid.y", "by");

    let block_row = builder.build_int_mul(bid_y, ci(BM as u64), "br").unwrap();
    let block_col = builder.build_int_mul(bid_x, ci(BN as u64), "bc").unwrap();

    let warp_id = builder.build_int_unsigned_div(tid, ci(32), "wid").unwrap();
    let lane = builder.build_int_unsigned_rem(tid, ci(32), "lane").unwrap();
    let wy = builder.build_int_unsigned_div(warp_id, ci(2), "wy").unwrap();
    let wx = builder.build_int_unsigned_rem(warp_id, ci(2), "wx").unwrap();
    let group = builder.build_int_unsigned_div(lane, ci(4), "grp").unwrap();
    let tg = builder.build_int_unsigned_rem(lane, ci(4), "tg").unwrap();
    let tg2 = builder.build_int_mul(tg, ci(2), "tg2").unwrap();

    let smem_base = smem_g.as_pointer_value();
    let smem_a = builder.build_pointer_cast(smem_base, ptr_s, "sa").unwrap();
    let smem_bt_off = unsafe {
        builder.build_gep(
            context.i8_type(), smem_base,
            &[ci((BM * BK * 2) as u64)], "bto",
        ).unwrap()
    };
    let smem_bt = builder.build_pointer_cast(smem_bt_off, ptr_s, "sbt").unwrap();

    let f0 = f32_ty.const_float(0.0);

    builder.build_unconditional_branch(kloop_hdr).unwrap();

    // ── K-loop header ──
    builder.position_at_end(kloop_hdr);
    let t_phi = builder.build_phi(i32_ty, "t").unwrap();
    let mut acc_phis = Vec::new();
    for i in 0..32 {
        acc_phis.push(builder.build_phi(f32_ty, &format!("acc{i}")).unwrap());
    }
    let t = t_phi.as_basic_value().into_int_value();
    let kcmp = builder.build_int_compare(IntPredicate::ULT, t, k_p, "kcmp").unwrap();
    builder.build_conditional_branch(kcmp, kloop_body, kloop_exit).unwrap();

    // ── K-loop body ──
    builder.position_at_end(kloop_body);

    // ── VECTORIZED load A tile [64×16 f16] ──
    // 128 threads, 1024 f16 elements. Each thread loads 8 f16 = one <8 x f16> = 128 bits.
    // Thread tid loads elements [tid*8 .. tid*8+7] in row-major linear order.
    // These 8 elements are contiguous: same row, consecutive columns (since BK=16 > 8).
    // Layout: tid*8/16 = row, tid*8%16 = start col → loads cols [start..start+8) of that row.
    let tid_x8 = builder.build_int_mul(tid, ci(8), "tx8").unwrap();
    let a_tile_row = builder.build_int_unsigned_div(tid_x8, ci(BK as u64), "atr").unwrap();
    let a_tile_col = builder.build_int_unsigned_rem(tid_x8, ci(BK as u64), "atc").unwrap();
    let a_glob_row = builder.build_int_add(block_row, a_tile_row, "agr").unwrap();
    let a_glob_col = builder.build_int_add(t, a_tile_col, "agc").unwrap();
    let a_idx = builder.build_int_add(
        builder.build_int_mul(a_glob_row, k_p, "").unwrap(), a_glob_col, "ai",
    ).unwrap();
    // Vector load: load <8 x f16> from A[a_idx]
    let a_gep = unsafe { builder.build_gep(f16_ty, a_ptr, &[a_idx], "agep").unwrap() };
    let a_vec = builder.build_load(v8f16_ty, a_gep, "avec").unwrap().into_vector_value();
    // Vector store to shared memory
    let sa_gep = unsafe { builder.build_gep(f16_ty, smem_a, &[tid_x8], "sagep").unwrap() };
    builder.build_store(sa_gep, a_vec).unwrap();

    // ── VECTORIZED load B tile [16×64 f16] → transposed store to smem_bt [64×16 f16] ──
    // 128 threads, 1024 elements, 8 per thread.
    // Load 8 contiguous f16 from B (same row, consecutive cols).
    // Then scatter-store transposed: each f16 goes to a different row of smem_bt.
    let b_tile_row = builder.build_int_unsigned_div(tid_x8, ci(BN as u64), "btr").unwrap();
    let b_tile_col = builder.build_int_unsigned_rem(tid_x8, ci(BN as u64), "btc").unwrap();
    let b_glob_row = builder.build_int_add(t, b_tile_row, "bgr").unwrap();
    let b_glob_col = builder.build_int_add(block_col, b_tile_col, "bgc").unwrap();
    let b_idx = builder.build_int_add(
        builder.build_int_mul(b_glob_row, n_p, "").unwrap(), b_glob_col, "bi",
    ).unwrap();
    let b_gep = unsafe { builder.build_gep(f16_ty, b_ptr, &[b_idx], "bgep").unwrap() };
    let b_vec = builder.build_load(v8f16_ty, b_gep, "bvec").unwrap().into_vector_value();

    // Scatter-store transposed: smem_bt[(b_tile_col + j) * BK + b_tile_row] for j in 0..8
    for j in 0..8u64 {
        let elem = builder.build_extract_element(
            b_vec, ci(j), &format!("be{j}"),
        ).unwrap();
        let dst_col = builder.build_int_add(b_tile_col, ci(j), "").unwrap();
        let bt_idx = builder.build_int_add(
            builder.build_int_mul(dst_col, ci(BK as u64), "").unwrap(),
            b_tile_row, "",
        ).unwrap();
        let sbt_gep = unsafe { builder.build_gep(f16_ty, smem_bt, &[bt_idx], "").unwrap() };
        builder.build_store(sbt_gep, elem).unwrap();
    }

    call_barrier0(&context, &module, &builder);

    // ── Load A fragments via ldmatrix ──
    // ldmatrix.sync.aligned.m8n8.x4.shared.b16 loads 4 registers (16×16 f16 tile)
    // Each thread provides addr for row (lane % 16) of the tile.
    // For m16n8k16 MMA, we need the A fragment from a 16×16 sub-tile.
    let wy_off = builder.build_int_mul(wy, ci(WM as u64), "wyo").unwrap();
    let lane_mod16 = builder.build_int_unsigned_rem(lane, ci(16), "lm16").unwrap();

    let mut a_frags: Vec<IntValue> = Vec::new();
    for rm in 0..2u64 {
        let rm_off = builder.build_int_add(wy_off, ci(rm * MMA_M as u64), "rmo").unwrap();
        // Row in shared memory for this thread's ldmatrix address
        let smem_row = builder.build_int_add(rm_off, lane_mod16, "asr").unwrap();
        // Byte offset: row * BK * sizeof(f16) = row * BK * 2
        let byte_off = builder.build_int_mul(
            builder.build_int_mul(smem_row, ci(BK as u64), "").unwrap(),
            ci(2), "abo",
        ).unwrap();
        let [r0, r1, r2, r3] = build_ldmatrix_x4_asm(
            &builder, &context, &module, smem_a, byte_off,
        );
        a_frags.push(r0); a_frags.push(r1); a_frags.push(r2); a_frags.push(r3);
    }

    // ── Load B fragments via ldmatrix with transpose ──
    // B is stored transposed in smem_bt[BN×BK]. Each column of original B = one row of smem_bt.
    // ldmatrix.sync.aligned.m8n8.x2.trans loads 2 registers with auto-transpose.
    let wx_off = builder.build_int_mul(wx, ci(WN as u64), "wxo").unwrap();

    let mut b_frags: Vec<IntValue> = Vec::new();
    for rn in 0..4u64 {
        let rn_off = builder.build_int_add(wx_off, ci(rn * MMA_N as u64), "rno").unwrap();
        // For .trans ldmatrix, each thread provides address for col (lane % 8) of the tile
        let lane_mod8 = builder.build_int_unsigned_rem(lane, ci(8), "lm8").unwrap();
        let bt_row = builder.build_int_add(rn_off, lane_mod8, "btr").unwrap();
        let byte_off = builder.build_int_mul(
            builder.build_int_mul(bt_row, ci(BK as u64), "").unwrap(),
            ci(2), "bbo",
        ).unwrap();
        let [r0, r1] = build_ldmatrix_x2_trans_asm(
            &builder, &context, &module, smem_bt, byte_off,
        );
        b_frags.push(r0); b_frags.push(r1);
    }

    // ── 2×4 MMA operations ──
    let mut new_accs: Vec<FloatValue> = Vec::new();
    for rm in 0..2u32 {
        for rn in 0..4u32 {
            let acc_base = (rm * 4 * 4 + rn * 4) as usize;
            let a_base = (rm * 4) as usize;
            let b_base = (rn * 2) as usize;

            let [d0, d1, d2, d3] = build_mma_asm(
                &builder, &context, &module,
                &[a_frags[a_base], a_frags[a_base+1], a_frags[a_base+2], a_frags[a_base+3]],
                &[b_frags[b_base], b_frags[b_base+1]],
                &[
                    acc_phis[acc_base].as_basic_value().into_float_value(),
                    acc_phis[acc_base+1].as_basic_value().into_float_value(),
                    acc_phis[acc_base+2].as_basic_value().into_float_value(),
                    acc_phis[acc_base+3].as_basic_value().into_float_value(),
                ],
            );
            new_accs.push(d0); new_accs.push(d1); new_accs.push(d2); new_accs.push(d3);
        }
    }

    call_barrier0(&context, &module, &builder);

    let t_next = builder.build_int_add(t, ci(BK as u64), "tn").unwrap();
    builder.build_unconditional_branch(kloop_hdr).unwrap();

    // Wire phi nodes
    t_phi.add_incoming(&[(&ci(0), entry), (&t_next, kloop_body)]);
    for i in 0..32 {
        acc_phis[i].add_incoming(&[(&f0, entry), (&new_accs[i], kloop_body)]);
    }

    // ── K-loop exit: store C ──
    builder.position_at_end(kloop_exit);

    let wy_off_exit = builder.build_int_mul(wy, ci(WM as u64), "wyo2").unwrap();
    let wx_off_exit = builder.build_int_mul(wx, ci(WN as u64), "wxo2").unwrap();

    for rm in 0..2u32 {
        for rn in 0..4u32 {
            let acc_base = (rm * 4 * 4 + rn * 4) as usize;
            for d in 0..4u32 {
                let mma_row_off = if d < 2 { group } else {
                    builder.build_int_add(group, ci(8), "").unwrap()
                };
                let mma_col_off = if d % 2 == 0 { tg2 } else {
                    builder.build_int_add(tg2, ci(1), "").unwrap()
                };

                let c_row = builder.build_int_add(
                    builder.build_int_add(block_row, wy_off_exit, "").unwrap(),
                    builder.build_int_add(ci(rm as u64 * MMA_M as u64), mma_row_off, "").unwrap(),
                    "",
                ).unwrap();
                let c_col = builder.build_int_add(
                    builder.build_int_add(block_col, wx_off_exit, "").unwrap(),
                    builder.build_int_add(ci(rn as u64 * MMA_N as u64), mma_col_off, "").unwrap(),
                    "",
                ).unwrap();
                let c_idx = builder.build_int_add(
                    builder.build_int_mul(c_row, n_p, "").unwrap(), c_col, "",
                ).unwrap();
                let gep = unsafe {
                    builder.build_gep(f32_ty, c_ptr, &[c_idx], "").unwrap()
                };
                let val = acc_phis[acc_base + d as usize].as_basic_value().into_float_value();
                builder.build_store(gep, val).unwrap();
            }
        }
    }

    builder.build_return(None).unwrap();

    let buf = machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow::anyhow!("PTX emission: {}", e))?;
    let ptx = std::str::from_utf8(buf.as_slice()).context("PTX not UTF-8")?.to_string();
    Ok(ptx)
}

// ===========================================================================
// Inline asm for mma.sync
// ===========================================================================

/// ldmatrix.sync.aligned.m8n8.x4.shared.b16 — loads 4 × i32 from shared memory.
/// `smem_ptr` is the base shared pointer, `byte_offset` is the per-thread byte offset.
fn build_ldmatrix_x4_asm<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    smem_ptr: inkwell::values::PointerValue<'ctx>,
    byte_offset: IntValue<'ctx>,
) -> [IntValue<'ctx>; 4] {
    let i32_ty = context.i32_type();
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();

        // Return type: {i32, i32, i32, i32}
        let mut ret_members = [
            i32_ty.as_type_ref(), i32_ty.as_type_ref(),
            i32_ty.as_type_ref(), i32_ty.as_type_ref(),
        ];
        let ret_struct = llvm_sys::core::LLVMStructTypeInContext(
            ctx_ref, ret_members.as_mut_ptr(), 4, 0,
        );
        // Input: i32 (shared memory address as 32-bit)
        let mut param_types = [i32_ty.as_type_ref()];
        let fn_type = llvm_sys::core::LLVMFunctionType(
            ret_struct, param_types.as_mut_ptr(), 1, 0,
        );

        // Convert shared pointer + byte offset to 32-bit shared address via cvta
        // We'll compute the address in LLVM IR and pass it as i32 to the asm.
        let addr_ptr = unsafe {
            builder.build_gep(context.i8_type(), smem_ptr, &[byte_offset], "ldm_addr").unwrap()
        };
        let addr_i32 = builder.build_ptr_to_int(addr_ptr, i32_ty, "addr32").unwrap();

        let asm_str = b"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0";
        let constraints = b"=r,=r,=r,=r,r\0";

        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type,
            asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            1, 0,
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0,
        );

        let mut args = [addr_i32.as_value_ref()];
        let call = llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val,
            args.as_mut_ptr(), 1, b"\0".as_ptr() as *const _,
        );

        let n = |s: &[u8]| s.as_ptr() as *const _;
        [
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 0, n(b"\0"))),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 1, n(b"\0"))),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 2, n(b"\0"))),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 3, n(b"\0"))),
        ]
    }
}

/// ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 — loads 2 × i32 with transpose.
fn build_ldmatrix_x2_trans_asm<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    smem_ptr: inkwell::values::PointerValue<'ctx>,
    byte_offset: IntValue<'ctx>,
) -> [IntValue<'ctx>; 2] {
    let i32_ty = context.i32_type();
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();

        let mut ret_members = [i32_ty.as_type_ref(), i32_ty.as_type_ref()];
        let ret_struct = llvm_sys::core::LLVMStructTypeInContext(
            ctx_ref, ret_members.as_mut_ptr(), 2, 0,
        );
        let mut param_types = [i32_ty.as_type_ref()];
        let fn_type = llvm_sys::core::LLVMFunctionType(
            ret_struct, param_types.as_mut_ptr(), 1, 0,
        );

        let addr_ptr = unsafe {
            builder.build_gep(context.i8_type(), smem_ptr, &[byte_offset], "ldmt_addr").unwrap()
        };
        let addr_i32 = builder.build_ptr_to_int(addr_ptr, i32_ty, "addr32t").unwrap();

        let asm_str = b"ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0";
        let constraints = b"=r,=r,r\0";

        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type,
            asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            1, 0,
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0,
        );

        let mut args = [addr_i32.as_value_ref()];
        let call = llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val,
            args.as_mut_ptr(), 1, b"\0".as_ptr() as *const _,
        );

        let n = |s: &[u8]| s.as_ptr() as *const _;
        [
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 0, n(b"\0"))),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 1, n(b"\0"))),
        ]
    }
}

fn build_mma_asm<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    a_regs: &[IntValue<'ctx>; 4],
    b_regs: &[IntValue<'ctx>; 2],
    c_regs: &[FloatValue<'ctx>; 4],
) -> [FloatValue<'ctx>; 4] {
    let i32_ty = context.i32_type();
    let f32_ty = context.f32_type();

    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();

        let mut ret_members = [
            f32_ty.as_type_ref(), f32_ty.as_type_ref(),
            f32_ty.as_type_ref(), f32_ty.as_type_ref(),
        ];
        let ret_struct = llvm_sys::core::LLVMStructTypeInContext(
            ctx_ref, ret_members.as_mut_ptr(), 4, 0,
        );

        let mut param_types = [
            i32_ty.as_type_ref(), i32_ty.as_type_ref(),
            i32_ty.as_type_ref(), i32_ty.as_type_ref(),
            i32_ty.as_type_ref(), i32_ty.as_type_ref(),
            f32_ty.as_type_ref(), f32_ty.as_type_ref(),
            f32_ty.as_type_ref(), f32_ty.as_type_ref(),
        ];
        let fn_type = llvm_sys::core::LLVMFunctionType(
            ret_struct, param_types.as_mut_ptr(), 10, 0,
        );

        let asm_str = b"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0";
        let constraints = b"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f\0";

        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type,
            asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            1, 0,
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0,
        );

        let mut args = [
            a_regs[0].as_value_ref(), a_regs[1].as_value_ref(),
            a_regs[2].as_value_ref(), a_regs[3].as_value_ref(),
            b_regs[0].as_value_ref(), b_regs[1].as_value_ref(),
            c_regs[0].as_value_ref(), c_regs[1].as_value_ref(),
            c_regs[2].as_value_ref(), c_regs[3].as_value_ref(),
        ];

        let name = b"\0";
        let call = llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val,
            args.as_mut_ptr(), 10, name.as_ptr() as *const _,
        );

        let n = |s: &[u8]| s.as_ptr() as *const _;
        let d0 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 0, n(b"\0"));
        let d1 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 1, n(b"\0"));
        let d2 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 2, n(b"\0"));
        let d3 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 3, n(b"\0"));

        [FloatValue::new(d0), FloatValue::new(d1), FloatValue::new(d2), FloatValue::new(d3)]
    }
}
