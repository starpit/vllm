// SPDX-License-Identifier: Apache-2.0
//
// Multi-warp MMA GEMM with register tiling.
//
// 4 warps (128 threads), 2×2 warp layout.
// Each warp computes 32×32 of C via 2×4 register tiling of m16n8k16 MMAs.
// Block tile: 64×64. K-step: 16.
//
// Shared memory per K-step:
//   smem_a[64×16] f16 = 2048 bytes
//   smem_bt[64×16] f16 = 2048 bytes (B transposed for contiguous fragment loads)
//   Total: 4096 bytes
//
// Compute per K-step per block: 4 warps × 8 MMAs × (16×8×16×2) = 131072 FLOPs
// Compute-to-memory ratio: 131072 / 4096 = 32 FLOPs/byte

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

use inkwell::context::Context as LlvmContext;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::targets::{TargetTriple, FileType};
use inkwell::types::AsTypeRef;
use inkwell::values::{AsValueRef, BasicValueEnum, IntValue, FloatValue};
use inkwell::{AddressSpace, IntPredicate};

use cudarc::driver::result as cuda;

use crate::{create_nvptx_target_machine, add_nvvm_kernel_metadata, call_sreg, call_barrier0};

// Tile configuration
const BM: u32 = 64;  // block tile M
const BN: u32 = 64;  // block tile N
const BK: u32 = 16;  // block tile K (= MMA K)
const WM: u32 = 32;  // warp tile M (2 × 16)
const WN: u32 = 32;  // warp tile N (4 × 8)
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const MMA_K: u32 = 16;
const WARPS: u32 = 4; // 2×2 warp layout
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
    let smem_bytes: c_uint = (BM * BK + BN * BK) * 2; // f16

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
    let pct = tflops / 181.0 * 100.0; // L40S f16 TC peak
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

    // Constants
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

    // Warp ID and lane
    let warp_id = builder.build_int_unsigned_div(tid, ci(32), "wid").unwrap();
    let lane = builder.build_int_unsigned_rem(tid, ci(32), "lane").unwrap();

    // 2×2 warp layout: wy = warp_id / 2, wx = warp_id % 2
    let wy = builder.build_int_unsigned_div(warp_id, ci(2), "wy").unwrap();
    let wx = builder.build_int_unsigned_rem(warp_id, ci(2), "wx").unwrap();

    // MMA thread mapping
    let group = builder.build_int_unsigned_div(lane, ci(4), "grp").unwrap();
    let tg = builder.build_int_unsigned_rem(lane, ci(4), "tg").unwrap();
    let tg2 = builder.build_int_mul(tg, ci(2), "tg2").unwrap();

    // Shared memory pointers
    let smem_base = smem_g.as_pointer_value();
    let smem_a = builder.build_pointer_cast(smem_base, ptr_s, "sa").unwrap();
    let smem_bt_off = unsafe {
        builder.build_gep(
            context.i8_type(), smem_base,
            &[ci((BM * BK * 2) as u64)], "bto",
        ).unwrap()
    };
    let smem_bt = builder.build_pointer_cast(smem_bt_off, ptr_s, "sbt").unwrap();

    // Initialize accumulator: 2×4×4 = 32 f32 values (register tiling)
    let f0 = f32_ty.const_float(0.0);

    builder.build_unconditional_branch(kloop_hdr).unwrap();

    // ── K-loop header ──
    builder.position_at_end(kloop_hdr);
    let t_phi = builder.build_phi(i32_ty, "t").unwrap();

    // 32 accumulator phi nodes: acc[rm][rn][d] for rm in 0..2, rn in 0..4, d in 0..4
    let mut acc_phis = Vec::new();
    for i in 0..32 {
        let phi = builder.build_phi(f32_ty, &format!("acc{i}")).unwrap();
        acc_phis.push(phi);
    }

    let t = t_phi.as_basic_value().into_int_value();
    let kcmp = builder.build_int_compare(IntPredicate::ULT, t, k_p, "kcmp").unwrap();
    builder.build_conditional_branch(kcmp, kloop_body, kloop_exit).unwrap();

    // ── K-loop body ──
    builder.position_at_end(kloop_body);

    // ── Load A tile [BM×BK f16] into shared memory ──
    // 128 threads, BM*BK=1024 elements → 8 per thread
    let tid_x8 = builder.build_int_mul(tid, ci(8), "tx8").unwrap();
    for j in 0..8u64 {
        let lin = builder.build_int_add(tid_x8, ci(j), "").unwrap();
        let row_t = builder.build_int_unsigned_div(lin, ci(BK as u64), "").unwrap();
        let col_t = builder.build_int_unsigned_rem(lin, ci(BK as u64), "").unwrap();
        let a_row = builder.build_int_add(block_row, row_t, "").unwrap();
        let a_col = builder.build_int_add(t, col_t, "").unwrap();
        let a_idx = builder.build_int_add(
            builder.build_int_mul(a_row, k_p, "").unwrap(), a_col, "",
        ).unwrap();
        let agep = unsafe { builder.build_gep(f16_ty, a_ptr, &[a_idx], "").unwrap() };
        let av = builder.build_load(f16_ty, agep, "").unwrap();
        let sagep = unsafe { builder.build_gep(f16_ty, smem_a, &[lin], "").unwrap() };
        builder.build_store(sagep, av).unwrap();
    }

    // ── Load B tile [BK×BN f16] transposed into smem_bt [BN×BK f16] ──
    // 128 threads, BK*BN=1024 elements → 8 per thread
    for j in 0..8u64 {
        let lin = builder.build_int_add(tid_x8, ci(j), "").unwrap();
        let row_b = builder.build_int_unsigned_div(lin, ci(BN as u64), "").unwrap();
        let col_b = builder.build_int_unsigned_rem(lin, ci(BN as u64), "").unwrap();
        let b_row = builder.build_int_add(t, row_b, "").unwrap();
        let b_col = builder.build_int_add(block_col, col_b, "").unwrap();
        let b_idx = builder.build_int_add(
            builder.build_int_mul(b_row, n_p, "").unwrap(), b_col, "",
        ).unwrap();
        let bgep = unsafe { builder.build_gep(f16_ty, b_ptr, &[b_idx], "").unwrap() };
        let bv = builder.build_load(f16_ty, bgep, "").unwrap();
        // Transposed index: smem_bt[col_b * BK + row_b]
        let bt_idx = builder.build_int_add(
            builder.build_int_mul(col_b, ci(BK as u64), "").unwrap(), row_b, "",
        ).unwrap();
        let sbtgep = unsafe { builder.build_gep(f16_ty, smem_bt, &[bt_idx], "").unwrap() };
        builder.build_store(sbtgep, bv).unwrap();
    }

    call_barrier0(&context, &module, &builder);

    // ── Load A fragments: a[rm][0..3] for rm in 0..2 ──
    // a[rm][i] = *(i32*)&smem_a[(wy*WM + rm*MMA_M + frag_row) * BK + frag_col]
    let wy_off = builder.build_int_mul(wy, ci(WM as u64), "wyo").unwrap();
    let mut a_frags: Vec<IntValue> = Vec::new(); // 2 * 4 = 8 values

    for rm in 0..2u64 {
        let rm_off = builder.build_int_add(wy_off, ci(rm * MMA_M as u64), "rmo").unwrap();
        // Fragment rows: group, group+8
        // Fragment cols: tg*2, tg*2+8
        for (fi, (row_add, col_add)) in [(0u64, 0u64), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
            let frow = builder.build_int_add(
                builder.build_int_add(rm_off, group, "").unwrap(),
                ci(*row_add), "",
            ).unwrap();
            let fcol = builder.build_int_add(tg2, ci(*col_add), "").unwrap();
            let idx = builder.build_int_add(
                builder.build_int_mul(frow, ci(BK as u64), "").unwrap(), fcol, "",
            ).unwrap();
            let gep = unsafe { builder.build_gep(f16_ty, smem_a, &[idx], "").unwrap() };
            let val = builder.build_load(i32_ty, gep, &format!("a{rm}_{fi}")).unwrap().into_int_value();
            a_frags.push(val);
        }
    }

    // ── Load B fragments: b[rn][0..1] for rn in 0..4 ──
    let wx_off = builder.build_int_mul(wx, ci(WN as u64), "wxo").unwrap();
    let mut b_frags: Vec<IntValue> = Vec::new(); // 4 * 2 = 8 values

    for rn in 0..4u64 {
        let rn_off = builder.build_int_add(wx_off, ci(rn * MMA_N as u64), "rno").unwrap();
        // b[rn] col = rn_off + group → smem_bt[(rn_off + group) * BK + tg*2]
        let bt_row = builder.build_int_add(rn_off, group, "").unwrap();
        for (bi, col_add) in [0u64, 8].iter().enumerate() {
            let fcol = builder.build_int_add(tg2, ci(*col_add), "").unwrap();
            let idx = builder.build_int_add(
                builder.build_int_mul(bt_row, ci(BK as u64), "").unwrap(), fcol, "",
            ).unwrap();
            let gep = unsafe { builder.build_gep(f16_ty, smem_bt, &[idx], "").unwrap() };
            let val = builder.build_load(i32_ty, gep, &format!("b{rn}_{bi}")).unwrap().into_int_value();
            b_frags.push(val);
        }
    }

    // ── Execute 2×4 = 8 MMA operations ──
    let mut new_accs: Vec<FloatValue> = Vec::new();
    for rm in 0..2u32 {
        for rn in 0..4u32 {
            let acc_base = (rm * 4 * 4 + rn * 4) as usize; // index into acc_phis
            let a_base = (rm * 4) as usize;
            let b_base = (rn * 2) as usize;

            let a_regs = [
                a_frags[a_base], a_frags[a_base + 1],
                a_frags[a_base + 2], a_frags[a_base + 3],
            ];
            let b_regs = [b_frags[b_base], b_frags[b_base + 1]];
            let c_regs = [
                acc_phis[acc_base].as_basic_value().into_float_value(),
                acc_phis[acc_base + 1].as_basic_value().into_float_value(),
                acc_phis[acc_base + 2].as_basic_value().into_float_value(),
                acc_phis[acc_base + 3].as_basic_value().into_float_value(),
            ];

            let [d0, d1, d2, d3] = build_mma_asm(
                &builder, &context, &module, &a_regs, &b_regs, &c_regs,
            );
            new_accs.push(d0);
            new_accs.push(d1);
            new_accs.push(d2);
            new_accs.push(d3);
        }
    }

    call_barrier0(&context, &module, &builder);

    let t_next = builder.build_int_add(t, ci(BK as u64), "tn").unwrap();
    builder.build_unconditional_branch(kloop_hdr).unwrap();

    // ── Wire phi nodes ──
    t_phi.add_incoming(&[(&ci(0), entry), (&t_next, kloop_body)]);
    for i in 0..32 {
        acc_phis[i].add_incoming(&[(&f0, entry), (&new_accs[i], kloop_body)]);
    }

    // ── K-loop exit: store C ──
    builder.position_at_end(kloop_exit);

    for rm in 0..2u32 {
        for rn in 0..4u32 {
            let acc_base = (rm * 4 * 4 + rn * 4) as usize;
            // MMA output fragment layout:
            // d[0] → row = group,     col = tg*2
            // d[1] → row = group,     col = tg*2 + 1
            // d[2] → row = group + 8, col = tg*2
            // d[3] → row = group + 8, col = tg*2 + 1
            for d in 0..4u32 {
                let mma_row_off = if d < 2 { group } else {
                    builder.build_int_add(group, ci(8), "").unwrap()
                };
                let mma_col_off = if d % 2 == 0 { tg2 } else {
                    builder.build_int_add(tg2, ci(1), "").unwrap()
                };

                let c_row = builder.build_int_add(
                    builder.build_int_add(block_row, wy_off, "").unwrap(),
                    builder.build_int_add(ci(rm as u64 * MMA_M as u64), mma_row_off, "").unwrap(),
                    "",
                ).unwrap();
                let c_col = builder.build_int_add(
                    builder.build_int_add(block_col, wx_off, "").unwrap(),
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
// Inline asm for mma.sync (same mechanism as step 3)
// ===========================================================================

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
        let d0 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 0, n(b"d0\0"));
        let d1 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 1, n(b"d1\0"));
        let d2 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 2, n(b"d2\0"));
        let d3 = llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 3, n(b"d3\0"));

        [FloatValue::new(d0), FloatValue::new(d1), FloatValue::new(d2), FloatValue::new(d3)]
    }
}
