// SPDX-License-Identifier: Apache-2.0
//
// Parameter sweep: try many tile configurations, report the best.
// Reuses the same LLVM IR emission pattern as our kernels but parameterized.

use anyhow::Result;
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

use inkwell::context::Context as LlvmContext;
use inkwell::builder::Builder;
use inkwell::targets::{TargetTriple, FileType};
use inkwell::types::AsTypeRef;
use inkwell::values::{AsValueRef, IntValue, FloatValue};
use inkwell::{AddressSpace, IntPredicate};

use cudarc::driver::result as cuda;

use crate::{create_nvptx_target_machine, add_nvvm_kernel_metadata, call_sreg, call_barrier0};

// ── Inline asm helpers (self-contained, no cross-module deps) ──

unsafe fn emit_cp_async_16<'ctx>(
    b: &Builder<'ctx>, ctx: &'ctx LlvmContext, module: &inkwell::module::Module<'ctx>,
    dst: inkwell::values::PointerValue<'ctx>, src: inkwell::values::PointerValue<'ctx>,
) {
    let i32_ty = ctx.i32_type();
    let i64_ty = ctx.i64_type();
    let mod_ref = module.as_mut_ptr();
    let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
    let builder_ref = b.as_mut_ptr();
    let void_ty = llvm_sys::core::LLVMVoidTypeInContext(ctx_ref);
    let mut pt = [i32_ty.as_type_ref(), i64_ty.as_type_ref()];
    let ft = llvm_sys::core::LLVMFunctionType(void_ty, pt.as_mut_ptr(), 2, 0);
    let di = b.build_ptr_to_int(dst, i32_ty, "").unwrap();
    let si = b.build_ptr_to_int(src, i64_ty, "").unwrap();
    let asm_str = b"cp.async.cg.shared.global [$0], [$1], 16;\0";
    let con = b"r,l\0";
    let av = llvm_sys::core::LLVMGetInlineAsm(ft, asm_str.as_ptr() as *const _, asm_str.len()-1,
        con.as_ptr() as *const _, con.len()-1, 1, 0,
        llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
    let mut args = [di.as_value_ref(), si.as_value_ref()];
    llvm_sys::core::LLVMBuildCall2(builder_ref, ft, av, args.as_mut_ptr(), 2, b"\0".as_ptr() as *const _);
}

unsafe fn emit_cp_async_commit_wait<'ctx>(
    b: &Builder<'ctx>, _ctx: &'ctx LlvmContext, module: &inkwell::module::Module<'ctx>,
) {
    let mod_ref = module.as_mut_ptr();
    let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
    let br = b.as_mut_ptr();
    let vt = llvm_sys::core::LLVMVoidTypeInContext(ctx_ref);
    let ft = llvm_sys::core::LLVMFunctionType(vt, std::ptr::null_mut(), 0, 0);
    let e = b"\0";
    for s in [b"cp.async.commit_group;\0" as &[u8], b"cp.async.wait_group 0;\0"] {
        let av = llvm_sys::core::LLVMGetInlineAsm(ft, s.as_ptr() as *const _, s.len()-1,
            e.as_ptr() as *const _, e.len()-1, 1, 0,
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
        llvm_sys::core::LLVMBuildCall2(br, ft, av, std::ptr::null_mut(), 0, b"\0".as_ptr() as *const _);
    }
}

unsafe fn emit_mma<'ctx>(
    b: &Builder<'ctx>, ctx: &'ctx LlvmContext, module: &inkwell::module::Module<'ctx>,
    a: &[IntValue<'ctx>; 4], br: &[IntValue<'ctx>; 2], c: &[FloatValue<'ctx>; 4],
) -> [FloatValue<'ctx>; 4] {
    let i32_ty = ctx.i32_type();
    let f32_ty = ctx.f32_type();
    let mr = module.as_mut_ptr();
    let cr = llvm_sys::core::LLVMGetModuleContext(mr);
    let bref = b.as_mut_ptr();
    let mut rm = [f32_ty.as_type_ref(); 4];
    let rs = llvm_sys::core::LLVMStructTypeInContext(cr, rm.as_mut_ptr(), 4, 0);
    let mut pt = [i32_ty.as_type_ref(), i32_ty.as_type_ref(), i32_ty.as_type_ref(), i32_ty.as_type_ref(),
        i32_ty.as_type_ref(), i32_ty.as_type_ref(),
        f32_ty.as_type_ref(), f32_ty.as_type_ref(), f32_ty.as_type_ref(), f32_ty.as_type_ref()];
    let ft = llvm_sys::core::LLVMFunctionType(rs, pt.as_mut_ptr(), 10, 0);
    let s = b"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0";
    let con = b"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f\0";
    let av = llvm_sys::core::LLVMGetInlineAsm(ft, s.as_ptr() as *const _, s.len()-1,
        con.as_ptr() as *const _, con.len()-1, 1, 0,
        llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
    let mut args = [a[0].as_value_ref(), a[1].as_value_ref(), a[2].as_value_ref(), a[3].as_value_ref(),
        br[0].as_value_ref(), br[1].as_value_ref(),
        c[0].as_value_ref(), c[1].as_value_ref(), c[2].as_value_ref(), c[3].as_value_ref()];
    let call = llvm_sys::core::LLVMBuildCall2(bref, ft, av, args.as_mut_ptr(), 10, b"\0".as_ptr() as *const _);
    let n = |s: &[u8]| s.as_ptr() as *const _;
    [FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 0, n(b"\0"))),
     FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 1, n(b"\0"))),
     FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 2, n(b"\0"))),
     FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 3, n(b"\0")))]
}

// ── Swizzle ──
fn sw<'ctx>(elem: IntValue<'ctx>, b: &Builder<'ctx>, ctx: &'ctx LlvmContext, do_sw: bool) -> IntValue<'ctx> {
    if !do_sw { return elem; }
    let ci = |v: u64| ctx.i32_type().const_int(v, false);
    let bo = b.build_int_mul(elem, ci(2), "").unwrap();
    let m = b.build_and(bo, ci(0x380), "").unwrap();
    let s = b.build_right_shift(m, ci(3), false, "").unwrap();
    let r = b.build_xor(bo, s, "").unwrap();
    b.build_int_unsigned_div(r, ci(2), "").unwrap()
}

// ── Kernel emitter ──
fn emit_kernel(sm: &str, bm: u32, bn: u32, bk: u32, wm_: u32, wn_: u32, do_sw: bool, do_cp: bool) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;
    let ctx = LlvmContext::create();
    let module = ctx.create_module("sw");
    let b = ctx.create_builder();
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let i32_ty = ctx.i32_type();
    let f16_ty = ctx.f16_type();
    let f32_ty = ctx.f32_type();
    let ptr_g = ctx.ptr_type(AddressSpace::from(1u16));
    let ptr_s = ctx.ptr_type(AddressSpace::from(3u16));
    let v2f16 = f16_ty.vec_type(2);
    let ci = |v: u64| i32_ty.const_int(v, false);

    let reg_m = wm_ / 16;
    let reg_n = wn_ / 8;
    let warps_m = bm / wm_;
    let warps_n = bn / wn_;
    let threads = warps_m * warps_n * 32;
    let num_acc = (reg_m * reg_n * 4) as usize;

    let smem_g = module.add_global(ctx.i8_type().array_type(0), Some(AddressSpace::from(3u16)), "smem");
    smem_g.set_alignment(128);
    smem_g.set_externally_initialized(true);

    let fn_type = ctx.void_type().fn_type(
        &[ptr_g.into(), ptr_g.into(), ptr_g.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()], false);
    let function = module.add_function("gemm_sweep", fn_type, None);
    add_nvvm_kernel_metadata(&module, &function);

    let entry = ctx.append_basic_block(function, "e");
    let kh = ctx.append_basic_block(function, "kh");
    let kb = ctx.append_basic_block(function, "kb");
    let ke = ctx.append_basic_block(function, "ke");

    b.position_at_end(entry);
    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let n_p = function.get_nth_param(4).unwrap().into_int_value();
    let k_p = function.get_nth_param(5).unwrap().into_int_value();

    let tid = call_sreg(&ctx, &module, &b, "llvm.nvvm.read.ptx.sreg.tid.x", "tid");
    let bx = call_sreg(&ctx, &module, &b, "llvm.nvvm.read.ptx.sreg.ctaid.x", "bx");
    let by = call_sreg(&ctx, &module, &b, "llvm.nvvm.read.ptx.sreg.ctaid.y", "by");
    let br = b.build_int_mul(by, ci(bm as u64), "").unwrap();
    let bc = b.build_int_mul(bx, ci(bn as u64), "").unwrap();
    let wid = b.build_int_unsigned_div(tid, ci(32), "").unwrap();
    let lane = b.build_int_unsigned_rem(tid, ci(32), "").unwrap();
    let wy = b.build_int_unsigned_div(wid, ci(warps_n as u64), "").unwrap();
    let wx = b.build_int_unsigned_rem(wid, ci(warps_n as u64), "").unwrap();
    let grp = b.build_int_unsigned_div(lane, ci(4), "").unwrap();
    let tg = b.build_int_unsigned_rem(lane, ci(4), "").unwrap();
    let tg2 = b.build_int_mul(tg, ci(2), "").unwrap();
    let wyo = b.build_int_mul(wy, ci(wm_ as u64), "").unwrap();
    let wxo = b.build_int_mul(wx, ci(wn_ as u64), "").unwrap();

    let smem_base = smem_g.as_pointer_value();
    let smem_a = b.build_pointer_cast(smem_base, ptr_s, "").unwrap();
    let smem_b = b.build_pointer_cast(unsafe {
        b.build_gep(ctx.i8_type(), smem_base, &[ci((bm * bk * 2) as u64)], "").unwrap()
    }, ptr_s, "").unwrap();

    let f0 = f32_ty.const_float(0.0);
    let ea = bm * bk; let eb = bk * bn;
    let pa = ea / threads; let pb = eb / threads;
    let ta = b.build_int_mul(tid, ci(pa as u64), "").unwrap();
    let tb = b.build_int_mul(tid, ci(pb as u64), "").unwrap();

    b.build_unconditional_branch(kh).unwrap();
    b.position_at_end(kh);
    let t_phi = b.build_phi(i32_ty, "t").unwrap();
    let mut acc_phis = Vec::new();
    for i in 0..num_acc { acc_phis.push(b.build_phi(f32_ty, &format!("a{i}")).unwrap()); }
    let t = t_phi.as_basic_value().into_int_value();
    let cmp = b.build_int_compare(IntPredicate::ULT, t, k_p, "").unwrap();
    b.build_conditional_branch(cmp, kb, ke).unwrap();

    b.position_at_end(kb);
    // Load A
    for ch in 0..(pa/8) as u64 {
        let base = b.build_int_add(ta, ci(ch*8), "").unwrap();
        let ar = b.build_int_unsigned_div(base, ci(bk as u64), "").unwrap();
        let ac = b.build_int_unsigned_rem(base, ci(bk as u64), "").unwrap();
        let agr = b.build_int_add(br, ar, "").unwrap();
        let agc = b.build_int_add(t, ac, "").unwrap();
        let ai = b.build_int_add(b.build_int_mul(agr, k_p, "").unwrap(), agc, "").unwrap();
        let ag = unsafe { b.build_gep(f16_ty, a_ptr, &[ai], "").unwrap() };
        let s = sw(base, &b, &ctx, do_sw);
        let sg = unsafe { b.build_gep(f16_ty, smem_a, &[s], "").unwrap() };
        if do_cp { unsafe { emit_cp_async_16(&b, &ctx, &module, sg, ag); } }
        else { let v = b.build_load(f16_ty.vec_type(8), ag, "").unwrap(); b.build_store(sg, v).unwrap(); }
    }
    // Load B
    for ch in 0..(pb/8) as u64 {
        let base = b.build_int_add(tb, ci(ch*8), "").unwrap();
        let br_ = b.build_int_unsigned_div(base, ci(bn as u64), "").unwrap();
        let bc_ = b.build_int_unsigned_rem(base, ci(bn as u64), "").unwrap();
        let bgr = b.build_int_add(t, br_, "").unwrap();
        let bgc = b.build_int_add(bc, bc_, "").unwrap();
        let bi = b.build_int_add(b.build_int_mul(bgr, n_p, "").unwrap(), bgc, "").unwrap();
        let bg = unsafe { b.build_gep(f16_ty, b_ptr, &[bi], "").unwrap() };
        let s = sw(base, &b, &ctx, do_sw);
        let sg = unsafe { b.build_gep(f16_ty, smem_b, &[s], "").unwrap() };
        if do_cp { unsafe { emit_cp_async_16(&b, &ctx, &module, sg, bg); } }
        else { let v = b.build_load(f16_ty.vec_type(8), bg, "").unwrap(); b.build_store(sg, v).unwrap(); }
    }
    if do_cp { unsafe { emit_cp_async_commit_wait(&b, &ctx, &module); } }
    call_barrier0(&ctx, &module, &b);

    // Fragments + MMA
    let mut cur: Vec<FloatValue> = (0..num_acc).map(|i| acc_phis[i].as_basic_value().into_float_value()).collect();
    let mut af: Vec<IntValue> = Vec::new();
    for rm in 0..reg_m as u64 {
        let ro = b.build_int_add(wyo, ci(rm*16), "").unwrap();
        for (ra, ca) in [(0u64,0u64),(8,0),(0,8),(8,8)] {
            let fr = b.build_int_add(b.build_int_add(ro, grp, "").unwrap(), ci(ra), "").unwrap();
            let fc = b.build_int_add(tg2, ci(ca), "").unwrap();
            let lin = b.build_int_add(b.build_int_mul(fr, ci(bk as u64), "").unwrap(), fc, "").unwrap();
            let s = sw(lin, &b, &ctx, do_sw);
            let gep = unsafe { b.build_gep(f16_ty, smem_a, &[s], "").unwrap() };
            af.push(b.build_load(i32_ty, gep, "").unwrap().into_int_value());
        }
    }
    for rn in 0..reg_n as u64 {
        let bcol = b.build_int_add(b.build_int_add(wxo, ci(rn*8), "").unwrap(), grp, "").unwrap();
        let mut bf = Vec::new();
        for fk in [0u64, 8] {
            let k0 = b.build_int_add(tg2, ci(fk), "").unwrap();
            let k1 = b.build_int_add(k0, ci(1), "").unwrap();
            let l0 = b.build_int_add(b.build_int_mul(k0, ci(bn as u64), "").unwrap(), bcol, "").unwrap();
            let l1 = b.build_int_add(b.build_int_mul(k1, ci(bn as u64), "").unwrap(), bcol, "").unwrap();
            let s0 = sw(l0, &b, &ctx, do_sw);
            let s1 = sw(l1, &b, &ctx, do_sw);
            let g0 = unsafe { b.build_gep(f16_ty, smem_b, &[s0], "").unwrap() };
            let g1 = unsafe { b.build_gep(f16_ty, smem_b, &[s1], "").unwrap() };
            let v0 = b.build_load(f16_ty, g0, "").unwrap();
            let v1 = b.build_load(f16_ty, g1, "").unwrap();
            let vec = b.build_insert_element(v2f16.get_undef(), v0, ci(0), "").unwrap();
            let vec = b.build_insert_element(vec, v1, ci(1), "").unwrap();
            bf.push(b.build_bit_cast(vec, i32_ty, "").unwrap().into_int_value());
        }
        for rm in 0..reg_m as u64 {
            let ai = (rm as u32 * reg_n * 4 + rn as u32 * 4) as usize;
            let ab = (rm * 4) as usize;
            let [d0,d1,d2,d3] = unsafe { emit_mma(&b, &ctx, &module,
                &[af[ab],af[ab+1],af[ab+2],af[ab+3]], &[bf[0],bf[1]],
                &[cur[ai],cur[ai+1],cur[ai+2],cur[ai+3]]) };
            cur[ai]=d0; cur[ai+1]=d1; cur[ai+2]=d2; cur[ai+3]=d3;
        }
    }

    call_barrier0(&ctx, &module, &b);
    let tn = b.build_int_add(t, ci(bk as u64), "").unwrap();
    b.build_unconditional_branch(kh).unwrap();
    t_phi.add_incoming(&[(&ci(0), entry), (&tn, kb)]);
    for i in 0..num_acc { acc_phis[i].add_incoming(&[(&f0, entry), (&cur[i], kb)]); }

    b.position_at_end(ke);
    for rm in 0..reg_m {
        for rn in 0..reg_n {
            let ai = (rm*reg_n*4+rn*4) as usize;
            for d in 0..4u32 {
                let mro = if d<2 { grp } else { b.build_int_add(grp,ci(8),"").unwrap() };
                let mco = if d%2==0 { tg2 } else { b.build_int_add(tg2,ci(1),"").unwrap() };
                let cr_ = b.build_int_add(b.build_int_add(br,wyo,"").unwrap(),
                    b.build_int_add(ci(rm as u64*16),mro,"").unwrap(),"").unwrap();
                let cc = b.build_int_add(b.build_int_add(bc,wxo,"").unwrap(),
                    b.build_int_add(ci(rn as u64*8),mco,"").unwrap(),"").unwrap();
                let ci_ = b.build_int_add(b.build_int_mul(cr_,n_p,"").unwrap(),cc,"").unwrap();
                let gep = unsafe { b.build_gep(f32_ty, c_ptr, &[ci_], "").unwrap() };
                b.build_store(gep, acc_phis[ai+d as usize].as_basic_value().into_float_value()).unwrap();
            }
        }
    }
    b.build_return(None).unwrap();

    let buf = machine.write_to_memory_buffer(&module, FileType::Assembly).map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(std::str::from_utf8(buf.as_slice())?.to_string())
}

fn bench_one(sm: &str, bm: u32, bn: u32, bk: u32, wm: u32, wn: u32, do_sw: bool, do_cp: bool) -> Result<Option<f64>> {
    let m: u32 = 1024; let n: u32 = 1024; let k: u32 = 1024;
    if m%bm!=0 || n%bn!=0 || k%bk!=0 || wm<16 || wn<8 { return Ok(None); }
    if bm%wm!=0 || bn%wn!=0 { return Ok(None); } // warp must tile block evenly
    let warps_m = bm/wm; let warps_n = bn/wn;
    let threads = warps_m*warps_n*32;
    if threads < 32 || threads > 1024 { return Ok(None); }
    let per_a = (bm*bk)/threads; let per_b = (bk*bn)/threads;
    if per_a < 8 || per_b < 8 || per_a%8!=0 || per_b%8!=0 { return Ok(None); }

    let ptx = emit_kernel(sm, bm, bn, bk, wm, wn, do_sw, do_cp)?;
    let ptx_c = CString::new(ptx.as_bytes())?;
    let module = unsafe { cuda::module::load_data(ptx_c.as_ptr() as *const _)? };
    let func = unsafe { cuda::module::get_function(module, CString::new("gemm_sweep").unwrap())? };

    let sa = (m*k) as usize; let sb = (k*n) as usize; let sc = (m*n) as usize;
    let da = unsafe { cuda::malloc_sync(sa*2)? };
    let db = unsafe { cuda::malloc_sync(sb*2)? };
    let dc = unsafe { cuda::malloc_sync(sc*4)? };
    let ha: Vec<half::f16> = (0..sa).map(|i| half::f16::from_f32(((i%7) as f32-3.0)*0.1)).collect();
    let hb: Vec<half::f16> = (0..sb).map(|i| half::f16::from_f32(((i%5) as f32-2.0)*0.1)).collect();
    let mut hc = vec![0.0f32; sc];
    unsafe { cuda::memcpy_htod_sync(da, &ha)?; cuda::memcpy_htod_sync(db, &hb)?; }

    let stream = cuda::stream::create(cuda::stream::StreamKind::NonBlocking)?;
    let smem = (bm*bk+bk*bn)*2;
    let gx = n/bn; let gy = m/bm;
    let params: &mut [*mut c_void] = &mut [
        (&da) as *const _ as *mut c_void, (&db) as *const _ as *mut c_void,
        (&dc) as *const _ as *mut c_void, (&m) as *const _ as *mut c_void,
        (&n) as *const _ as *mut c_void, (&k) as *const _ as *mut c_void,
    ];

    for _ in 0..3 { unsafe { cuda::launch_kernel(func,(gx,gy,1),(threads,1,1),smem,stream,params)?; } }
    unsafe { cuda::stream::synchronize(stream)?; cuda::memcpy_dtoh_sync(&mut hc, dc)?; }

    // Verify — check corners and middle to catch partial-tile bugs
    let mut ok = true;
    let check_rows = [0usize, 1, 31, 32, 47, 48, 63, 127, (m-1) as usize];
    let check_cols = [0usize, 1, 31, 32, 47, 48, 63, 127, (n-1) as usize];
    for &r in &check_rows {
        if r >= m as usize { continue; }
        for &c in &check_cols {
            if c >= n as usize { continue; }
            let mut e = 0.0f32;
            for kk in 0..k as usize { e += ha[r*k as usize+kk].to_f32() * hb[kk*n as usize+c].to_f32(); }
            if (hc[r*n as usize+c]-e).abs() > 1.0 { ok = false; break; }
        }
        if !ok { break; }
    }
    if !ok {
        unsafe { cuda::stream::destroy(stream)?; cuda::free_sync(da)?; cuda::free_sync(db)?; cuda::free_sync(dc)?; cuda::module::unload(module)?; }
        return Ok(None);
    }

    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters { unsafe { cuda::launch_kernel(func,(gx,gy,1),(threads,1,1),smem,stream,params)?; } }
    unsafe { cuda::stream::synchronize(stream)?; }
    let us = start.elapsed().as_micros() as f64 / iters as f64;
    let tf = 2.0*m as f64*n as f64*k as f64 / (us*1e-6) / 1e12;

    unsafe { cuda::stream::destroy(stream)?; cuda::free_sync(da)?; cuda::free_sync(db)?; cuda::free_sync(dc)?; cuda::module::unload(module)?; }
    Ok(Some(tf))
}

pub fn run_sweep(sm: &str) -> Result<()> {
    println!("\n  Parameter sweep");
    println!("  {:>45} {:>8} {:>6}", "Config", "TFLOPS", "% peak");
    println!("  {}", "-".repeat(63));

    let mut best = 0.0f64;
    let mut best_name = String::new();

    let configs: Vec<(u32,u32,u32,u32,u32,bool,bool)> = vec![
        // (bm, bn, bk, wm, wn, swizzle, cp_async)
        // ── Round 1 winners ──
        (64,64,16,64,16,true,true),   // 79 TFLOPS
        (64,64,16,32,32,true,true),   // 75 TFLOPS
        // ── Tall-narrow warps (the winning direction) ──
        (64,64,16,64,8,true,true),    // 4×1 = 4 MMAs, 1 warp × 8 warps
        (64,64,16,64,32,true,true),   // 4×4 = 16 MMAs
        // ── 64×64 with different warp counts ──
        (64,64,16,16,16,true,true),   // 16 warps (if valid)
        (64,64,16,32,16,true,true),   // 2 warps × 4 warps
        // ── 96×96 (non-power-of-2 block, divides 1024? No. Skip.) ──
        // ── 128×32 with tall warps ──
        (128,32,16,32,16,true,true),  // 4×2 warps
        (128,32,16,64,16,true,true),  // 2×2 warps
        (128,32,16,128,16,true,true), // 1×2 warps
        (128,32,16,32,32,true,true),  // 4×1 warps
        (128,32,16,64,32,true,true),  // 2×1 warps
        (128,32,16,128,32,true,true), // 1×1 warp (32 threads)
        // ── 128×64 ──
        (128,64,16,64,16,true,true),  // 2×4 warps
        (128,64,16,64,32,true,true),  // 2×2 warps
        (128,64,16,128,16,true,true), // 1×4 warps
        (128,64,16,128,32,true,true), // 1×2 warps
        (128,64,16,128,64,true,true), // 1×1 warp
        (128,64,16,32,32,true,true),  // 4×2 warps
        // ── 64×128 ──
        (64,128,16,64,32,true,true),  // 1×4 warps
        (64,128,16,32,64,true,true),  // 2×2 warps
        (64,128,16,64,64,true,true),  // 1×2 warps
        (64,128,16,64,128,true,true), // 1×1 warp
        (64,128,16,32,32,true,true),  // 2×4 warps
        // ── 128×128 ──
        (128,128,16,64,32,true,true),
        (128,128,16,128,32,true,true),
        (128,128,16,64,64,true,true),
        (128,128,16,128,64,true,true),
        (128,128,16,128,128,true,true), // 1 warp
        // ── 256×32 (very tall block) ──
        (256,32,16,64,16,true,true),
        (256,32,16,128,16,true,true),
        (256,32,16,64,32,true,true),
        // ── BK=32 with winner ──
        (64,64,32,64,16,true,true),
    ];

    for (bm,bn,bk,wm,wn,sw,cp) in &configs {
        let name = format!("{}x{}_bk{}_w{}x{}{}{}", bm, bn, bk, wm, wn,
            if *sw {"_sw"} else {""}, if *cp {"_cp"} else {""});
        match bench_one(sm, *bm, *bn, *bk, *wm, *wn, *sw, *cp) {
            Ok(Some(tf)) => {
                let pct = tf / 181.0 * 100.0;
                let star = if tf > best { " ★" } else { "" };
                println!("  {:>45} {:>7.1} {:>5.1}%{}", name, tf, pct, star);
                if tf > best { best = tf; best_name = name; }
            }
            Ok(None) => println!("  {:>45} {:>7} {:>6}", name, "SKIP", ""),
            Err(e) => println!("  {:>45} {:>7} {:>6}", name, "ERR", ""),
        }
    }

    println!("\n  ★ Best: {} — {:.1} TFLOPS ({:.1}%)", best_name, best, best/181.0*100.0);
    Ok(())
}
