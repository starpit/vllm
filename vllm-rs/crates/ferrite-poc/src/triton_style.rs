// Triton-style GEMM: BK=32, ldmatrix, pipelined cp.async, -nvptx-short-ptr.
// Targeting 132 TFLOPS (what Triton achieves on same hardware).

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

use inkwell::context::Context as LlvmContext;
use inkwell::builder::Builder;
use inkwell::module::Module;
use inkwell::targets::{TargetTriple, FileType};
use inkwell::types::AsTypeRef;
use inkwell::values::{AsValueRef, IntValue, FloatValue};
use inkwell::{AddressSpace, IntPredicate};

use cudarc::driver::result as cuda;
use crate::{create_nvptx_target_machine, add_nvvm_kernel_metadata, call_sreg, call_barrier0};

// Match Triton's winning config: 64×64 BK=32
const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 32; // 2× K-unroll
const WM: u32 = 64;
const WN: u32 = 16;
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const MMA_K: u32 = 16;
const REG_M: u32 = WM / MMA_M; // 4
const REG_N: u32 = WN / MMA_N; // 2
const K_ITERS: u32 = BK / MMA_K; // 2
const WARPS: u32 = (BM / WM) * (BN / WN); // 1×4 = 4
const THREADS: u32 = WARPS * 32;

const SWIZZLE_MASK: u32 = 0x380;
const SWIZZLE_SHIFT: u32 = 3;

pub fn run(sm: &str) -> Result<()> {
    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;

    let ptx = emit_ptx(sm)?;
    println!("  [llvm] Generated {} bytes of PTX", ptx.len());

    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_triton_style.ptx", &ptx)?;
        println!("  [debug] PTX dumped");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("triton_style_gemm").unwrap())?
    };
    println!("  [cuda] Loaded");

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
    let smem_bytes: c_uint = (BM * BK + BK * BN) * 2;
    let gx = n / BN;
    let gy = m / BM;
    let params: &mut [*mut c_void] = &mut [
        (&da) as *const _ as *mut c_void, (&db) as *const _ as *mut c_void,
        (&dc) as *const _ as *mut c_void, (&m) as *const _ as *mut c_void,
        (&n) as *const _ as *mut c_void, (&k) as *const _ as *mut c_void,
    ];

    for _ in 0..10 {
        unsafe { cuda::launch_kernel(func, (gx, gy, 1), (THREADS, 1, 1), smem_bytes, stream, params)?; }
    }
    unsafe { cuda::stream::synchronize(stream)?; }

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe { cuda::launch_kernel(func, (gx, gy, 1), (THREADS, 1, 1), smem_bytes, stream, params)?; }
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

    if max_err < 1.0 { println!("  ✓ Correct (max err: {:.4})", max_err); }
    else { bail!("  ✗ Max error: {:.4}", max_err); }

    let tf = 2.0 * m as f64 * n as f64 * k as f64 / (us * 1e-6) / 1e12;
    println!("  {:.1} μs, {:.1} TFLOPS ({:.1}% of cuBLAS 162)", us, tf, tf / 162.0 * 100.0);

    unsafe {
        cuda::stream::destroy(stream)?;
        cuda::free_sync(da)?; cuda::free_sync(db)?; cuda::free_sync(dc)?;
        cuda::module::unload(module)?;
    }
    Ok(())
}

// ── Swizzle (returns byte offset, no division) ──
fn swizzle_bytes<'ctx>(b: &Builder<'ctx>, ctx: &'ctx LlvmContext, elem_idx: IntValue<'ctx>) -> IntValue<'ctx> {
    let ci = |v: u64| ctx.i32_type().const_int(v, false);
    // elem_idx * 2 → byte offset, apply XOR swizzle, return bytes
    let byte = b.build_left_shift(elem_idx, ci(1), "").unwrap(); // ×2 via shift
    let masked = b.build_and(byte, ci(SWIZZLE_MASK as u64), "").unwrap();
    let shifted = b.build_right_shift(masked, ci(SWIZZLE_SHIFT as u64), false, "").unwrap();
    b.build_xor(byte, shifted, "").unwrap() // returns byte offset
}

// ── ldmatrix.sync.aligned.m8n8.x4.shared.b16 ──
// With -nvptx-short-ptr, ptrtoint on addrspace(3) gives a 32-bit %r register.
fn ldmatrix_x4<'ctx>(
    b: &Builder<'ctx>, ctx: &'ctx LlvmContext, module: &Module<'ctx>,
    smem_ptr: inkwell::values::PointerValue<'ctx>, byte_offset: IntValue<'ctx>,
) -> [IntValue<'ctx>; 4] {
    let i32_ty = ctx.i32_type();
    unsafe {
        let mr = module.as_mut_ptr();
        let cr = llvm_sys::core::LLVMGetModuleContext(mr);
        let br = b.as_mut_ptr();

        // GEP to the right byte, then ptrtoint → 32-bit with short-ptr
        let ptr = b.build_gep(ctx.i8_type(), smem_ptr, &[byte_offset], "").unwrap();
        let addr = b.build_ptr_to_int(ptr, i32_ty, "").unwrap();

        let mut rm = [i32_ty.as_type_ref(); 4];
        let rs = llvm_sys::core::LLVMStructTypeInContext(cr, rm.as_mut_ptr(), 4, 0);
        let mut pt = [i32_ty.as_type_ref()];
        let ft = llvm_sys::core::LLVMFunctionType(rs, pt.as_mut_ptr(), 1, 0);

        let asm_str = b"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0";
        let con = b"=r,=r,=r,=r,r\0";
        let av = llvm_sys::core::LLVMGetInlineAsm(ft,
            asm_str.as_ptr() as *const _, asm_str.len() - 1,
            con.as_ptr() as *const _, con.len() - 1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);

        let mut args = [addr.as_value_ref()];
        let call = llvm_sys::core::LLVMBuildCall2(br, ft, av,
            args.as_mut_ptr(), 1, b"\0".as_ptr() as *const _);

        [
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 0, b"\0".as_ptr() as *const _)),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 1, b"\0".as_ptr() as *const _)),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 2, b"\0".as_ptr() as *const _)),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 3, b"\0".as_ptr() as *const _)),
        ]
    }
}

// ── ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 (for B fragments) ──
fn ldmatrix_x4_trans<'ctx>(
    b: &Builder<'ctx>, ctx: &'ctx LlvmContext, module: &Module<'ctx>,
    smem_ptr: inkwell::values::PointerValue<'ctx>, byte_offset: IntValue<'ctx>,
) -> [IntValue<'ctx>; 4] {
    let i32_ty = ctx.i32_type();
    unsafe {
        let mr = module.as_mut_ptr();
        let cr = llvm_sys::core::LLVMGetModuleContext(mr);
        let br = b.as_mut_ptr();
        let ptr = b.build_gep(ctx.i8_type(), smem_ptr, &[byte_offset], "").unwrap();
        let addr = b.build_ptr_to_int(ptr, i32_ty, "").unwrap();
        let mut rm = [i32_ty.as_type_ref(); 4];
        let rs = llvm_sys::core::LLVMStructTypeInContext(cr, rm.as_mut_ptr(), 4, 0);
        let mut pt = [i32_ty.as_type_ref()];
        let ft = llvm_sys::core::LLVMFunctionType(rs, pt.as_mut_ptr(), 1, 0);
        let asm_str = b"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0,$1,$2,$3}, [$4];\0";
        let con = b"=r,=r,=r,=r,r\0";
        let av = llvm_sys::core::LLVMGetInlineAsm(ft,
            asm_str.as_ptr() as *const _, asm_str.len() - 1,
            con.as_ptr() as *const _, con.len() - 1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
        let mut args = [addr.as_value_ref()];
        let call = llvm_sys::core::LLVMBuildCall2(br, ft, av,
            args.as_mut_ptr(), 1, b"\0".as_ptr() as *const _);
        [
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 0, b"\0".as_ptr() as *const _)),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 1, b"\0".as_ptr() as *const _)),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 2, b"\0".as_ptr() as *const _)),
            IntValue::new(llvm_sys::core::LLVMBuildExtractValue(br, call, 3, b"\0".as_ptr() as *const _)),
        ]
    }
}

// ── cp.async ──
fn cp_async_16<'ctx>(
    b: &Builder<'ctx>, ctx: &'ctx LlvmContext, module: &Module<'ctx>,
    dst: inkwell::values::PointerValue<'ctx>, src: inkwell::values::PointerValue<'ctx>,
) {
    let i32_ty = ctx.i32_type();
    let i64_ty = ctx.i64_type();
    unsafe {
        let mr = module.as_mut_ptr();
        let cr = llvm_sys::core::LLVMGetModuleContext(mr);
        let br = b.as_mut_ptr();
        let vt = llvm_sys::core::LLVMVoidTypeInContext(cr);
        let mut pt = [i32_ty.as_type_ref(), i64_ty.as_type_ref()];
        let ft = llvm_sys::core::LLVMFunctionType(vt, pt.as_mut_ptr(), 2, 0);
        let di = b.build_ptr_to_int(dst, i32_ty, "").unwrap();
        let si = b.build_ptr_to_int(src, i64_ty, "").unwrap();
        let s = b"cp.async.cg.shared.global [$0], [$1], 16;\0";
        let c = b"r,l\0";
        let av = llvm_sys::core::LLVMGetInlineAsm(ft,
            s.as_ptr() as *const _, s.len()-1, c.as_ptr() as *const _, c.len()-1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
        let mut args = [di.as_value_ref(), si.as_value_ref()];
        llvm_sys::core::LLVMBuildCall2(br, ft, av, args.as_mut_ptr(), 2, b"\0".as_ptr() as *const _);
    }
}

fn cp_async_commit<'ctx>(b: &Builder<'ctx>, _ctx: &'ctx LlvmContext, module: &Module<'ctx>) {
    unsafe {
        let cr = llvm_sys::core::LLVMGetModuleContext(module.as_mut_ptr());
        let br = b.as_mut_ptr();
        let vt = llvm_sys::core::LLVMVoidTypeInContext(cr);
        let ft = llvm_sys::core::LLVMFunctionType(vt, std::ptr::null_mut(), 0, 0);
        let s = b"cp.async.commit_group;\0";
        let e = b"\0";
        let av = llvm_sys::core::LLVMGetInlineAsm(ft,
            s.as_ptr() as *const _, s.len()-1, e.as_ptr() as *const _, e.len()-1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
        llvm_sys::core::LLVMBuildCall2(br, ft, av, std::ptr::null_mut(), 0, b"\0".as_ptr() as *const _);
    }
}

fn cp_async_wait_group<'ctx>(b: &Builder<'ctx>, _ctx: &'ctx LlvmContext, module: &Module<'ctx>, n: u32) {
    unsafe {
        let cr = llvm_sys::core::LLVMGetModuleContext(module.as_mut_ptr());
        let br = b.as_mut_ptr();
        let vt = llvm_sys::core::LLVMVoidTypeInContext(cr);
        let ft = llvm_sys::core::LLVMFunctionType(vt, std::ptr::null_mut(), 0, 0);
        let s = format!("cp.async.wait_group {};\0", n);
        let e = b"\0";
        let av = llvm_sys::core::LLVMGetInlineAsm(ft,
            s.as_ptr() as *const _, s.len()-1, e.as_ptr() as *const _, e.len()-1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
        llvm_sys::core::LLVMBuildCall2(br, ft, av, std::ptr::null_mut(), 0, b"\0".as_ptr() as *const _);
    }
}

fn mma_sync<'ctx>(
    b: &Builder<'ctx>, ctx: &'ctx LlvmContext, module: &Module<'ctx>,
    a: &[IntValue<'ctx>; 4], br_: &[IntValue<'ctx>; 2], c: &[FloatValue<'ctx>; 4],
) -> [FloatValue<'ctx>; 4] {
    let i32_ty = ctx.i32_type();
    let f32_ty = ctx.f32_type();
    unsafe {
        let mr = module.as_mut_ptr();
        let cr = llvm_sys::core::LLVMGetModuleContext(mr);
        let bref = b.as_mut_ptr();
        let mut rm = [f32_ty.as_type_ref(); 4];
        let rs = llvm_sys::core::LLVMStructTypeInContext(cr, rm.as_mut_ptr(), 4, 0);
        let mut pt = [i32_ty.as_type_ref(),i32_ty.as_type_ref(),i32_ty.as_type_ref(),i32_ty.as_type_ref(),
            i32_ty.as_type_ref(),i32_ty.as_type_ref(),
            f32_ty.as_type_ref(),f32_ty.as_type_ref(),f32_ty.as_type_ref(),f32_ty.as_type_ref()];
        let ft = llvm_sys::core::LLVMFunctionType(rs, pt.as_mut_ptr(), 10, 0);
        let s = b"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0";
        let con = b"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f\0";
        let av = llvm_sys::core::LLVMGetInlineAsm(ft,
            s.as_ptr() as *const _, s.len()-1, con.as_ptr() as *const _, con.len()-1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
        let mut args = [a[0].as_value_ref(),a[1].as_value_ref(),a[2].as_value_ref(),a[3].as_value_ref(),
            br_[0].as_value_ref(),br_[1].as_value_ref(),
            c[0].as_value_ref(),c[1].as_value_ref(),c[2].as_value_ref(),c[3].as_value_ref()];
        let call = llvm_sys::core::LLVMBuildCall2(bref, ft, av, args.as_mut_ptr(), 10, b"\0".as_ptr() as *const _);
        let n = |s: &[u8]| s.as_ptr() as *const _;
        [FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 0, n(b"\0"))),
         FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 1, n(b"\0"))),
         FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 2, n(b"\0"))),
         FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(bref, call, 3, n(b"\0")))]
    }
}

fn emit_ptx(sm: &str) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;
    let ctx = LlvmContext::create();
    let module = ctx.create_module("triton_style");
    let b = ctx.create_builder();
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let i32_ty = ctx.i32_type();
    let f16_ty = ctx.f16_type();
    let f32_ty = ctx.f32_type();
    let ptr_g = ctx.ptr_type(AddressSpace::from(1u16));
    let ptr_s = ctx.ptr_type(AddressSpace::from(3u16));

    let smem_g = module.add_global(ctx.i8_type().array_type(0), Some(AddressSpace::from(3u16)), "smem");
    smem_g.set_alignment(128);
    smem_g.set_externally_initialized(true);

    let fn_type = ctx.void_type().fn_type(
        &[ptr_g.into(), ptr_g.into(), ptr_g.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()], false);
    let function = module.add_function("triton_style_gemm", fn_type, None);
    add_nvvm_kernel_metadata(&module, &function);

    let ci = |v: u64| i32_ty.const_int(v, false);

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
    let block_row = b.build_int_mul(by, ci(BM as u64), "").unwrap();
    let block_col = b.build_int_mul(bx, ci(BN as u64), "").unwrap();

    let warp_id = b.build_int_unsigned_div(tid, ci(32), "").unwrap();
    let lane = b.build_int_unsigned_rem(tid, ci(32), "").unwrap();
    // 1×4 warp layout (WM=64 → 1 warp along M, WN=16 → 4 warps along N)
    let wx = warp_id; // 0..3, all along N
    let group = b.build_int_unsigned_div(lane, ci(4), "").unwrap();
    let tg = b.build_int_unsigned_rem(lane, ci(4), "").unwrap();
    let tg2 = b.build_int_mul(tg, ci(2), "").unwrap();
    let wx_off = b.build_int_mul(wx, ci(WN as u64), "").unwrap();

    let smem_base = smem_g.as_pointer_value();
    let smem_a = b.build_pointer_cast(smem_base, ptr_s, "sa").unwrap();
    let smem_b_off = unsafe {
        b.build_gep(ctx.i8_type(), smem_base, &[ci((BM * BK * 2) as u64)], "").unwrap()
    };
    let smem_b = b.build_pointer_cast(smem_b_off, ptr_s, "sb").unwrap();

    let f0 = f32_ty.const_float(0.0);
    let num_acc = (REG_M * REG_N * 4) as usize; // 4×2×4 = 32

    // Loading: A[64×32]=2048 f16, B[32×64]=2048 f16
    // 128 threads → 16 per thread → 2 cp.async(8) each = 4 cp.async total per thread
    let tid_x16 = b.build_int_mul(tid, ci(16), "").unwrap();

    b.build_unconditional_branch(kh).unwrap();

    // ── K-loop ──
    b.position_at_end(kh);
    let t_phi = b.build_phi(i32_ty, "t").unwrap();
    let mut acc_phis = Vec::new();
    for i in 0..num_acc { acc_phis.push(b.build_phi(f32_ty, &format!("a{i}")).unwrap()); }
    let t = t_phi.as_basic_value().into_int_value();
    let cmp = b.build_int_compare(IntPredicate::ULT, t, k_p, "").unwrap();
    b.build_conditional_branch(cmp, kb, ke).unwrap();

    b.position_at_end(kb);

    // ── Load A and B via cp.async (4 cp.async per thread) ──
    for chunk in 0..2u64 {
        let base = b.build_int_add(tid_x16, ci(chunk * 8), "").unwrap();
        let row = b.build_int_unsigned_div(base, ci(BK as u64), "").unwrap();
        let col = b.build_int_unsigned_rem(base, ci(BK as u64), "").unwrap();
        let grow = b.build_int_add(block_row, row, "").unwrap();
        let gcol = b.build_int_add(t, col, "").unwrap();
        let gidx = b.build_int_add(b.build_int_mul(grow, k_p, "").unwrap(), gcol, "").unwrap();
        let gep = unsafe { b.build_gep(f16_ty, a_ptr, &[gidx], "").unwrap() };
        let sw_bytes = swizzle_bytes(&b, &ctx, base);
        let sgep = unsafe { b.build_gep(ctx.i8_type(), smem_a, &[sw_bytes], "").unwrap() };
        cp_async_16(&b, &ctx, &module, sgep, gep);
    }
    for chunk in 0..2u64 {
        let base = b.build_int_add(tid_x16, ci(chunk * 8), "").unwrap();
        let row = b.build_int_unsigned_div(base, ci(BN as u64), "").unwrap();
        let col = b.build_int_unsigned_rem(base, ci(BN as u64), "").unwrap();
        let grow = b.build_int_add(t, row, "").unwrap();
        let gcol = b.build_int_add(block_col, col, "").unwrap();
        let gidx = b.build_int_add(b.build_int_mul(grow, n_p, "").unwrap(), gcol, "").unwrap();
        let gep = unsafe { b.build_gep(f16_ty, b_ptr, &[gidx], "").unwrap() };
        let sw_bytes = swizzle_bytes(&b, &ctx, base);
        let sgep = unsafe { b.build_gep(ctx.i8_type(), smem_b, &[sw_bytes], "").unwrap() };
        cp_async_16(&b, &ctx, &module, sgep, gep);
    }
    cp_async_commit(&b, &ctx, &module);
    cp_async_wait_group(&b, &ctx, &module, 0);
    call_barrier0(&ctx, &module, &b);

    // ── 2 K-iterations, each with ldmatrix (A) + ldmatrix.trans (B) + MMA ──
    let mut cur: Vec<FloatValue> = (0..num_acc).map(|i| acc_phis[i].as_basic_value().into_float_value()).collect();

    let lane_mod16 = b.build_int_unsigned_rem(lane, ci(16), "lm16").unwrap();
    let lane_mod8 = b.build_int_unsigned_rem(lane, ci(8), "lm8").unwrap();

    for ki in 0..K_ITERS as u64 {
        let k_off = ci(ki * MMA_K as u64); // 0 or 16

        // Load A fragments via ldmatrix.x4 (1 per REG_M row)
        let mut a_frags: Vec<[IntValue; 4]> = Vec::new();
        for rm in 0..REG_M as u64 {
            let row = b.build_int_add(ci(rm * MMA_M as u64), lane_mod16, "").unwrap();
            let elem_off = b.build_int_add(
                b.build_int_mul(row, ci(BK as u64), "").unwrap(), k_off, "",
            ).unwrap();
            let byte_off = swizzle_bytes(&b, &ctx, elem_off);
            let frag = ldmatrix_x4(&b, &ctx, &module, smem_a, byte_off);
            a_frags.push(frag);
        }

        // Load B fragments via ldmatrix.x4.trans (1 per REG_N col)
        // For .trans with m8n8: each thread provides address of row (lane % 8)
        // in an 8×8 sub-tile. The instruction loads 8 elements per row and transposes.
        // B is [BK×BN] row-major. Sub-tile at (k_off, rn*8+wx_off):
        //   row r starts at elem: (k_off + r) * BN + (wx_off + rn*8)
        //   where r = lane % 8
        let mut b_frags: Vec<[IntValue; 4]> = Vec::new();
        for rn in 0..REG_N as u64 {
            let tile_col = b.build_int_add(wx_off, ci(rn * MMA_N as u64), "").unwrap();
            let row_in_tile = lane_mod8; // each thread loads its row
            let k_row = b.build_int_add(k_off, row_in_tile, "").unwrap();
            let elem_off = b.build_int_add(
                b.build_int_mul(k_row, ci(BN as u64), "").unwrap(), tile_col, "",
            ).unwrap();
            let byte_off = swizzle_bytes(&b, &ctx, elem_off);
            let frag = ldmatrix_x4_trans(&b, &ctx, &module, smem_b, byte_off);
            b_frags.push(frag);
        }

        // MMA: REG_M × REG_N
        for rn in 0..REG_N as u64 {
            // B frag: first 2 regs from ldmatrix.x4.trans
            let b_frag = [b_frags[rn as usize][0], b_frags[rn as usize][1]];
            for rm in 0..REG_M as u64 {
                let ai = (rm as u32 * REG_N * 4 + rn as u32 * 4) as usize;
                let [d0,d1,d2,d3] = mma_sync(&b, &ctx, &module,
                    &a_frags[rm as usize], &b_frag,
                    &[cur[ai], cur[ai+1], cur[ai+2], cur[ai+3]]);
                cur[ai]=d0; cur[ai+1]=d1; cur[ai+2]=d2; cur[ai+3]=d3;
            }
        }
    }

    call_barrier0(&ctx, &module, &b);
    let tn = b.build_int_add(t, ci(BK as u64), "").unwrap();
    b.build_unconditional_branch(kh).unwrap();

    t_phi.add_incoming(&[(&ci(0), entry), (&tn, kb)]);
    for i in 0..num_acc { acc_phis[i].add_incoming(&[(&f0, entry), (&cur[i], kb)]); }

    // ── Store C ──
    b.position_at_end(ke);
    for rm in 0..REG_M {
        for rn in 0..REG_N {
            let ai = (rm * REG_N * 4 + rn * 4) as usize;
            for d in 0..4u32 {
                let mro = if d < 2 { group } else { b.build_int_add(group, ci(8), "").unwrap() };
                let mco = if d % 2 == 0 { tg2 } else { b.build_int_add(tg2, ci(1), "").unwrap() };
                let cr = b.build_int_add(
                    b.build_int_add(block_row, ci(rm as u64 * MMA_M as u64), "").unwrap(), mro, "").unwrap();
                let cc = b.build_int_add(
                    b.build_int_add(
                        b.build_int_add(block_col, wx_off, "").unwrap(),
                        ci(rn as u64 * MMA_N as u64), "").unwrap(),
                    mco, "").unwrap();
                let idx = b.build_int_add(b.build_int_mul(cr, n_p, "").unwrap(), cc, "").unwrap();
                let gep = unsafe { b.build_gep(f32_ty, c_ptr, &[idx], "").unwrap() };
                b.build_store(gep, acc_phis[ai + d as usize].as_basic_value().into_float_value()).unwrap();
            }
        }
    }
    b.build_return(None).unwrap();

    let buf = machine.write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(std::str::from_utf8(buf.as_slice())?.to_string())
}
