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

// 64×64 BK=32, 1×4 warp layout
const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 32;
const WM: u32 = 64;
const WN: u32 = 16;
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const MMA_K: u32 = 16;
const REG_M: u32 = WM / MMA_M; // 2
const REG_N: u32 = WN / MMA_N; // 4
const K_ITERS: u32 = BK / MMA_K; // 2
const WARPS: u32 = (BM / WM) * (BN / WN); // 2×2 = 4
const THREADS: u32 = WARPS * 32;

const SWIZZLE_MASK: u32 = 0x380;
const SWIZZLE_SHIFT: u32 = 3;

pub fn run(sm: &str) -> Result<()> {
    for &sz in &[1024u32, 4096] {
        println!("  --- {}×{} ---", sz, sz);
        run_size(sm, sz, sz, sz)?;
    }
    Ok(())
}

fn run_size(sm: &str, m: u32, n: u32, k: u32) -> Result<()> {

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
    // Query actual register usage
    use cudarc::driver::sys::CUfunction_attribute as FA;
    let nregs = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_NUM_REGS)? };
    let local_bytes = unsafe { cuda::function::get_function_attribute(func, FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)? };
    println!("  [cuda] Loaded -- {} regs/thread, {} bytes local (spills)", nregs, local_bytes);

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
    let smem_bytes: c_uint = (BM * BK + BK * BN) * 2 * 2; // double-buffered
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

// -- Swizzle (returns byte offset, no division) --
fn swizzle_bytes<'ctx>(b: &Builder<'ctx>, ctx: &'ctx LlvmContext, elem_idx: IntValue<'ctx>) -> IntValue<'ctx> {
    let ci = |v: u64| ctx.i32_type().const_int(v, false);
    // elem_idx * 2 -> byte offset, apply XOR swizzle, return bytes
    let byte = b.build_left_shift(elem_idx, ci(1), "").unwrap(); // ×2 via shift
    let masked = b.build_and(byte, ci(SWIZZLE_MASK as u64), "").unwrap();
    let shifted = b.build_right_shift(masked, ci(SWIZZLE_SHIFT as u64), false, "").unwrap();
    b.build_xor(byte, shifted, "").unwrap() // returns byte offset
}

// -- ldmatrix.sync.aligned.m8n8.x4.shared.b16 --
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

        // GEP to the right byte, then ptrtoint -> 32-bit with short-ptr
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

// -- ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 (for B fragments) --
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

// -- cp.async with predicated src-size (4th operand) --
// src_size = 16 to copy, 0 for noop (fills shared with zeros)
fn cp_async_16_pred<'ctx>(
    b: &Builder<'ctx>, ctx: &'ctx LlvmContext, module: &Module<'ctx>,
    dst: inkwell::values::PointerValue<'ctx>, src: inkwell::values::PointerValue<'ctx>,
    src_size: IntValue<'ctx>,
) {
    let i32_ty = ctx.i32_type();
    let i64_ty = ctx.i64_type();
    unsafe {
        let mr = module.as_mut_ptr();
        let cr = llvm_sys::core::LLVMGetModuleContext(mr);
        let br = b.as_mut_ptr();
        let vt = llvm_sys::core::LLVMVoidTypeInContext(cr);
        let mut pt = [i32_ty.as_type_ref(), i64_ty.as_type_ref(), i32_ty.as_type_ref()];
        let ft = llvm_sys::core::LLVMFunctionType(vt, pt.as_mut_ptr(), 3, 0);
        let di = b.build_ptr_to_int(dst, i32_ty, "").unwrap();
        let si = b.build_ptr_to_int(src, i64_ty, "").unwrap();
        let s = b"cp.async.cg.shared.global [$0], [$1], 16, $2;\0";
        let c = b"r,l,r\0";
        let av = llvm_sys::core::LLVMGetInlineAsm(ft,
            s.as_ptr() as *const _, s.len()-1, c.as_ptr() as *const _, c.len()-1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0);
        let mut args = [di.as_value_ref(), si.as_value_ref(), src_size.as_value_ref()];
        llvm_sys::core::LLVMBuildCall2(br, ft, av, args.as_mut_ptr(), 3, b"\0".as_ptr() as *const _);
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

fn emit_ptx(_sm: &str) -> Result<String> {
    Ok(emit_ptx_handwritten())
}

/// Hand-crafted PTX GEMM kernel matching Triton's K-loop structure.
///
/// Layout (matching Triton exactly):
///   - Shared memory: buf0_A [0..4095], buf1_A [4096..8191],
///                    buf0_B [8192..12287], buf1_B [12288..16383]
///   - 128 threads, 4 warps, BM=BN=64, BK=32
///   - 2 cp.async groups in prologue, wait_group 2 in loop
///   - Same swizzle, ldmatrix, mma ordering as Triton
///   - Output is f32 (not f16 like Triton) for verification compatibility
///
/// Params: (ptr A, ptr B, ptr C, M, N, K)
///   A is MxK row-major f16, B is KxN row-major f16, C is MxN row-major f32
fn emit_ptx_handwritten() -> String {
    // Shared memory: A tile = 64*32*2 = 4096 bytes, B tile = 32*64*2 = 4096 bytes
    // Double-buffered: A0[0..4095] A1[4096..8191] B0[8192..12287] B1[12288..16383]
    // Total: 16384 bytes
    //
    // cp.async: each thread loads 16 bytes (8 f16 elements), 2 chunks per matrix
    //   => 128 threads * 16 bytes * 2 chunks = 4096 bytes per matrix per tile
    //
    // Swizzle for cp.async store: byte_offset = elem_idx * 2; byte_offset ^= (byte_offset & 0x380) >> 3
    //   Same as Triton's: shl tid,4 -> and 2032; shl tid,1 -> and 48/112 -> xor
    //
    // ldmatrix A (non-transposed): 4 loads of m8n8.x4 per K-iter
    //   swizzle on (row*BK + k_off) elem index
    // ldmatrix.trans B: 4 loads per K-iter, different swizzle pattern
    //
    // mma: 16 per K-iter (2 A-halves * 4 B-columns), 32 total for BK=32

    format!(r#"
.version 8.7
.target sm_89
.address_size 64

.extern .shared .align 128 .b8 global_smem[];

.visible .entry triton_style_gemm(
    .param .u64 .ptr .global .align 2 triton_style_gemm_param_0,
    .param .u64 .ptr .global .align 2 triton_style_gemm_param_1,
    .param .u64 .ptr .global .align 4 triton_style_gemm_param_2,
    .param .u32 triton_style_gemm_param_3,
    .param .u32 triton_style_gemm_param_4,
    .param .u32 triton_style_gemm_param_5
)
.reqntid 128
{{
    // Register declarations -- match Triton's style
    .reg .pred  %p<10>;
    .reg .b32   %r<256>;
    .reg .b64   %rd<51>;

    // ===============================================================
    // Load parameters
    // ===============================================================
    ld.param.b64    %rd1, [triton_style_gemm_param_0];   // A ptr
    ld.param.b64    %rd2, [triton_style_gemm_param_1];   // B ptr
    ld.param.b64    %rd3, [triton_style_gemm_param_2];   // C ptr
    // param_3 = M (unused for address computation when grid covers M)
    ld.param.b32    %r1, [triton_style_gemm_param_4];    // N
    ld.param.b32    %r2, [triton_style_gemm_param_5];    // K

    // ===============================================================
    // Thread/block indices
    // ===============================================================
    mov.u32         %r3, %ctaid.x;          // block_x (N dimension)
    mov.u32         %r4, %ctaid.y;          // block_y (M dimension)
    mov.u32         %r5, %tid.x;            // tid [0..127]

    // block_row = block_y * 64, block_col = block_x * 64
    shl.b32         %r6, %r4, 6;            // block_row
    shl.b32         %r7, %r3, 6;            // block_col

    // Warp and lane decomposition
    // warp_id = tid >> 5, lane = tid & 31
    bfe.u32         %r8, %r5, 5, 2;         // warp_id = bits[6:5] of tid (0..3)
    and.b32         %r9, %r5, 31;           // lane

    // group = lane / 4 [0..7], tg = lane % 4 [0..3]
    shr.u32         %r10, %r9, 2;           // group = lane / 4 [0..7]
    and.b32         %r11, %r9, 3;           // tg = lane % 4

    // ===============================================================
    // cp.async store address computation (swizzled shared memory)
    // Each thread stores 16 bytes (8 f16 elements), 2 chunks for A, 2 for B
    //
    // For A: base_elem = tid * 16 + chunk * 8
    //        row = base_elem / 32 (BK=32), col = base_elem % 32
    //        byte_offset = base_elem * 2
    //        swizzled = byte_offset ^ ((byte_offset & 0x380) >> 3)
    //
    // For B: base_elem = tid * 16 + chunk * 8
    //        row = base_elem / 64 (BN=64), col = base_elem % 64
    //        byte_offset = base_elem * 2
    //        swizzled = byte_offset ^ ((byte_offset & 0x380) >> 3)
    // ===============================================================

    // Triton-style swizzle for cp.async:
    // %r48 = (tid << 4) & 2032   (byte offset high bits)
    // For A chunk0: xor with (tid << 1) & 48
    // For A chunk1: xor with (tid << 1) & 48, then add 2048 (chunk offset)
    // For B: offset by 8192, similar swizzle with & 112

    shl.b32         %r12, %r5, 4;           // tid * 16
    and.b32         %r13, %r12, 2032;       // (tid<<4) & 0x7F0 = byte offset high bits
    shl.b32         %r14, %r5, 1;           // tid * 2
    and.b32         %r15, %r14, 48;         // (tid<<1) & 48 -- for A swizzle row 0..15
    xor.b32         %r16, %r13, %r15;       // A cp.async swizzled offset (chunk 0)
    mov.b32         %r17, global_smem;
    add.s32         %r18, %r17, %r16;       // smem addr for A chunk 0

    // A chunk 1 is +2048 bytes from chunk 0 (next 32 rows of A)
    add.s32         %r19, %r18, 2048;       // smem addr for A chunk 1

    // B swizzle: same byte offset, but XOR with (tid<<1) & 112
    and.b32         %r20, %r14, 112;        // (tid<<1) & 112 -- for B swizzle
    xor.b32         %r21, %r13, %r20;       // B cp.async swizzled offset
    add.s32         %r22, %r17, %r21;
    add.s32         %r23, %r22, 8192;       // smem addr for B chunk 0 (B starts at 8192)
    add.s32         %r24, %r22, 10240;      // smem addr for B chunk 1 (+2048 from B start)

    // ===============================================================
    // Global addresses for cp.async
    // A[row, col] where row = block_row + (tid*16 + chunk*8)/32
    //                    col = (tid*16 + chunk*8) % 32
    // Triton uses stride params; we compute from K directly.
    //
    // A is row-major MxK: A[r,c] = A + (r*K + c) * 2
    // B is row-major KxN: B[r,c] = B + (r*N + c) * 2
    // ===============================================================

    // A cp.async mapping: element_idx = tid * 8
    //   row = tid / 4 (0..31), col = (tid % 4) * 8 (0,8,16,24)
    //   Each cp.async loads 16 bytes = 8 f16 elements
    //   128 threads cover 32 rows * 4 cols_of_8 = 32 * 32 = 1024 elements (half of A tile)
    //   Chunk 1 at smem +2048 covers rows 32-63 (same col pattern)
    and.b32         %r25, %r5, 3;           // tid & 3
    shl.b32         %r26, %r25, 3;          // (tid & 3) * 8 = col for A
    shr.u32         %r27, %r5, 2;           // tid / 4 = row offset (0..31) for A chunk 0
    add.s32         %r28, %r6, %r27;        // grow_a0 = block_row + tid/4
    add.s32         %r29, %r28, 32;         // grow_a1 = grow_a0 + 32 (chunk 1)

    // Global byte address for A chunk 0:
    // addr = A_ptr + (grow_a0 * K + gcol_a0) * 2
    mul.lo.s32      %r30, %r28, %r2;        // grow_a0 * K
    add.s32         %r31, %r30, %r26;       // grow_a0 * K + gcol_a0
    mad.wide.s32    %rd4, %r31, 2, %rd1;    // A + offset * 2

    // Global byte address for A chunk 1:
    mul.lo.s32      %r32, %r29, %r2;        // grow_a1 * K
    add.s32         %r33, %r32, %r26;       // grow_a1 * K + gcol_a0
    mad.wide.s32    %rd5, %r33, 2, %rd1;    // A + offset * 2

    // B cp.async mapping: element_idx = tid * 8
    //   row = tid / 8 (0..15), col = (tid % 8) * 8 (0,8,16,24,32,40,48,56)
    //   128 threads cover 16 rows * 8 cols_of_8 = 16 * 64 = 1024 elements (half of B tile)
    //   Chunk 1 at smem +2048 covers rows 16-31
    and.b32         %r34, %r5, 7;           // tid & 7
    shl.b32         %r35, %r34, 3;          // (tid & 7) * 8 = col for B
    shr.u32         %r36, %r5, 3;           // tid / 8 = row offset (0..15) for B chunk 0
    add.s32         %r37, %r36, 0;          // B row for chunk 0 (relative to K tile start)
    add.s32         %r38, %r37, 16;         // B row for chunk 1

    // gcol_b = block_col + (tid % 8) * 8
    add.s32         %r39, %r7, %r35;        // global col for B

    // Global byte address for B chunk 0:
    // addr = B_ptr + (row_b0 * N + gcol_b) * 2
    mul.lo.s32      %r40, %r37, %r1;        // row_b0 * N
    add.s32         %r41, %r40, %r39;       // row_b0 * N + gcol_b
    mad.wide.s32    %rd6, %r41, 2, %rd2;    // B + offset * 2

    // Global byte address for B chunk 1:
    mul.lo.s32      %r42, %r38, %r1;        // row_b1 * N
    add.s32         %r43, %r42, %r39;       // row_b1 * N + gcol_b
    mad.wide.s32    %rd7, %r43, 2, %rd2;    // B + offset * 2

    // ===============================================================
    // Prologue: load tile 0 into buffer 0
    // ===============================================================
    setp.gt.s32     %p1, %r2, 0;            // K > 0?
    selp.b32        %r44, 16, 0, %p1;       // src_size for group 0

    cp.async.cg.shared.global [ %r18 + 0 ], [ %rd4 + 0 ], 0x10, %r44;
    cp.async.cg.shared.global [ %r19 + 0 ], [ %rd5 + 0 ], 0x10, %r44;
    cp.async.commit_group;

    cp.async.cg.shared.global [ %r23 + 0 ], [ %rd6 + 0 ], 0x10, %r44;
    cp.async.cg.shared.global [ %r24 + 0 ], [ %rd7 + 0 ], 0x10, %r44;
    cp.async.commit_group;

    // ===============================================================
    // Advance global pointers by BK=32 for tile 1
    // A: advance col by 32 -> +32*2=64 bytes per element row, but stride is K
    //    So advance = 32 elements = 64 bytes along K dimension
    // B: advance row by 32 -> +32*N elements = 32*N*2 bytes
    // ===============================================================
    // A advance: new_addr = old_addr + 32 * 2 = +64 bytes (K dim, contiguous in row-major)
    add.s64         %rd8, %rd4, 64;         // A chunk 0, tile 1
    add.s64         %rd9, %rd5, 64;         // A chunk 1, tile 1

    // B advance: new_addr = old_addr + 32 * N * 2 bytes
    shl.b32         %r45, %r1, 5;           // N * 32
    mul.wide.s32    %rd10, %r45, 2;         // N * 32 * 2 bytes
    add.s64         %rd11, %rd6, %rd10;     // B chunk 0, tile 1
    add.s64         %rd12, %rd7, %rd10;     // B chunk 1, tile 1

    // ===============================================================
    // Prologue: load tile 1 into buffer 1
    // Buffer 1: A at +4096, B at +12288
    // ===============================================================
    setp.gt.s32     %p2, %r2, 32;           // K > 32?
    bar.sync        0;
    add.s32         %r46, %r18, 4096;       // buf1 A chunk 0
    add.s32         %r47, %r19, 4096;       // buf1 A chunk 1
    selp.b32        %r48, 16, 0, %p2;       // src_size for group 1

    cp.async.cg.shared.global [ %r46 + 0 ], [ %rd8 + 0 ], 0x10, %r48;
    cp.async.cg.shared.global [ %r47 + 0 ], [ %rd9 + 0 ], 0x10, %r48;
    cp.async.commit_group;

    add.s32         %r49, %r22, 12288;      // buf1 B chunk 0
    add.s32         %r50, %r22, 14336;      // buf1 B chunk 1

    cp.async.cg.shared.global [ %r49 + 0 ], [ %rd11 + 0 ], 0x10, %r48;
    cp.async.cg.shared.global [ %r50 + 0 ], [ %rd12 + 0 ], 0x10, %r48;
    cp.async.commit_group;

    // ===============================================================
    // Precompute ldmatrix addresses (relative to buffer base)
    //
    // Triton's approach: compute swizzled offsets once, then add buf_base each iter
    //
    // A ldmatrix (non-transposed, m8n8.x4):
    //   Each warp loads a 16x16 tile. With 1x4 warp layout and WM=64:
    //   rm=0: rows [0..15], rm=1: rows [32..47] (with +2048 offset like Triton)
    //   Wait -- Triton loads from [%r110] and [%r110+2048] for the two A halves.
    //   The swizzled offset %r16 (Triton) encodes the row+col swizzle.
    //   Let me match Triton exactly.
    //
    // Triton A ldmatrix swizzle (from the K-loop body):
    //   %r53 = (tid<<6) & 960    -- row * 16 part
    //   %r54 = (tid<<3) & 48     -- within-warp row offset
    //   %r55 = tid & 16          -- K-iter toggle
    //   %r56 = (tid<<4) & 1024   -- second half flag
    //   %r57 = %r53 | %r54
    //   %r58 = %r57 ^ %r55
    //   %r16 = %r58 | %r56       -- for ki=0
    //   %r17 = %r16 ^ 32         -- for ki=1
    //   Two ldmatrix from [buf_base + %r16] and [buf_base + %r16 + 2048]
    //   And [buf_base + %r17] and [buf_base + %r17 + 2048]
    // ===============================================================

    shl.b32         %r51, %r5, 6;           // tid << 6

    // A ldmatrix swizzle offsets (matching Triton exactly)
    and.b32         %r52, %r51, 960;        // (tid<<6) & 960
    shl.b32         %r53, %r5, 3;           // tid << 3
    and.b32         %r54, %r53, 48;         // (tid<<3) & 48
    and.b32         %r55, %r5, 16;          // tid & 16
    and.b32         %r56, %r12, 1024;       // (tid<<4) & 1024
    or.b32          %r57, %r52, %r54;       // combine row parts
    xor.b32         %r58, %r57, %r55;       // XOR swizzle
    or.b32          %r59, %r58, %r56;       // A offset for ki=0 (= Triton's %r16)
    xor.b32         %r60, %r59, 32;         // A offset for ki=1 (= Triton's %r17)

    // B ldmatrix.trans swizzle offsets (matching Triton exactly)
    // Triton:
    //   %r59 = (tid<<7) & 3968
    //   %r61 = (lane&7) << 4      -- col part
    //   %r62 = (tid>>1) & 16      -- swizzle bit
    //   %r64 = %r61 ^ %r63
    //   %r18 = %r64 | %r60        -- for rn=0
    //   %r19 = %r18 ^ 32          -- for rn=1
    //   %r20 = %r18 ^ 64          -- for rn=2
    //   %r21 = %r18 ^ 96          -- for rn=3

    shl.b32         %r61, %r5, 7;           // tid << 7
    and.b32         %r62, %r61, 3968;       // (tid<<7) & 3968 -- row part
    and.b32         %r63, %r9, 7;           // lane & 7
    shl.b32         %r64, %r63, 4;          // (lane&7) << 4 -- col part
    shr.u32         %r65, %r5, 1;           // tid >> 1
    and.b32         %r66, %r65, 16;         // (tid>>1) & 16 -- swizzle bit
    xor.b32         %r67, %r64, %r66;       // col ^ swizzle
    or.b32          %r68, %r67, %r62;       // B offset for rn=0 (= Triton's %r18)
    xor.b32         %r69, %r68, 32;         // B offset for rn=1
    xor.b32         %r70, %r68, 64;         // B offset for rn=2
    xor.b32         %r71, %r68, 96;         // B offset for rn=3

    // ===============================================================
    // Advance global pointers to tile 2 position (for loop's cp.async)
    // ===============================================================
    add.s64         %rd13, %rd8, 64;        // A chunk 0, tile 2
    add.s64         %rd14, %rd9, 64;        // A chunk 1, tile 2
    add.s64         %rd15, %rd11, %rd10;    // B chunk 0, tile 2
    add.s64         %rd16, %rd12, %rd10;    // B chunk 1, tile 2

    // K loop limit check
    @%p1 bra        $L_LOOP_ENTRY;
    bra.uni         $L_K0_FALLTHROUGH;

$L_LOOP_ENTRY:
    // ===============================================================
    // Precompute stride for advancing global A pointer in loop
    // A advances by BK=32 cols -> +64 bytes (already computed as constant)
    // B advances by BK=32 rows -> +32*N*2 bytes (already in %rd10)
    //
    // Also compute:
    //   K - 64 for predication of cp.async in loop body
    //   (loop loads tile t+2, valid when t < K-64)
    // ===============================================================
    add.s32         %r72, %r2, -64;         // K - 64

    // Pre-compute global addresses for epilogue C store
    // (done after loop, but we need block offsets which don't change)
    // row offsets for mma: group (lane/4), group+8
    // col offsets for mma: tg*2, tg*2+1

    // ===============================================================
    // Initialize accumulators to 0.0f
    // 32 accumulators: acc0..acc31
    // Matching Triton's %r172..%r203
    // ===============================================================
    mov.b32         %r100, 0x00000000;      // 0.0f
    mov.b32         %r101, %r100;
    mov.b32         %r102, %r100;
    mov.b32         %r103, %r100;
    mov.b32         %r104, %r100;
    mov.b32         %r105, %r100;
    mov.b32         %r106, %r100;
    mov.b32         %r107, %r100;
    mov.b32         %r108, %r100;
    mov.b32         %r109, %r100;
    mov.b32         %r110, %r100;
    mov.b32         %r111, %r100;
    mov.b32         %r112, %r100;
    mov.b32         %r113, %r100;
    mov.b32         %r114, %r100;
    mov.b32         %r115, %r100;
    mov.b32         %r116, %r100;
    mov.b32         %r117, %r100;
    mov.b32         %r118, %r100;
    mov.b32         %r119, %r100;
    mov.b32         %r120, %r100;
    mov.b32         %r121, %r100;
    mov.b32         %r122, %r100;
    mov.b32         %r123, %r100;
    mov.b32         %r124, %r100;
    mov.b32         %r125, %r100;
    mov.b32         %r126, %r100;
    mov.b32         %r127, %r100;
    mov.b32         %r128, %r100;
    mov.b32         %r129, %r100;
    mov.b32         %r130, %r100;
    mov.b32         %r131, %r100;

    // Buffer toggle counter (like Triton's %r170/%r171)
    mov.b32         %r73, 1;               // read buf counter (starts at -1+1=0 on first toggle)
    mov.b32         %r74, -1;              // write buf counter (matches Triton's init)

    // K loop counter
    mov.b32         %r75, 0;               // t = 0

    // ===============================================================
    // K-LOOP: This is the hot loop. Matches Triton's instruction ordering.
    //
    // Structure per iteration:
    //   1. Check if more iterations remain for cp.async predication
    //   2. Toggle read buffer, wait_group 2, barrier
    //   3. ldmatrix A (4 loads: 2 for ki=0 + 2 for ki=1, each split by +2048)
    //   4. ldmatrix.trans B (4 loads for 4 N-columns, each gives 4 regs)
    //   5. 16 mma instructions (ki=0: 8 mma, ki=1: 8 mma)
    //   6. Advance B global pointer
    //   7. Toggle write buffer, barrier
    //   8. Predicated cp.async for next-next tile
    //   9. Advance loop counter, branch
    // ===============================================================

$L_KLOOP:
    // Check if we should load more tiles (predication for cp.async)
    setp.lt.s32     %p3, %r75, %r72;       // t < K - 64?

    // Toggle read buffer: %r74 cycles 0->1->0->1
    add.s32         %r76, %r74, 1;
    setp.gt.s32     %p4, %r76, 1;
    selp.b32        %r74, 0, %r76, %p4;

    // -- wait_group 2 + barrier --
    cp.async.wait_group     2;
    bar.sync        0;

    // Compute read buffer base: buf_id * 4096
    shl.b32         %r77, %r74, 12;         // buf_id * 4096
    add.s32         %r78, %r17, %r77;       // smem_base + buf_offset

    // ===============================================================
    // ldmatrix A -- 4 loads matching Triton exactly
    // Load from buf_base + A_swizzle_offset
    //   ki=0: [base + %r59] and [base + %r59 + 2048]
    //   ki=1: [base + %r60] and [base + %r60 + 2048]
    // Each ldmatrix.x4 gives 4 registers
    // ===============================================================
    add.s32         %r79, %r78, %r59;       // ki=0, rm=0
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r132, %r133, %r134, %r135}}, [%r79];
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r136, %r137, %r138, %r139}}, [%r79+2048];

    add.s32         %r80, %r78, %r60;       // ki=1, rm=0
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r140, %r141, %r142, %r143}}, [%r80];
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r144, %r145, %r146, %r147}}, [%r80+2048];

    // ===============================================================
    // ldmatrix.trans B -- 4 loads matching Triton exactly
    // B is at buf_base + 8192 + B_swizzle_offset
    // Each load gives 4 regs: [b0_ki0, b1_ki0, b0_ki1, b1_ki1]
    // ===============================================================
    add.s32         %r81, %r78, %r68;       // rn=0
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r148, %r149, %r150, %r151}}, [%r81+8192];
    add.s32         %r82, %r78, %r69;       // rn=1
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r152, %r153, %r154, %r155}}, [%r82+8192];
    add.s32         %r83, %r78, %r70;       // rn=2
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r156, %r157, %r158, %r159}}, [%r83+8192];
    add.s32         %r84, %r78, %r71;       // rn=3
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%r160, %r161, %r162, %r163}}, [%r84+8192];

    // ===============================================================
    // MMA instructions -- 16 total, matching Triton's exact ordering
    //
    // Accumulator layout (32 f32 regs, %r100..%r131):
    //   rm=0, rn=0: %r100..%r103  (A[ki0,rm0] x B[ki0,rn0])
    //   rm=0, rn=1: %r104..%r107
    //   rm=0, rn=2: %r108..%r111
    //   rm=0, rn=3: %r112..%r115
    //   rm=1, rn=0: %r116..%r119  (A[ki0,rm1] x B[ki0,rn0])
    //   rm=1, rn=1: %r120..%r123
    //   rm=1, rn=2: %r124..%r127
    //   rm=1, rn=3: %r128..%r131
    //
    // Triton order: ki=0 first (rm0 then rm1 for each rn), then ki=1
    //   ki=0: A_rm0 x B_rn0, A_rm0 x B_rn1, A_rm0 x B_rn2, A_rm0 x B_rn3,
    //          A_rm1 x B_rn0, A_rm1 x B_rn1, A_rm1 x B_rn2, A_rm1 x B_rn3
    //   ki=1: same pattern with ki=1 A and B fragments
    // ===============================================================

    // -- ki=0: A_rm0 x B_rn* --
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r100, %r101, %r102, %r103 }},
        {{ %r132, %r133, %r134, %r135 }},
        {{ %r148, %r149 }},
        {{ %r100, %r101, %r102, %r103 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r104, %r105, %r106, %r107 }},
        {{ %r132, %r133, %r134, %r135 }},
        {{ %r152, %r153 }},
        {{ %r104, %r105, %r106, %r107 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r108, %r109, %r110, %r111 }},
        {{ %r132, %r133, %r134, %r135 }},
        {{ %r156, %r157 }},
        {{ %r108, %r109, %r110, %r111 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r112, %r113, %r114, %r115 }},
        {{ %r132, %r133, %r134, %r135 }},
        {{ %r160, %r161 }},
        {{ %r112, %r113, %r114, %r115 }};

    // -- ki=0: A_rm1 x B_rn* --
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r116, %r117, %r118, %r119 }},
        {{ %r136, %r137, %r138, %r139 }},
        {{ %r148, %r149 }},
        {{ %r116, %r117, %r118, %r119 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r120, %r121, %r122, %r123 }},
        {{ %r136, %r137, %r138, %r139 }},
        {{ %r152, %r153 }},
        {{ %r120, %r121, %r122, %r123 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r124, %r125, %r126, %r127 }},
        {{ %r136, %r137, %r138, %r139 }},
        {{ %r156, %r157 }},
        {{ %r124, %r125, %r126, %r127 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r128, %r129, %r130, %r131 }},
        {{ %r136, %r137, %r138, %r139 }},
        {{ %r160, %r161 }},
        {{ %r128, %r129, %r130, %r131 }};

    // -- ki=1: A_rm0 x B_rn* --
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r100, %r101, %r102, %r103 }},
        {{ %r140, %r141, %r142, %r143 }},
        {{ %r150, %r151 }},
        {{ %r100, %r101, %r102, %r103 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r104, %r105, %r106, %r107 }},
        {{ %r140, %r141, %r142, %r143 }},
        {{ %r154, %r155 }},
        {{ %r104, %r105, %r106, %r107 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r108, %r109, %r110, %r111 }},
        {{ %r140, %r141, %r142, %r143 }},
        {{ %r158, %r159 }},
        {{ %r108, %r109, %r110, %r111 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r112, %r113, %r114, %r115 }},
        {{ %r140, %r141, %r142, %r143 }},
        {{ %r162, %r163 }},
        {{ %r112, %r113, %r114, %r115 }};

    // -- ki=1: A_rm1 x B_rn* --
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r116, %r117, %r118, %r119 }},
        {{ %r144, %r145, %r146, %r147 }},
        {{ %r150, %r151 }},
        {{ %r116, %r117, %r118, %r119 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r120, %r121, %r122, %r123 }},
        {{ %r144, %r145, %r146, %r147 }},
        {{ %r154, %r155 }},
        {{ %r120, %r121, %r122, %r123 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r124, %r125, %r126, %r127 }},
        {{ %r144, %r145, %r146, %r147 }},
        {{ %r158, %r159 }},
        {{ %r124, %r125, %r126, %r127 }};

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %r128, %r129, %r130, %r131 }},
        {{ %r144, %r145, %r146, %r147 }},
        {{ %r162, %r163 }},
        {{ %r128, %r129, %r130, %r131 }};

    // ===============================================================
    // Advance B global pointers
    // ===============================================================
    add.s64         %rd17, %rd15, %rd10;    // B chunk 0, next-next tile (for cp.async)
    add.s64         %rd18, %rd16, %rd10;    // B chunk 1, next-next tile

    // ===============================================================
    // Toggle write buffer, barrier, predicated cp.async
    // ===============================================================
    add.s32         %r85, %r73, 1;
    setp.gt.s32     %p5, %r85, 1;
    selp.b32        %r73, 0, %r85, %p5;

    shl.b32         %r86, %r73, 12;         // write buf * 4096
    add.s32         %r87, %r17, %r86;       // smem_base + write_buf_offset

    bar.sync        0;

    add.s32         %r88, %r87, %r16;       // A cp.async dest in write buffer
    selp.b32        %r89, 16, 0, %p3;       // src_size (0 if no more tiles)
    cp.async.cg.shared.global [ %r88 + 0 ], [ %rd13 + 0 ], 0x10, %r89;
    add.s32         %r90, %r88, 2048;       // A chunk 1
    cp.async.cg.shared.global [ %r90 + 0 ], [ %rd14 + 0 ], 0x10, %r89;
    cp.async.commit_group;

    add.s32         %r91, %r87, %r21;       // B swizzle offset in write buffer
    add.s32         %r92, %r91, 8192;       // B cp.async dest chunk 0
    cp.async.cg.shared.global [ %r92 + 0 ], [ %rd15 + 0 ], 0x10, %r89;
    add.s32         %r93, %r91, 10240;      // B chunk 1
    cp.async.cg.shared.global [ %r93 + 0 ], [ %rd16 + 0 ], 0x10, %r89;
    cp.async.commit_group;

    // ===============================================================
    // Advance loop state
    // ===============================================================
    add.s32         %r75, %r75, 32;         // t += BK
    add.s64         %rd13, %rd13, 64;       // A chunk 0 advance
    add.s64         %rd14, %rd14, 64;       // A chunk 1 advance
    mov.b64         %rd15, %rd17;           // B chunk 0 = pre-advanced
    mov.b64         %rd16, %rd18;           // B chunk 1 = pre-advanced
    setp.lt.s32     %p6, %r75, %r2;        // t < K?
    @%p6 bra        $L_KLOOP;

    bra.uni         $L_EPILOGUE;

$L_K0_FALLTHROUGH:
    // K <= 0: zero accumulators, skip to store
    mov.b32         %r100, 0x00000000;
    mov.b32         %r101, %r100; mov.b32 %r102, %r100; mov.b32 %r103, %r100;
    mov.b32         %r104, %r100; mov.b32 %r105, %r100; mov.b32 %r106, %r100;
    mov.b32         %r107, %r100; mov.b32 %r108, %r100; mov.b32 %r109, %r100;
    mov.b32         %r110, %r100; mov.b32 %r111, %r100; mov.b32 %r112, %r100;
    mov.b32         %r113, %r100; mov.b32 %r114, %r100; mov.b32 %r115, %r100;
    mov.b32         %r116, %r100; mov.b32 %r117, %r100; mov.b32 %r118, %r100;
    mov.b32         %r119, %r100; mov.b32 %r120, %r100; mov.b32 %r121, %r100;
    mov.b32         %r122, %r100; mov.b32 %r123, %r100; mov.b32 %r124, %r100;
    mov.b32         %r125, %r100; mov.b32 %r126, %r100; mov.b32 %r127, %r100;
    mov.b32         %r128, %r100; mov.b32 %r129, %r100; mov.b32 %r130, %r100;
    mov.b32         %r131, %r100;

$L_EPILOGUE:
    // ===============================================================
    // Epilogue: drain pipeline and store C as f32
    //
    // Wait for all async copies to complete
    // ===============================================================
    cp.async.wait_group     0;
    bar.sync        0;

    // ===============================================================
    // Store C -- f32 output (simpler than Triton's f16 shuffle)
    //
    // MMA m16n8k16 output layout per thread:
    //   d0 = C[group, tg*2]       (group = lane/4, tg = lane%4)
    //   d1 = C[group, tg*2+1]
    //   d2 = C[group+8, tg*2]
    //   d3 = C[group+8, tg*2+1]
    //
    // With our tile config (WM=64, 1x4 warp layout):
    //   rm=0: rows [0..15] (group and group+8)
    //   rm=1: rows [32..47] (group+32 and group+32+8)
    //   rn=0..3: cols [warp_id*16 + rn*8 + tg*2 + 0 or 1]
    //
    // C[r,c] stored at C_ptr + (r * N + c) * 4  (f32)
    // ===============================================================

    // Precompute row and column bases
    // group = lane / 4 = %r10
    // tg = lane % 4 = %r11
    shl.b32         %r164, %r11, 1;         // tg * 2
    add.s32         %r165, %r164, 1;        // tg * 2 + 1
    add.s32         %r166, %r10, 8;         // group + 8

    // warp_col_base = block_col + warp_id * 16
    shl.b32         %r167, %r8, 4;          // warp_id * 16
    add.s32         %r168, %r7, %r167;      // block_col + warp_id * 16

    // For each (rm, rn), store 4 f32 values
    // rm=0, rn=0: acc %r100..%r103
    //   row0 = block_row + group, row1 = block_row + group + 8
    //   col0 = warp_col_base + 0 + tg*2, col1 = col0 + 1

    // -- rm=0 rows --
    add.s32         %r169, %r6, %r10;       // row_a = block_row + group
    add.s32         %r170, %r6, %r166;      // row_b = block_row + group + 8

    // Row strides in bytes (f32): row * N * 4
    mul.lo.s32      %r171, %r169, %r1;      // row_a * N
    mul.lo.s32      %r172, %r170, %r1;      // row_b * N

    // rm=0, rn=0
    add.s32         %r173, %r168, %r164;    // col = warp_col_base + tg*2
    add.s32         %r174, %r171, %r173;    // row_a * N + col
    add.s32         %r175, %r172, %r173;    // row_b * N + col
    add.s32         %r176, %r174, 1;        // row_a * N + col + 1
    add.s32         %r177, %r175, 1;        // row_b * N + col + 1
    mad.wide.s32    %rd19, %r174, 4, %rd3;
    mad.wide.s32    %rd20, %r176, 4, %rd3;
    mad.wide.s32    %rd21, %r175, 4, %rd3;
    mad.wide.s32    %rd22, %r177, 4, %rd3;
    st.global.b32   [%rd19], %r100;
    st.global.b32   [%rd20], %r101;
    st.global.b32   [%rd21], %r102;
    st.global.b32   [%rd22], %r103;

    // rm=0, rn=1
    add.s32         %r178, %r168, 8;        // warp_col_base + 8
    add.s32         %r179, %r178, %r164;    // col = warp_col_base + 8 + tg*2
    add.s32         %r180, %r171, %r179;
    add.s32         %r181, %r172, %r179;
    add.s32         %r182, %r180, 1;
    add.s32         %r183, %r181, 1;
    mad.wide.s32    %rd23, %r180, 4, %rd3;
    mad.wide.s32    %rd24, %r182, 4, %rd3;
    mad.wide.s32    %rd25, %r181, 4, %rd3;
    mad.wide.s32    %rd26, %r183, 4, %rd3;
    st.global.b32   [%rd23], %r104;
    st.global.b32   [%rd24], %r105;
    st.global.b32   [%rd25], %r106;
    st.global.b32   [%rd26], %r107;

    // rm=0, rn=2
    add.s32         %r184, %r168, 16;
    add.s32         %r185, %r184, %r164;
    add.s32         %r186, %r171, %r185;
    add.s32         %r187, %r172, %r185;
    add.s32         %r188, %r186, 1;
    add.s32         %r189, %r187, 1;
    mad.wide.s32    %rd27, %r186, 4, %rd3;
    mad.wide.s32    %rd28, %r188, 4, %rd3;
    mad.wide.s32    %rd29, %r187, 4, %rd3;
    mad.wide.s32    %rd30, %r189, 4, %rd3;
    st.global.b32   [%rd27], %r108;
    st.global.b32   [%rd28], %r109;
    st.global.b32   [%rd29], %r110;
    st.global.b32   [%rd30], %r111;

    // rm=0, rn=3
    add.s32         %r190, %r168, 24;
    add.s32         %r191, %r190, %r164;
    add.s32         %r192, %r171, %r191;
    add.s32         %r193, %r172, %r191;
    add.s32         %r194, %r192, 1;
    add.s32         %r195, %r193, 1;
    mad.wide.s32    %rd31, %r192, 4, %rd3;
    mad.wide.s32    %rd32, %r194, 4, %rd3;
    mad.wide.s32    %rd33, %r193, 4, %rd3;
    mad.wide.s32    %rd34, %r195, 4, %rd3;
    st.global.b32   [%rd31], %r112;
    st.global.b32   [%rd32], %r113;
    st.global.b32   [%rd33], %r114;
    st.global.b32   [%rd34], %r115;

    // -- rm=1 rows (offset by 32) --
    add.s32         %r196, %r169, 32;       // row_a + 32
    add.s32         %r197, %r170, 32;       // row_b + 32
    mul.lo.s32      %r198, %r196, %r1;
    mul.lo.s32      %r199, %r197, %r1;

    // rm=1, rn=0
    add.s32         %r200, %r198, %r173;
    add.s32         %r201, %r199, %r173;
    add.s32         %r202, %r200, 1;
    add.s32         %r203, %r201, 1;
    mad.wide.s32    %rd35, %r200, 4, %rd3;
    mad.wide.s32    %rd36, %r202, 4, %rd3;
    mad.wide.s32    %rd37, %r201, 4, %rd3;
    mad.wide.s32    %rd38, %r203, 4, %rd3;
    st.global.b32   [%rd35], %r116;
    st.global.b32   [%rd36], %r117;
    st.global.b32   [%rd37], %r118;
    st.global.b32   [%rd38], %r119;

    // rm=1, rn=1
    add.s32         %r204, %r198, %r179;
    add.s32         %r205, %r199, %r179;
    add.s32         %r206, %r204, 1;
    add.s32         %r207, %r205, 1;
    mad.wide.s32    %rd39, %r204, 4, %rd3;
    mad.wide.s32    %rd40, %r206, 4, %rd3;
    mad.wide.s32    %rd41, %r205, 4, %rd3;
    mad.wide.s32    %rd42, %r207, 4, %rd3;
    st.global.b32   [%rd39], %r120;
    st.global.b32   [%rd40], %r121;
    st.global.b32   [%rd41], %r122;
    st.global.b32   [%rd42], %r123;

    // rm=1, rn=2
    add.s32         %r208, %r198, %r185;
    add.s32         %r209, %r199, %r185;
    add.s32         %r210, %r208, 1;
    add.s32         %r211, %r209, 1;
    mad.wide.s32    %rd43, %r208, 4, %rd3;
    mad.wide.s32    %rd44, %r210, 4, %rd3;
    mad.wide.s32    %rd45, %r209, 4, %rd3;
    mad.wide.s32    %rd46, %r211, 4, %rd3;
    st.global.b32   [%rd43], %r124;
    st.global.b32   [%rd44], %r125;
    st.global.b32   [%rd45], %r126;
    st.global.b32   [%rd46], %r127;

    // rm=1, rn=3
    add.s32         %r212, %r198, %r191;
    add.s32         %r213, %r199, %r191;
    add.s32         %r214, %r212, 1;
    add.s32         %r215, %r213, 1;
    mad.wide.s32    %rd47, %r212, 4, %rd3;
    mad.wide.s32    %rd48, %r214, 4, %rd3;
    mad.wide.s32    %rd49, %r213, 4, %rd3;
    mad.wide.s32    %rd50, %r215, 4, %rd3;
    st.global.b32   [%rd47], %r128;
    st.global.b32   [%rd48], %r129;
    st.global.b32   [%rd49], %r130;
    st.global.b32   [%rd50], %r131;

    ret;
}}
"#)
}
