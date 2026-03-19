// SPDX-License-Identifier: Apache-2.0
//
// Multi-warp MMA GEMM with register tiling, vectorized loads, cp.async
// double buffering, and shared memory swizzle (B128).
//
// 4 warps (128 threads), 2×2 warp layout.
// Each warp: 32×32 of C via 2×4 register tiling of m16n8k16 MMAs.
// Block tile: 64×64. K-step: 16.
//
// Key optimizations (learned from CubeK/Burn):
// - cp.async.cg.shared.global for async global→shared loads
// - Double buffering: load buffer N+1 while computing buffer N
// - B stored non-transposed in shared memory (enables cp.async for both A and B)
// - 128-bit vectorized global loads (8 f16 per load)

use anyhow::{Context, Result, bail};
use std::ffi::{CString, c_uint, c_void};
use std::time::Instant;

use inkwell::context::Context as LlvmContext;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::targets::{TargetTriple, FileType};
use inkwell::types::AsTypeRef;
use inkwell::values::{AsValueRef, IntValue, FloatValue};
use inkwell::{AddressSpace, IntPredicate};

use cudarc::driver::result as cuda;

use crate::{create_nvptx_target_machine, add_nvvm_kernel_metadata, call_sreg, call_barrier0};

const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 16;
const WM: u32 = 64;
const WN: u32 = 64;
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const REG_M: u32 = WM / MMA_M; // 4
const REG_N: u32 = WN / MMA_N; // 8
const K_UNROLL: u32 = 1;
const WARPS: u32 = 4;
const THREADS: u32 = WARPS * 32;

// B128 swizzle parameters (from CubeK: Swizzle::new(3, 4, 3))
// offset_bytes ^= (offset_bytes & 0x380) >> 3
const SWIZZLE_MASK: u32 = 0x380; // bits [9:7]
const SWIZZLE_SHIFT: u32 = 3;

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
// PTX emission — CubeK-style: B128 swizzle, K-unroll=2, pipelined cp.async
// ===========================================================================

/// Apply B128 swizzle to a shared memory element index.
/// CubeK formula: offset_bytes ^= (offset_bytes & 0x380) >> 3
/// This XORs bits [9:7] into bits [6:4] of the byte address.
fn build_swizzle<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    elem_idx: IntValue<'ctx>,  // element index (in f16 units)
) -> IntValue<'ctx> {
    let i32_ty = context.i32_type();
    let ci = |v: u64| i32_ty.const_int(v, false);
    // Convert element index to byte offset (×2 for f16)
    let byte_off = builder.build_int_mul(elem_idx, ci(2), "").unwrap();
    // XOR: byte_off ^= (byte_off & 0x380) >> 3
    let masked = builder.build_and(byte_off, ci(SWIZZLE_MASK as u64), "").unwrap();
    let shifted = builder.build_right_shift(masked, ci(SWIZZLE_SHIFT as u64), false, "").unwrap();
    let swizzled_bytes = builder.build_xor(byte_off, shifted, "").unwrap();
    // Convert back to element index (÷2 for f16)
    builder.build_int_unsigned_div(swizzled_bytes, ci(2), "sw").unwrap()
}

fn emit_multiwarp_gemm_ptx(sm: &str) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;
    let ctx = LlvmContext::create();
    let module = ctx.create_module("ferrite_mw_gemm");
    let b = ctx.create_builder();

    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let i32_ty = ctx.i32_type();
    let f16_ty = ctx.f16_type();
    let f32_ty = ctx.f32_type();
    let void_ty = ctx.void_type();
    let ptr_g = ctx.ptr_type(AddressSpace::from(1u16));
    let ptr_s = ctx.ptr_type(AddressSpace::from(3u16));
    let v2f16_ty = f16_ty.vec_type(2);

    let smem_g = module.add_global(
        ctx.i8_type().array_type(0),
        Some(AddressSpace::from(3u16)),
        "smem",
    );
    smem_g.set_alignment(128); // Align for swizzle
    smem_g.set_externally_initialized(true);

    let fn_type = void_ty.fn_type(
        &[ptr_g.into(), ptr_g.into(), ptr_g.into(),
          i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );
    let function = module.add_function("multiwarp_gemm", fn_type, None);
    add_nvvm_kernel_metadata(&module, &function);

    let ci = |v: u64| i32_ty.const_int(v, false);

    let entry = ctx.append_basic_block(function, "entry");
    let kloop_hdr = ctx.append_basic_block(function, "kloop_hdr");
    let kloop_body = ctx.append_basic_block(function, "kloop_body");
    let kloop_exit = ctx.append_basic_block(function, "kloop_exit");

    // ── Entry ──
    b.position_at_end(entry);

    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let _m_p = function.get_nth_param(3).unwrap().into_int_value();
    let n_p = function.get_nth_param(4).unwrap().into_int_value();
    let k_p = function.get_nth_param(5).unwrap().into_int_value();

    let tid = call_sreg(&ctx, &module, &b, "llvm.nvvm.read.ptx.sreg.tid.x", "tid");
    let bid_x = call_sreg(&ctx, &module, &b, "llvm.nvvm.read.ptx.sreg.ctaid.x", "bx");
    let bid_y = call_sreg(&ctx, &module, &b, "llvm.nvvm.read.ptx.sreg.ctaid.y", "by");

    let block_row = b.build_int_mul(bid_y, ci(BM as u64), "br").unwrap();
    let block_col = b.build_int_mul(bid_x, ci(BN as u64), "bc").unwrap();

    let warp_id = b.build_int_unsigned_div(tid, ci(32), "wid").unwrap();
    let lane = b.build_int_unsigned_rem(tid, ci(32), "lane").unwrap();
    let wy = b.build_int_unsigned_div(warp_id, ci(2), "wy").unwrap();
    let wx = b.build_int_unsigned_rem(warp_id, ci(2), "wx").unwrap();
    let group = b.build_int_unsigned_div(lane, ci(4), "grp").unwrap();
    let tg = b.build_int_unsigned_rem(lane, ci(4), "tg").unwrap();
    let tg2 = b.build_int_mul(tg, ci(2), "tg2").unwrap();

    // Shared memory: single buffer with B128 swizzle
    // smem_a[BM×BK f16] + smem_b[BK×BN f16]
    let smem_base = smem_g.as_pointer_value();
    let smem_a = b.build_pointer_cast(smem_base, ptr_s, "sa").unwrap();
    let smem_b = b.build_pointer_cast(unsafe {
        b.build_gep(ctx.i8_type(), smem_base, &[ci((BM * BK * 2) as u64)], "").unwrap()
    }, ptr_s, "sb").unwrap();

    let f0 = f32_ty.const_float(0.0);
    let tid_x8 = b.build_int_mul(tid, ci(8), "tx8").unwrap();

    // Fragment index precomputation
    let wy_off = b.build_int_mul(wy, ci(WM as u64), "wyo").unwrap();
    let wx_off = b.build_int_mul(wx, ci(WN as u64), "wxo").unwrap();

    // ── Tile loading: precompute per-thread offsets ──
    // A: 128×16=2048 f16, B: 16×128=2048 f16 → 16 per thread → 2 cp.async each
    let tid_x16 = b.build_int_mul(tid, ci(16), "tx16").unwrap();
    let tid_x16_p8 = b.build_int_add(tid_x16, ci(8), "tx16p8").unwrap();
    // A chunk 0 and 1
    let a_row0 = b.build_int_unsigned_div(tid_x16, ci(BK as u64), "ar0").unwrap();
    let a_col0 = b.build_int_unsigned_rem(tid_x16, ci(BK as u64), "ac0").unwrap();
    let a_grow0 = b.build_int_add(block_row, a_row0, "agr0").unwrap();
    let a_sw0 = build_swizzle(&b, &ctx, tid_x16);
    let a_row1 = b.build_int_unsigned_div(tid_x16_p8, ci(BK as u64), "ar1").unwrap();
    let a_col1 = b.build_int_unsigned_rem(tid_x16_p8, ci(BK as u64), "ac1").unwrap();
    let a_grow1 = b.build_int_add(block_row, a_row1, "agr1").unwrap();
    let a_sw1 = build_swizzle(&b, &ctx, tid_x16_p8);
    // B chunk 0 and 1
    let b_row0 = b.build_int_unsigned_div(tid_x16, ci(BN as u64), "br0").unwrap();
    let b_col0 = b.build_int_unsigned_rem(tid_x16, ci(BN as u64), "bc0").unwrap();
    let b_gcol0 = b.build_int_add(block_col, b_col0, "bgc0").unwrap();
    let b_sw0 = build_swizzle(&b, &ctx, tid_x16);
    let b_row1 = b.build_int_unsigned_div(tid_x16_p8, ci(BN as u64), "br1").unwrap();
    let b_col1 = b.build_int_unsigned_rem(tid_x16_p8, ci(BN as u64), "bc1").unwrap();
    let b_gcol1 = b.build_int_add(block_col, b_col1, "bgc1").unwrap();
    let b_sw1 = build_swizzle(&b, &ctx, tid_x16_p8);

    b.build_unconditional_branch(kloop_hdr).unwrap();

    // ── K-loop header ──
    b.position_at_end(kloop_hdr);
    let t_phi = b.build_phi(i32_ty, "t").unwrap();
    let num_acc = (REG_M * REG_N * 4) as usize;
    let mut acc_phis = Vec::new();
    for i in 0..num_acc {
        acc_phis.push(b.build_phi(f32_ty, &format!("acc{i}")).unwrap());
    }
    let t = t_phi.as_basic_value().into_int_value();
    let kcmp = b.build_int_compare(IntPredicate::ULT, t, k_p, "kcmp").unwrap();
    b.build_conditional_branch(kcmp, kloop_body, kloop_exit).unwrap();

    // ── K-loop body ──
    b.position_at_end(kloop_body);

    // ── cp.async load A tile [BM×BK] — 2 cp.async per thread ──
    let a_gc0 = b.build_int_add(t, a_col0, "").unwrap();
    let a_i0 = b.build_int_add(b.build_int_mul(a_grow0, k_p, "").unwrap(), a_gc0, "").unwrap();
    let a_gp0 = unsafe { b.build_gep(f16_ty, a_ptr, &[a_i0], "").unwrap() };
    let sa_gp0 = unsafe { b.build_gep(f16_ty, smem_a, &[a_sw0], "").unwrap() };
    build_cp_async_16(&b, &ctx, &module, sa_gp0, a_gp0);

    let a_gc1 = b.build_int_add(t, a_col1, "").unwrap();
    let a_i1 = b.build_int_add(b.build_int_mul(a_grow1, k_p, "").unwrap(), a_gc1, "").unwrap();
    let a_gp1 = unsafe { b.build_gep(f16_ty, a_ptr, &[a_i1], "").unwrap() };
    let sa_gp1 = unsafe { b.build_gep(f16_ty, smem_a, &[a_sw1], "").unwrap() };
    build_cp_async_16(&b, &ctx, &module, sa_gp1, a_gp1);

    // ── cp.async load B tile [BK×BN] — 2 cp.async per thread ──
    let b_gr0 = b.build_int_add(t, b_row0, "").unwrap();
    let b_i0 = b.build_int_add(b.build_int_mul(b_gr0, n_p, "").unwrap(), b_gcol0, "").unwrap();
    let b_gp0 = unsafe { b.build_gep(f16_ty, b_ptr, &[b_i0], "").unwrap() };
    let sb_gp0 = unsafe { b.build_gep(f16_ty, smem_b, &[b_sw0], "").unwrap() };
    build_cp_async_16(&b, &ctx, &module, sb_gp0, b_gp0);

    let b_gr1 = b.build_int_add(t, b_row1, "").unwrap();
    let b_i1 = b.build_int_add(b.build_int_mul(b_gr1, n_p, "").unwrap(), b_gcol1, "").unwrap();
    let b_gp1 = unsafe { b.build_gep(f16_ty, b_ptr, &[b_i1], "").unwrap() };
    let sb_gp1 = unsafe { b.build_gep(f16_ty, smem_b, &[b_sw1], "").unwrap() };
    build_cp_async_16(&b, &ctx, &module, sb_gp1, b_gp1);

    build_cp_async_commit_and_wait(&b, &ctx, &module);
    call_barrier0(&ctx, &module, &b);

    // ── K-unroll: 2 rounds of fragment load + MMA from the same smem tile ──
    // First round uses K offset 0, second uses K offset MMA_K=16
    let mut cur_accs: Vec<FloatValue> = (0..num_acc)
        .map(|i| acc_phis[i].as_basic_value().into_float_value())
        .collect();

    for ki in 0..K_UNROLL as u64 {
        let k_off = ci(ki * 16); // MMA K dimension = 16

        // Load A fragments for this K-slice
        let mut a_frags: Vec<IntValue> = Vec::new();
        for rm in 0..REG_M as u64 {
            let rm_off = b.build_int_add(wy_off, ci(rm * MMA_M as u64), "").unwrap();
            for (row_add, col_add) in [(0u64, 0u64), (8, 0), (0, 8), (8, 8)] {
                let frow = b.build_int_add(
                    b.build_int_add(rm_off, group, "").unwrap(), ci(row_add), "",
                ).unwrap();
                let fcol = b.build_int_add(
                    b.build_int_add(tg2, ci(col_add), "").unwrap(), k_off, "",
                ).unwrap();
                let lin = b.build_int_add(
                    b.build_int_mul(frow, ci(BK as u64), "").unwrap(), fcol, "",
                ).unwrap();
                let sw = build_swizzle(&b, &ctx, lin);
                let gep = unsafe { b.build_gep(f16_ty, smem_a, &[sw], "").unwrap() };
                a_frags.push(b.build_load(i32_ty, gep, "").unwrap().into_int_value());
            }
        }

        // Load B fragments for this K-slice
        let mut b_frags: Vec<IntValue> = Vec::new();
        for rn in 0..REG_N as u64 {
            let b_col = b.build_int_add(
                b.build_int_add(wx_off, ci(rn * MMA_N as u64), "").unwrap(),
                group, "",
            ).unwrap();
            for fk_add in [0u64, 8] {
                let k0 = b.build_int_add(
                    b.build_int_add(tg2, ci(fk_add), "").unwrap(), k_off, "",
                ).unwrap();
                let k1 = b.build_int_add(k0, ci(1), "").unwrap();
                let lin0 = b.build_int_add(
                    b.build_int_mul(k0, ci(BN as u64), "").unwrap(), b_col, "",
                ).unwrap();
                let lin1 = b.build_int_add(
                    b.build_int_mul(k1, ci(BN as u64), "").unwrap(), b_col, "",
                ).unwrap();
                let sw0 = build_swizzle(&b, &ctx, lin0);
                let sw1 = build_swizzle(&b, &ctx, lin1);
                let gep0 = unsafe { b.build_gep(f16_ty, smem_b, &[sw0], "").unwrap() };
                let gep1 = unsafe { b.build_gep(f16_ty, smem_b, &[sw1], "").unwrap() };
                let v0 = b.build_load(f16_ty, gep0, "").unwrap();
                let v1 = b.build_load(f16_ty, gep1, "").unwrap();
                let vec = b.build_insert_element(v2f16_ty.get_undef(), v0, ci(0), "").unwrap();
                let vec = b.build_insert_element(vec, v1, ci(1), "").unwrap();
                b_frags.push(b.build_bit_cast(vec, i32_ty, "").unwrap().into_int_value());
            }
        }

        // REG_M × REG_N MMA operations
        let mut next_accs: Vec<FloatValue> = Vec::new();
        for rm in 0..REG_M {
            for rn in 0..REG_N {
                let acc_base = (rm * REG_N * 4 + rn * 4) as usize;
                let a_base = (rm * 4) as usize;
                let b_base = (rn * 2) as usize;
                let [d0, d1, d2, d3] = build_mma_asm(
                    &b, &ctx, &module,
                    &[a_frags[a_base], a_frags[a_base+1], a_frags[a_base+2], a_frags[a_base+3]],
                    &[b_frags[b_base], b_frags[b_base+1]],
                    &[cur_accs[acc_base], cur_accs[acc_base+1],
                      cur_accs[acc_base+2], cur_accs[acc_base+3]],
                );
                next_accs.push(d0); next_accs.push(d1); next_accs.push(d2); next_accs.push(d3);
            }
        }
        cur_accs = next_accs;
    }

    let new_accs = cur_accs;

    call_barrier0(&ctx, &module, &b);

    let t_next = b.build_int_add(t, ci(BK as u64), "tn").unwrap();
    b.build_unconditional_branch(kloop_hdr).unwrap();

    // Wire phi nodes
    t_phi.add_incoming(&[(&ci(0), entry), (&t_next, kloop_body)]);
    for i in 0..num_acc {
        acc_phis[i].add_incoming(&[(&f0, entry), (&new_accs[i], kloop_body)]);
    }

    // ── K-loop exit: store C ──
    b.position_at_end(kloop_exit);

    for rm in 0..REG_M {
        for rn in 0..REG_N {
            let acc_base = (rm * REG_N * 4 + rn * 4) as usize;
            for d in 0..4u32 {
                let mma_row_off = if d < 2 { group } else {
                    b.build_int_add(group, ci(8), "").unwrap()
                };
                let mma_col_off = if d % 2 == 0 { tg2 } else {
                    b.build_int_add(tg2, ci(1), "").unwrap()
                };

                let c_row = b.build_int_add(
                    b.build_int_add(block_row, wy_off, "").unwrap(),
                    b.build_int_add(ci(rm as u64 * MMA_M as u64), mma_row_off, "").unwrap(),
                    "",
                ).unwrap();
                let c_col = b.build_int_add(
                    b.build_int_add(block_col, wx_off, "").unwrap(),
                    b.build_int_add(ci(rn as u64 * MMA_N as u64), mma_col_off, "").unwrap(),
                    "",
                ).unwrap();
                let c_idx = b.build_int_add(
                    b.build_int_mul(c_row, n_p, "").unwrap(), c_col, "",
                ).unwrap();
                let gep = unsafe { b.build_gep(f32_ty, c_ptr, &[c_idx], "").unwrap() };
                let val = acc_phis[acc_base + d as usize].as_basic_value().into_float_value();
                b.build_store(gep, val).unwrap();
            }
        }
    }

    b.build_return(None).unwrap();

    let buf = machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow::anyhow!("PTX emission: {}", e))?;
    let ptx = std::str::from_utf8(buf.as_slice()).context("PTX not UTF-8")?.to_string();
    Ok(ptx)
}

// ===========================================================================
// Inline asm for mma.sync
// ===========================================================================

/// cp.async.cg.shared.global [dst_shared], [src_global], 16;
/// Asynchronously copies 16 bytes (128 bits) from global to shared memory.
fn build_cp_async_16<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    dst_shared: inkwell::values::PointerValue<'ctx>,
    src_global: inkwell::values::PointerValue<'ctx>,
) {
    let i32_ty = context.i32_type();
    let i64_ty = context.i64_type();
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();

        let void_ty = llvm_sys::core::LLVMVoidTypeInContext(ctx_ref);
        let mut param_types = [i32_ty.as_type_ref(), i64_ty.as_type_ref()];
        let fn_type = llvm_sys::core::LLVMFunctionType(
            void_ty, param_types.as_mut_ptr(), 2, 0,
        );

        // Convert pointers to integer addresses for the asm
        let dst_i32 = builder.build_ptr_to_int(dst_shared, i32_ty, "cp_dst").unwrap();
        let src_i64 = builder.build_ptr_to_int(src_global, i64_ty, "cp_src").unwrap();

        let asm_str = b"cp.async.cg.shared.global [$0], [$1], 16;\0";
        let constraints = b"r,l\0";

        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type,
            asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            1, 0,
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0,
        );

        let mut args = [dst_i32.as_value_ref(), src_i64.as_value_ref()];
        llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val,
            args.as_mut_ptr(), 2, b"\0".as_ptr() as *const _,
        );
    }
}

/// cp.async.commit_group + cp.async.wait_group 0
fn build_cp_async_commit_and_wait<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
) {
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();

        let void_ty = llvm_sys::core::LLVMVoidTypeInContext(ctx_ref);
        let fn_type = llvm_sys::core::LLVMFunctionType(void_ty, std::ptr::null_mut(), 0, 0);

        // commit_group
        let asm_commit = b"cp.async.commit_group;\0";
        let constraints_empty = b"\0";
        let commit = llvm_sys::core::LLVMGetInlineAsm(
            fn_type,
            asm_commit.as_ptr() as *const _, asm_commit.len() - 1,
            constraints_empty.as_ptr() as *const _, constraints_empty.len() - 1,
            1, 0,
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0,
        );
        llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, commit,
            std::ptr::null_mut(), 0, b"\0".as_ptr() as *const _,
        );

        // wait_group 0
        let asm_wait = b"cp.async.wait_group 0;\0";
        let wait = llvm_sys::core::LLVMGetInlineAsm(
            fn_type,
            asm_wait.as_ptr() as *const _, asm_wait.len() - 1,
            constraints_empty.as_ptr() as *const _, constraints_empty.len() - 1,
            1, 0,
            llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
            0,
        );
        llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, wait,
            std::ptr::null_mut(), 0, b"\0".as_ptr() as *const _,
        );
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
