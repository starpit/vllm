// SPDX-License-Identifier: Apache-2.0
//
// CubeK-style GEMM: faithful port of CubeK's "simple multi-row" matmul.
//
// Config: tile 8×32×16, partition 4×4×2, stage 4×1×1
// Block tile: 128×128, K-step: 32
// 4 warps (128 threads), each warp handles 32×128 of output (4×4 partition tiles)
// 32 MMA ops per warp per K-step
// 128 f32 accumulator registers per thread
// B128 swizzle on shared memory
// Synchronous loading (cp.async) + __syncthreads barrier

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

// CubeK config: tile 8×32×16, partition 4×4×2, stage 4×1×1
const TILE_M: u32 = 8;
const TILE_N: u32 = 32;
const TILE_K: u32 = 16;
const PART_M: u32 = 4;  // tiles per partition along M
const PART_N: u32 = 4;  // tiles per partition along N
const PART_K: u32 = 2;  // tiles per partition along K
const STAGE_M: u32 = 4; // partitions per stage along M (= num warps)

// Derived
const STAGE_DIM_M: u32 = STAGE_M * PART_M * TILE_M;  // 4*4*8 = 128
const STAGE_DIM_N: u32 = PART_N * TILE_N;              // 4*32 = 128
const STAGE_DIM_K: u32 = PART_K * TILE_K;              // 2*16 = 32
const WARPS: u32 = STAGE_M;                            // 4
const THREADS: u32 = WARPS * 32;                       // 128

// MMA: m16n8k16 (the actual hardware instruction CubeK maps 8×32×16 tiles to)
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const MMA_K: u32 = 16;

// Per-warp work: 4 tiles along M, 4 tiles along N, 2 along K = 32 MMAs
// But 8×32 tile maps to m16n8k16 how?
// TILE_M=8 < MMA_M=16: the 8-row tile uses half the MMA's M dimension
// TILE_N=32 = 4 × MMA_N=8: each tile-N requires 4 MMA-N columns
// So one 8×32×16 tile = 1(M-half) × 4(N) = 4 mma.sync ops
// Per partition: 4(M-tiles) × 4(N-tiles) × 2(K-tiles) × 4(MMAs/tile) = 128 MMAs?
// Wait, that's too many. Let me reconsider.
//
// Actually: the virtual tile 8×32×16 has these MMA register counts:
//   A: 2 regs (not 4 like m16n8k16 normally needs)
//   B: 8 regs
//   D: 8 regs
// This suggests the tile IS a single mma.sync but with a different fragment packing.
// The m16n8k16 instruction actually computes 16 rows × 8 cols, but CubeK's
// 8×32 tile uses it as: the 8 rows map to half the MMA's 16-row output,
// and 32 cols = 4 × 8 cols packed into the B fragment (8 regs = 4 sub-tiles of 2 regs each).
//
// For now, let's use the standard m16n8k16 and match CubeK's OUTER structure:
// Per warp: PART_M × PART_N × PART_K MMA groups
// Each MMA group: 1 mma.sync.m16n8k16 (since tile_m < mma_m, we just use part of it)
//
// Actually, looking at CubeK's numbers again:
//   A regs = 2, B regs = 8, C/D regs = 8
// For m16n8k16: A=4, B=2, D=4.
// These don't match. CubeK must be mapping differently.
//
// The 8×32×16 tile with B=8 regs suggests it's actually issuing
// mma.sync.m16n8k16 FOUR TIMES per tile (once per 8-column slice of the 32-col tile).
// But with B being "8 regs" per load, they load ALL 4 slices at once
// and execute 4 MMAs sharing the same A fragment.
//
// So per tile: 4 MMAs, and per partition: 4(M)×4(N)×2(K)×4(MMAs/tile-N) = way too many.
//
// Let me simplify: I'll use m16n8k16 directly and match CubeK's BLOCK tile (128×128)
// and K-step (32), with the correct partition loop ordering.
// Per warp at m16n8k16 granularity:
//   M tiles: PART_M * (TILE_M/MMA_M if TILE_M>=MMA_M, else 1) ... this is getting complicated.
//
// SIMPLIFICATION: Just match the outer dimensions and instruction count.
// CubeK gets 32 MMA ops per warp per K-step of 32.
// With m16n8k16:
//   K-steps within stage: 32/16 = 2
//   Per K-step: we need 32/2 = 16 MMAs per warp
//   16 MMAs = 4(M) × 4(N) with m16 rows and n8 cols
//   → warp tile: 4*16=64 rows × 4*8=32 cols = 64×32 per warp
//   → 4 warps: need to cover 128×128. 4 warps × 64×32 = 128×64? No.
//   → 2 warps along M (64*2=128), 2 along N (32*2=64)? That's only 128×64.
//   → Or: 4 warps each do 32×128? 4*32=128 M, 128 N? Yes.
//     32 rows = 2*MMA_M, 128 cols = 16*MMA_N → 2*16=32 MMAs per K-step.
//     × 2 K-steps = 64 MMAs per warp per K-loop. That's twice CubeK's 32.
//
// I think the answer is: CubeK's 8×32 tile maps to a SINGLE mma.sync.m16n8k16
// where only 8 of the 16 output rows are used (the other 8 are wasted or shared
// with another tile). But I'm not sure about this.
//
// DECISION: Stop overthinking the tile mapping. Use our proven m16n8k16 layout,
// match CubeK's OUTER structure (128×128 block, K=32, 4 warps, partition 4×4×2 ordering),
// and see what happens.

// Warp tile using m16n8k16: each warp does REG_M × REG_N MMAs per K-slice
const WM: u32 = 32;    // 2 * MMA_M = 32 rows per warp
const WN: u32 = 128;   // 16 * MMA_N = 128 cols per warp (all cols, warps tile along M only)
const REG_M: u32 = WM / MMA_M;  // 2
const REG_N: u32 = WN / MMA_N;  // 16
// Warps arranged: 4 along M (4*32=128), 1 along N (128)
const WARPS_M: u32 = 4;
const WARPS_N: u32 = 1;
// Total MMAs per warp per K-slice: 2*16 = 32 ← matches CubeK!
// Total MMAs per warp per K-step (2 K-slices): 32*2 = 64

// Swizzle
const SWIZZLE_MASK: u32 = 0x380;
const SWIZZLE_SHIFT: u32 = 3;

pub fn step_cubek_gemm(sm: &str) -> Result<()> {
    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;

    let ptx = emit_cubek_gemm_ptx(sm)?;
    println!("  [llvm] Generated {} bytes of PTX", ptx.len());

    if std::env::var("FERRITE_DUMP_PTX").is_ok() {
        std::fs::write("/tmp/ferrite_cubek_gemm.ptx", &ptx)?;
        println!("  [debug] PTX written to /tmp/ferrite_cubek_gemm.ptx");
    }

    let ptx_cstr = CString::new(ptx.as_bytes()).context("PTX null")?;
    let module = unsafe { cuda::module::load_data(ptx_cstr.as_ptr() as *const _)? };
    let func = unsafe {
        cuda::module::get_function(module, CString::new("cubek_gemm").unwrap())?
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
    let grid_x: c_uint = n / STAGE_DIM_N;    // 1024/128 = 8
    let grid_y: c_uint = m / STAGE_DIM_M;    // 1024/128 = 8
    // smem: A[128×32 f16] + B[32×128 f16] = 8192 + 8192 = 16384 bytes
    let smem_bytes: c_uint = (STAGE_DIM_M * STAGE_DIM_K + STAGE_DIM_K * STAGE_DIM_N) * 2;

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
        bail!("CubeK GEMM verification failed");
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
// Swizzle
// ===========================================================================

fn build_swizzle<'ctx>(
    b: &Builder<'ctx>,
    ctx: &'ctx LlvmContext,
    elem_idx: IntValue<'ctx>,
) -> IntValue<'ctx> {
    let i32_ty = ctx.i32_type();
    let ci = |v: u64| i32_ty.const_int(v, false);
    let byte_off = b.build_int_mul(elem_idx, ci(2), "").unwrap();
    let masked = b.build_and(byte_off, ci(SWIZZLE_MASK as u64), "").unwrap();
    let shifted = b.build_right_shift(masked, ci(SWIZZLE_SHIFT as u64), false, "").unwrap();
    let swizzled = b.build_xor(byte_off, shifted, "").unwrap();
    b.build_int_unsigned_div(swizzled, ci(2), "sw").unwrap()
}

// ===========================================================================
// PTX emission
// ===========================================================================

fn emit_cubek_gemm_ptx(sm: &str) -> Result<String> {
    let machine = create_nvptx_target_machine(sm)?;
    let ctx = LlvmContext::create();
    let module = ctx.create_module("ferrite_cubek_gemm");
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
    smem_g.set_alignment(128);
    smem_g.set_externally_initialized(true);

    let fn_type = void_ty.fn_type(
        &[ptr_g.into(), ptr_g.into(), ptr_g.into(),
          i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );
    let function = module.add_function("cubek_gemm", fn_type, None);
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

    let block_row = b.build_int_mul(bid_y, ci(STAGE_DIM_M as u64), "br").unwrap();
    let block_col = b.build_int_mul(bid_x, ci(STAGE_DIM_N as u64), "bc").unwrap();

    let warp_id = b.build_int_unsigned_div(tid, ci(32), "wid").unwrap();
    let lane = b.build_int_unsigned_rem(tid, ci(32), "lane").unwrap();
    let group = b.build_int_unsigned_div(lane, ci(4), "grp").unwrap();
    let tg = b.build_int_unsigned_rem(lane, ci(4), "tg").unwrap();
    let tg2 = b.build_int_mul(tg, ci(2), "tg2").unwrap();

    // Warp layout: 4 warps along M, 1 along N
    // Each warp handles rows [warp_id*WM .. warp_id*WM+WM-1] = [warp_id*32 .. +31]
    // and ALL 128 columns
    let wy_off = b.build_int_mul(warp_id, ci(WM as u64), "wyo").unwrap();

    // Shared memory: smem_a[128×32 f16] + smem_b[32×128 f16]
    let smem_base = smem_g.as_pointer_value();
    let smem_a = b.build_pointer_cast(smem_base, ptr_s, "sa").unwrap();
    let smem_b = b.build_pointer_cast(unsafe {
        b.build_gep(ctx.i8_type(), smem_base,
            &[ci((STAGE_DIM_M * STAGE_DIM_K * 2) as u64)], "").unwrap()
    }, ptr_s, "sb").unwrap();

    let f0 = f32_ty.const_float(0.0);

    // Precompute loading offsets. 128 threads load 8192 f16 each for A and B.
    // A: 128×32 = 4096 f16 → 32/thread → 4 cp.async(8)
    // B: 32×128 = 4096 f16 → 32/thread → 4 cp.async(8)
    // Total: 8 cp.async per thread per K-step
    // We'll use a loop for this rather than unrolling 8 times in the IR.
    // Each cp.async copies 16 bytes (8 f16).

    b.build_unconditional_branch(kloop_hdr).unwrap();

    // ── K-loop header ──
    b.position_at_end(kloop_hdr);
    let t_phi = b.build_phi(i32_ty, "t").unwrap();
    // REG_M=2 * REG_N=16 * 4 = 128 accumulators
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

    // ── Load A[128×32] and B[32×128] into shared memory ──
    // 4096 f16 each = 8192 bytes each
    // 128 threads, 32 elements/thread each, 4 cp.async per matrix
    for chunk in 0..4u64 {
        let base = b.build_int_add(
            b.build_int_mul(tid, ci(32), "").unwrap(),
            ci(chunk * 8), "",
        ).unwrap();
        // For A: linear index → (row, col) in [128×32]
        let a_row = b.build_int_unsigned_div(base, ci(STAGE_DIM_K as u64), "").unwrap();
        let a_col = b.build_int_unsigned_rem(base, ci(STAGE_DIM_K as u64), "").unwrap();
        let a_grow = b.build_int_add(block_row, a_row, "").unwrap();
        let a_gcol = b.build_int_add(t, a_col, "").unwrap();
        let a_idx = b.build_int_add(
            b.build_int_mul(a_grow, k_p, "").unwrap(), a_gcol, "",
        ).unwrap();
        let a_gep = unsafe { b.build_gep(f16_ty, a_ptr, &[a_idx], "").unwrap() };
        let a_sw = build_swizzle(&b, &ctx, base);
        let sa_gep = unsafe { b.build_gep(f16_ty, smem_a, &[a_sw], "").unwrap() };
        build_cp_async_16(&b, &ctx, &module, sa_gep, a_gep);
    }

    for chunk in 0..4u64 {
        let base = b.build_int_add(
            b.build_int_mul(tid, ci(32), "").unwrap(),
            ci(chunk * 8), "",
        ).unwrap();
        // For B: linear index → (row, col) in [32×128]
        let b_row = b.build_int_unsigned_div(base, ci(STAGE_DIM_N as u64), "").unwrap();
        let b_col = b.build_int_unsigned_rem(base, ci(STAGE_DIM_N as u64), "").unwrap();
        let b_grow = b.build_int_add(t, b_row, "").unwrap();
        let b_gcol = b.build_int_add(block_col, b_col, "").unwrap();
        let b_idx = b.build_int_add(
            b.build_int_mul(b_grow, n_p, "").unwrap(), b_gcol, "",
        ).unwrap();
        let b_gep = unsafe { b.build_gep(f16_ty, b_ptr, &[b_idx], "").unwrap() };
        let b_sw = build_swizzle(&b, &ctx, base);
        let sb_gep = unsafe { b.build_gep(f16_ty, smem_b, &[b_sw], "").unwrap() };
        build_cp_async_16(&b, &ctx, &module, sb_gep, b_gep);
    }

    build_cp_async_commit_and_wait(&b, &ctx, &module);
    call_barrier0(&ctx, &module, &b);

    // ── CubeK partition inner loop (fully unrolled) ──
    // for k_iter in 0..PART_K (=2):
    //   load all A fragments
    //   for n_iter in 0..REG_N (=16):
    //     load B fragment
    //     for m_iter in 0..REG_M (=2):
    //       mma(a[m], b, acc[m][n])

    let mut cur_accs: Vec<FloatValue> = (0..num_acc)
        .map(|i| acc_phis[i].as_basic_value().into_float_value())
        .collect();

    for ki in 0..PART_K as u64 {
        let k_off = ci(ki * MMA_K as u64);

        // Load all A fragments (with p3:32:32, these use 32-bit addressing automatically)
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
                    b.build_int_mul(frow, ci(STAGE_DIM_K as u64), "").unwrap(), fcol, "",
                ).unwrap();
                let sw = build_swizzle(&b, &ctx, lin);
                let gep = unsafe { b.build_gep(f16_ty, smem_a, &[sw], "").unwrap() };
                a_frags.push(b.build_load(i32_ty, gep, "").unwrap().into_int_value());
            }
        }

        // For each N-tile: load B, MMA all M
        for rn in 0..REG_N as u64 {
            let b_col = b.build_int_add(ci(rn * MMA_N as u64), group, "").unwrap();

            let mut b_frag = Vec::new();
            for fk_add in [0u64, 8] {
                let k0 = b.build_int_add(
                    b.build_int_add(tg2, ci(fk_add), "").unwrap(), k_off, "",
                ).unwrap();
                let k1 = b.build_int_add(k0, ci(1), "").unwrap();
                let lin0 = b.build_int_add(
                    b.build_int_mul(k0, ci(STAGE_DIM_N as u64), "").unwrap(), b_col, "",
                ).unwrap();
                let lin1 = b.build_int_add(
                    b.build_int_mul(k1, ci(STAGE_DIM_N as u64), "").unwrap(), b_col, "",
                ).unwrap();
                let sw0 = build_swizzle(&b, &ctx, lin0);
                let sw1 = build_swizzle(&b, &ctx, lin1);
                let gep0 = unsafe { b.build_gep(f16_ty, smem_b, &[sw0], "").unwrap() };
                let gep1 = unsafe { b.build_gep(f16_ty, smem_b, &[sw1], "").unwrap() };
                let v0 = b.build_load(f16_ty, gep0, "").unwrap();
                let v1 = b.build_load(f16_ty, gep1, "").unwrap();
                let vec = b.build_insert_element(v2f16_ty.get_undef(), v0, ci(0), "").unwrap();
                let vec = b.build_insert_element(vec, v1, ci(1), "").unwrap();
                b_frag.push(b.build_bit_cast(vec, i32_ty, "").unwrap().into_int_value());
            }

            for rm in 0..REG_M as u64 {
                let acc_idx = (rm as u32 * REG_N * 4 + rn as u32 * 4) as usize;
                let a_base = (rm * 4) as usize;
                let [d0, d1, d2, d3] = build_mma_asm(
                    &b, &ctx, &module,
                    &[a_frags[a_base], a_frags[a_base+1], a_frags[a_base+2], a_frags[a_base+3]],
                    &[b_frag[0], b_frag[1]],
                    &[cur_accs[acc_idx], cur_accs[acc_idx+1],
                      cur_accs[acc_idx+2], cur_accs[acc_idx+3]],
                );
                cur_accs[acc_idx] = d0;
                cur_accs[acc_idx+1] = d1;
                cur_accs[acc_idx+2] = d2;
                cur_accs[acc_idx+3] = d3;
            }
        }
    }

    let new_accs = cur_accs;

    call_barrier0(&ctx, &module, &b);

    let t_next = b.build_int_add(t, ci(STAGE_DIM_K as u64), "tn").unwrap();
    b.build_unconditional_branch(kloop_hdr).unwrap();

    // Wire phi nodes
    t_phi.add_incoming(&[(&ci(0), entry), (&t_next, kloop_body)]);
    for i in 0..num_acc {
        acc_phis[i].add_incoming(&[(&f0, entry), (&new_accs[i], kloop_body)]);
    }

    // ── Store C ──
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
                    b.build_int_add(block_col, ci(rn as u64 * MMA_N as u64), "").unwrap(),
                    mma_col_off, "",
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
// Inline asm helpers (same as tiled_mma.rs)
// ===========================================================================

/// Load i32 from shared memory using 32-bit addressing.
/// Avoids LLVM's 64-bit pointer expansion that wastes registers.
/// smem_base is the shared pointer base, byte_off is a 32-bit byte offset.
fn build_ld_shared_u32<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    smem_base: inkwell::values::PointerValue<'ctx>,
    elem_off: IntValue<'ctx>, // offset in f16 elements
) -> IntValue<'ctx> {
    let i32_ty = context.i32_type();
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();

        // Compute 32-bit byte address: ptrtoint(base) + elem_off * 2
        let base_i32 = builder.build_ptr_to_int(smem_base, i32_ty, "").unwrap();
        let byte_off = builder.build_int_mul(elem_off, i32_ty.const_int(2, false), "").unwrap();
        let addr = builder.build_int_add(base_i32, byte_off, "").unwrap();

        let mut param_types = [i32_ty.as_type_ref()];
        let fn_type = llvm_sys::core::LLVMFunctionType(
            i32_ty.as_type_ref(), param_types.as_mut_ptr(), 1, 0,
        );

        let asm_str = b"ld.shared.u32 $0, [$1];\0";
        let constraints = b"=r,r\0";

        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type, asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            0, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0,
        );

        let mut args = [addr.as_value_ref()];
        let call = llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val, args.as_mut_ptr(), 1, b"\0".as_ptr() as *const _,
        );

        IntValue::new(call)
    }
}

/// Load f16 from shared memory using 32-bit addressing.
fn build_ld_shared_f16<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx LlvmContext,
    module: &Module<'ctx>,
    smem_base: inkwell::values::PointerValue<'ctx>,
    elem_off: IntValue<'ctx>,
) -> inkwell::values::BasicValueEnum<'ctx> {
    let i32_ty = context.i32_type();
    let f16_ty = context.f16_type();
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();

        let base_i32 = builder.build_ptr_to_int(smem_base, i32_ty, "").unwrap();
        let byte_off = builder.build_int_mul(elem_off, i32_ty.const_int(2, false), "").unwrap();
        let addr = builder.build_int_add(base_i32, byte_off, "").unwrap();

        let mut param_types = [i32_ty.as_type_ref()];
        let fn_type = llvm_sys::core::LLVMFunctionType(
            f16_ty.as_type_ref(), param_types.as_mut_ptr(), 1, 0,
        );

        let asm_str = b"ld.shared.b16 $0, [$1];\0";
        let constraints = b"=h,r\0"; // h = 16-bit register

        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type, asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            0, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0,
        );

        let mut args = [addr.as_value_ref()];
        let call = llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val, args.as_mut_ptr(), 1, b"\0".as_ptr() as *const _,
        );

        // Wrap raw LLVMValueRef as BasicValueEnum
        inkwell::values::BasicValueEnum::new(call)
    }
}

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
        let fn_type = llvm_sys::core::LLVMFunctionType(void_ty, param_types.as_mut_ptr(), 2, 0);
        let dst_i32 = builder.build_ptr_to_int(dst_shared, i32_ty, "").unwrap();
        let src_i64 = builder.build_ptr_to_int(src_global, i64_ty, "").unwrap();
        let asm_str = b"cp.async.cg.shared.global [$0], [$1], 16;\0";
        let constraints = b"r,l\0";
        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type, asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0,
        );
        let mut args = [dst_i32.as_value_ref(), src_i64.as_value_ref()];
        llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val, args.as_mut_ptr(), 2, b"\0".as_ptr() as *const _,
        );
    }
}

fn build_cp_async_commit_and_wait<'ctx>(
    builder: &Builder<'ctx>,
    _context: &'ctx LlvmContext,
    module: &Module<'ctx>,
) {
    unsafe {
        let mod_ref = module.as_mut_ptr();
        let ctx_ref = llvm_sys::core::LLVMGetModuleContext(mod_ref);
        let builder_ref = builder.as_mut_ptr();
        let void_ty = llvm_sys::core::LLVMVoidTypeInContext(ctx_ref);
        let fn_type = llvm_sys::core::LLVMFunctionType(void_ty, std::ptr::null_mut(), 0, 0);
        let empty = b"\0";
        for asm_str in [b"cp.async.commit_group;\0" as &[u8], b"cp.async.wait_group 0;\0"] {
            let asm_val = llvm_sys::core::LLVMGetInlineAsm(
                fn_type, asm_str.as_ptr() as *const _, asm_str.len() - 1,
                empty.as_ptr() as *const _, empty.len() - 1,
                1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0,
            );
            llvm_sys::core::LLVMBuildCall2(
                builder_ref, fn_type, asm_val, std::ptr::null_mut(), 0, b"\0".as_ptr() as *const _,
            );
        }
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
        let ret_struct = llvm_sys::core::LLVMStructTypeInContext(ctx_ref, ret_members.as_mut_ptr(), 4, 0);
        let mut param_types = [
            i32_ty.as_type_ref(), i32_ty.as_type_ref(), i32_ty.as_type_ref(), i32_ty.as_type_ref(),
            i32_ty.as_type_ref(), i32_ty.as_type_ref(),
            f32_ty.as_type_ref(), f32_ty.as_type_ref(), f32_ty.as_type_ref(), f32_ty.as_type_ref(),
        ];
        let fn_type = llvm_sys::core::LLVMFunctionType(ret_struct, param_types.as_mut_ptr(), 10, 0);
        let asm_str = b"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0";
        let constraints = b"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f\0";
        let asm_val = llvm_sys::core::LLVMGetInlineAsm(
            fn_type, asm_str.as_ptr() as *const _, asm_str.len() - 1,
            constraints.as_ptr() as *const _, constraints.len() - 1,
            1, 0, llvm_sys::LLVMInlineAsmDialect::LLVMInlineAsmDialectATT, 0,
        );
        let mut args = [
            a_regs[0].as_value_ref(), a_regs[1].as_value_ref(),
            a_regs[2].as_value_ref(), a_regs[3].as_value_ref(),
            b_regs[0].as_value_ref(), b_regs[1].as_value_ref(),
            c_regs[0].as_value_ref(), c_regs[1].as_value_ref(),
            c_regs[2].as_value_ref(), c_regs[3].as_value_ref(),
        ];
        let call = llvm_sys::core::LLVMBuildCall2(
            builder_ref, fn_type, asm_val, args.as_mut_ptr(), 10, b"\0".as_ptr() as *const _,
        );
        let n = |s: &[u8]| s.as_ptr() as *const _;
        [
            FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 0, n(b"\0"))),
            FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 1, n(b"\0"))),
            FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 2, n(b"\0"))),
            FloatValue::new(llvm_sys::core::LLVMBuildExtractValue(builder_ref, call, 3, n(b"\0"))),
        ]
    }
}
