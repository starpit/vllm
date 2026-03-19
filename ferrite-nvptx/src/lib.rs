#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

// Minimal MMA GEMM kernel compiled via rustc nightly → nvptx64.
// Tests whether LLVM 22's NVPTX backend produces better codegen than LLVM 20.

use core::arch::asm;

const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 16;
const WM: u32 = 64;
const WN: u32 = 16;
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const REG_M: u32 = WM / MMA_M; // 4
const REG_N: u32 = WN / MMA_N; // 2

#[inline(always)]
fn swizzle(idx: u32) -> u32 {
    let b = idx * 2;
    let s = b ^ ((b & 0x380) >> 3);
    s / 2
}

#[inline(always)]
unsafe fn tid_x() -> u32 {
    let r: u32;
    asm!("mov.u32 {}, %tid.x;", out(reg32) r);
    r
}

#[inline(always)]
unsafe fn ctaid_x() -> u32 {
    let r: u32;
    asm!("mov.u32 {}, %ctaid.x;", out(reg32) r);
    r
}

#[inline(always)]
unsafe fn ctaid_y() -> u32 {
    let r: u32;
    asm!("mov.u32 {}, %ctaid.y;", out(reg32) r);
    r
}

#[inline(always)]
unsafe fn syncthreads() {
    asm!("bar.sync 0;");
}

#[inline(always)]
unsafe fn cp_async_16(dst: *mut u8, src: *const u8) {
    asm!(
        "cp.async.cg.shared.global [{dst}], [{src}], 16;",
        dst = in(reg32) dst as u32,
        src = in(reg64) src as u64,
    );
}

#[inline(always)]
unsafe fn cp_async_commit_wait() {
    asm!("cp.async.commit_group;");
    asm!("cp.async.wait_group 0;");
}

#[inline(always)]
unsafe fn mma_sync(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    let mut d = [0.0f32; 4];
    asm!(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
            {{{d0},{d1},{d2},{d3}}}, \
            {{{a0},{a1},{a2},{a3}}}, \
            {{{b0},{b1}}}, \
            {{{c0},{c1},{c2},{c3}}};",
        d0 = out(reg32) d[0],
        d1 = out(reg32) d[1],
        d2 = out(reg32) d[2],
        d3 = out(reg32) d[3],
        a0 = in(reg32) a[0],
        a1 = in(reg32) a[1],
        a2 = in(reg32) a[2],
        a3 = in(reg32) a[3],
        b0 = in(reg32) b[0],
        b1 = in(reg32) b[1],
        c0 = in(reg32) c[0],
        c1 = in(reg32) c[1],
        c2 = in(reg32) c[2],
        c3 = in(reg32) c[3],
    );
    d
}

// Shared memory — declared as a large static array
// On nvptx64, address space 3 = shared
#[unsafe(link_section = ".shared")]
static mut SMEM: [u8; (BM * BK + BK * BN) as usize * 2] =
    [0u8; (BM * BK + BK * BN) as usize * 2];

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn mma_gemm(
    a_ptr: *const u16,
    b_ptr: *const u16,
    c_ptr: *mut f32,
    _m: u32,
    n: u32,
    k: u32,
) {
    let tid = tid_x();
    let bid_x = ctaid_x();
    let bid_y = ctaid_y();

    let block_row = bid_y * BM;
    let block_col = bid_x * BN;

    let warp_id = tid / 32;
    let lane = tid % 32;
    // Warp layout: 1 warp along M (w64), 4 warps along N (w16 each? No — 64/16=4 warps along N)
    // Actually: BM/WM = 64/64 = 1, BN/WN = 64/16 = 4. So 1×4 = 4 warps.
    let wy = warp_id / 4; // 0 for all warps (only 1 row)
    let wx = warp_id % 4; // 0..3
    let group = lane / 4;
    let tg = lane % 4;
    let tg2 = tg * 2;

    let wy_off = wy * WM;
    let wx_off = wx * WN;

    let smem_a = core::ptr::addr_of_mut!(SMEM) as *mut u16;
    let smem_b = smem_a.add((BM * BK) as usize);

    // Accumulators: REG_M=4 × REG_N=2 × 4 f32 = 32
    let mut acc = [[0.0f32; 4]; (REG_M * REG_N) as usize];

    let tid_x8 = tid * 8;
    let a_tile_row = tid_x8 / BK;
    let a_tile_col = tid_x8 % BK;
    let a_glob_row = block_row + a_tile_row;

    let b_tile_row = tid_x8 / BN;
    let b_tile_col = tid_x8 % BN;
    let b_glob_col = block_col + b_tile_col;

    let a_smem_sw = swizzle(tid_x8);
    let b_smem_sw = swizzle(tid_x8);

    let mut t: u32 = 0;
    while t < k {
        // Load A via cp.async
        let a_glob_col = t + a_tile_col;
        let a_idx = a_glob_row * k + a_glob_col;
        let a_src = a_ptr.add(a_idx as usize) as *const u8;
        let a_dst = (smem_a as *mut u8).add((a_smem_sw * 2) as usize);
        cp_async_16(a_dst, a_src);

        // Load B via cp.async
        let b_glob_row = t + b_tile_row;
        let b_idx = b_glob_row * n + b_glob_col;
        let b_src = b_ptr.add(b_idx as usize) as *const u8;
        let b_dst = (smem_b as *mut u8).add((b_smem_sw * 2) as usize);
        cp_async_16(b_dst, b_src);

        cp_async_commit_wait();
        syncthreads();

        // Load A fragments
        let mut a_frags = [[0u32; 4]; REG_M as usize];
        for rm in 0..REG_M {
            let rm_off = wy_off + rm * MMA_M;
            let offsets: [(u32, u32); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];
            for (fi, (ra, ca)) in offsets.iter().enumerate() {
                let frow = rm_off + group + ra;
                let fcol = tg2 + ca;
                let lin = frow * BK + fcol;
                let sw = swizzle(lin);
                a_frags[rm as usize][fi] = *(smem_a.add(sw as usize) as *const u32);
            }
        }

        // Per-N: load B, MMA all M
        for rn in 0..REG_N {
            let b_col = wx_off + rn * MMA_N + group;
            let mut b_frag = [0u32; 2];
            for (fi, fk_add) in [0u32, 8].iter().enumerate() {
                let k0 = tg2 + fk_add;
                let k1 = k0 + 1;
                let lin0 = k0 * BN + b_col;
                let lin1 = k1 * BN + b_col;
                let sw0 = swizzle(lin0);
                let sw1 = swizzle(lin1);
                let v0 = *smem_b.add(sw0 as usize);
                let v1 = *smem_b.add(sw1 as usize);
                b_frag[fi] = (v0 as u32) | ((v1 as u32) << 16);
            }

            for rm in 0..REG_M {
                let acc_idx = (rm * REG_N + rn) as usize;
                acc[acc_idx] = mma_sync(a_frags[rm as usize], b_frag, acc[acc_idx]);
            }
        }

        syncthreads();
        t += BK;
    }

    // Store C
    for rm in 0..REG_M {
        for rn in 0..REG_N {
            let acc_idx = (rm * REG_N + rn) as usize;
            for d in 0..4u32 {
                let mma_row_off = if d < 2 { group } else { group + 8 };
                let mma_col_off = if d % 2 == 0 { tg2 } else { tg2 + 1 };
                let c_row = block_row + wy_off + rm * MMA_M + mma_row_off;
                let c_col = block_col + wx_off + rn * MMA_N + mma_col_off;
                let c_idx = (c_row * n + c_col) as usize;
                *c_ptr.add(c_idx) = acc[acc_idx][d as usize];
            }
        }
    }
}
