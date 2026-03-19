#![no_std]
#![feature(asm_experimental_arch)]

//! Ferrite Phase 1: MMA GEMM kernel compiled through rust-cuda/libnvvm.
//!
//! This proves that libnvvm gives us nvcc-quality register allocation
//! for the same kernel that hit 255 registers through upstream LLVM.

use core::mem::MaybeUninit;
use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::address_space;

#[cfg(target_os = "cuda")]
use core::arch::asm;

// Tile configuration matching our inkwell POC: 64×64, 4 warps, 2×4 register tiling
const BM: usize = 64;
const BN: usize = 64;
const BK: usize = 16;
const WM: usize = 32;
const WN: usize = 32;
const MMA_M: usize = 16;
const MMA_N: usize = 8;
const REG_M: usize = WM / MMA_M; // 2
const REG_N: usize = WN / MMA_N; // 4
const WARPS: usize = 4;

// Shared memory tiles
const SMEM_A_SIZE: usize = BM * BK; // 1024 f16 elements
const SMEM_B_SIZE: usize = BK * BN; // 1024 f16 elements

/// B128 swizzle: offset_bytes ^= (offset_bytes & 0x380) >> 3
#[inline(always)]
fn swizzle(elem_idx: u32) -> u32 {
    let byte_off = elem_idx * 2; // f16 = 2 bytes
    let swizzled = byte_off ^ ((byte_off & 0x380) >> 3);
    swizzled / 2
}

/// Load i32 (packed 2×f16) from swizzled shared memory address.
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn ld_shared_u32(smem_base: *const u16, elem_offset: u32) -> u32 {
    let sw = swizzle(elem_offset);
    let ptr = smem_base.add(sw as usize) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Load single f16 from swizzled shared memory address.
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn ld_shared_f16(smem_base: *const u16, elem_offset: u32) -> u16 {
    let sw = swizzle(elem_offset);
    core::ptr::read_volatile(smem_base.add(sw as usize))
}

/// Execute mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn mma_sync(
    a: [u32; 4], b: [u32; 2], c: [f32; 4],
) -> [f32; 4] {
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

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn mma_gemm(
    mat_a: &[u16],  // f16 as u16
    mat_b: &[u16],  // f16 as u16
    mat_c: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    #[address_space(shared)]
    static mut SMEM_A: [MaybeUninit<u16>; SMEM_A_SIZE] =
        [MaybeUninit::uninit(); SMEM_A_SIZE];
    #[address_space(shared)]
    static mut SMEM_B: [MaybeUninit<u16>; SMEM_B_SIZE] =
        [MaybeUninit::uninit(); SMEM_B_SIZE];

    let tid = thread::thread_idx_x();
    let bid_x = thread::block_idx_x();
    let bid_y = thread::block_idx_y();

    let block_row = bid_y * BM as u32;
    let block_col = bid_x * BN as u32;

    let warp_id = tid / 32;
    let lane = tid % 32;
    let wy = warp_id / 2;
    let wx = warp_id % 2;
    let group = lane / 4;
    let tg = lane % 4;
    let tg2 = tg * 2;

    let wy_off = wy * WM as u32;
    let wx_off = wx * WN as u32;

    // Initialize accumulators
    let mut acc = [[0.0f32; 4]; REG_M * REG_N]; // 2*4 = 8 tiles × 4 regs = 32 f32

    let smem_a_ptr = SMEM_A.as_ptr() as *const u16;
    let smem_b_ptr = SMEM_B.as_ptr() as *const u16;

    // K-loop
    let mut t: u32 = 0;
    while t < k {
        // ── Load A tile [64×16] into shared memory ──
        // 128 threads, 1024 elements → 8 per thread
        let tid_x8 = tid * 8;
        let a_tile_row = tid_x8 / BK as u32;
        let a_tile_col = tid_x8 % BK as u32;
        let a_glob_row = block_row + a_tile_row;
        let a_glob_col = t + a_tile_col;

        // Load 8 f16 elements (could use cp.async later)
        for j in 0..8u32 {
            let lin = tid_x8 + j;
            let sw = swizzle(lin);
            let src_idx = (a_glob_row * k + a_glob_col + j) as usize;
            SMEM_A[sw as usize].write(mat_a[src_idx]);
        }

        // ── Load B tile [16×64] into shared memory ──
        let b_tile_row = tid_x8 / BN as u32;
        let b_tile_col = tid_x8 % BN as u32;
        let b_glob_row = t + b_tile_row;
        let b_glob_col = block_col + b_tile_col;

        for j in 0..8u32 {
            let lin = tid_x8 + j;
            let sw = swizzle(lin);
            let src_idx = (b_glob_row * n + b_glob_col + j) as usize;
            SMEM_B[sw as usize].write(mat_b[src_idx]);
        }

        thread::sync_threads();

        // ── Load A fragments and compute ──
        // CubeK ordering: load all A, then per-N: load B + MMA all M
        let mut a_frags = [[0u32; 4]; REG_M];
        for rm in 0..REG_M as u32 {
            let rm_off = wy_off + rm * MMA_M as u32;
            a_frags[rm as usize] = [
                ld_shared_u32(smem_a_ptr, (rm_off + group) * BK as u32 + tg2),
                ld_shared_u32(smem_a_ptr, (rm_off + group + 8) * BK as u32 + tg2),
                ld_shared_u32(smem_a_ptr, (rm_off + group) * BK as u32 + tg2 + 8),
                ld_shared_u32(smem_a_ptr, (rm_off + group + 8) * BK as u32 + tg2 + 8),
            ];
        }

        for rn in 0..REG_N as u32 {
            let b_col = wx_off + rn * MMA_N as u32 + group;

            // Load B fragment: pack two strided f16 into u32
            let k0 = tg2;
            let k1 = tg2 + 1;
            let k8 = tg2 + 8;
            let k9 = tg2 + 9;

            let bv0_lo = ld_shared_f16(smem_b_ptr, k0 * BN as u32 + b_col);
            let bv0_hi = ld_shared_f16(smem_b_ptr, k1 * BN as u32 + b_col);
            let bv1_lo = ld_shared_f16(smem_b_ptr, k8 * BN as u32 + b_col);
            let bv1_hi = ld_shared_f16(smem_b_ptr, k9 * BN as u32 + b_col);

            let b_frag: [u32; 2] = [
                (bv0_lo as u32) | ((bv0_hi as u32) << 16),
                (bv1_lo as u32) | ((bv1_hi as u32) << 16),
            ];

            for rm in 0..REG_M as u32 {
                let acc_idx = rm as usize * REG_N + rn as usize;
                acc[acc_idx] = mma_sync(a_frags[rm as usize], b_frag, acc[acc_idx]);
            }
        }

        thread::sync_threads();
        t += BK as u32;
    }

    // ── Store C ──
    for rm in 0..REG_M as u32 {
        for rn in 0..REG_N as u32 {
            let acc_idx = rm as usize * REG_N + rn as usize;
            for d in 0..4u32 {
                let mma_row_off = if d < 2 { group } else { group + 8 };
                let mma_col_off = if d % 2 == 0 { tg2 } else { tg2 + 1 };

                let c_row = block_row + wy_off + rm * MMA_M as u32 + mma_row_off;
                let c_col = block_col + wx_off + rn * MMA_N as u32 + mma_col_off;
                let c_idx = (c_row * n + c_col) as usize;
                *mat_c.add(c_idx) = acc[acc_idx][d as usize];
            }
        }
    }
}
