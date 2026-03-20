use crate::{PtxBuilder, Reg};
use crate::config::GemmConfig;

/// Shared memory layout for double-buffered GEMM.
pub struct SmemLayout {
    pub a_offset: [u32; 2],  // byte offset of A in each buffer
    pub b_offset: [u32; 2],  // byte offset of B in each buffer
    pub buf_stride: u32,     // bytes per buffer
    pub total: u32,
}

impl SmemLayout {
    pub fn new(c: &GemmConfig) -> Self {
        let a_bytes = c.smem_a_bytes();
        let b_bytes = c.smem_b_bytes();
        let stride = a_bytes + b_bytes;
        Self {
            a_offset: [0, stride],
            b_offset: [a_bytes, stride + a_bytes],
            buf_stride: stride,
            total: stride * c.num_stages,
        }
    }
}

/// Registers holding cp.async shared memory destination addresses.
pub struct CpAsyncAddrs {
    /// Swizzled smem byte offsets for A chunks (relative to buffer A start)
    pub a_offsets: Vec<Reg>,
    /// Swizzled smem byte offsets for B chunks (relative to buffer B start)
    pub b_offsets: Vec<Reg>,
}

/// Emits the swizzle computation for cp.async store destinations.
/// Returns the swizzled byte offsets (relative to each buffer's A/B region).
pub fn emit_cpasync_swizzle(ptx: &mut PtxBuilder, _c: &GemmConfig, tid: Reg) -> CpAsyncAddrs {
    // Each thread handles 16 bytes per cp.async. With 128 threads:
    // A tile = 4096 bytes = 2 chunks of 2048 bytes (128 threads * 16 bytes)
    // B tile = 4096 bytes = 2 chunks of 2048 bytes
    //
    // tid_x16 = tid * 16 (element index base for this thread)
    // For chunk c: base_elem = tid*16 + c*8 (8 f16 = 16 bytes)
    //
    // Swizzle: byte = base_elem * 2; swizzled = byte ^ ((byte & 0x380) >> 3)

    let tid_x16 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x16, tid, 4);

    let mut a_offsets = Vec::new();
    let mut b_offsets = Vec::new();

    for chunk in 0..2u32 {
        let base = ptx.regs.alloc_b32();
        ptx.add_s32_imm(base, tid_x16, (chunk * 8) as i32);
        a_offsets.push(emit_swizzle_bytes(ptx, base));
    }
    for chunk in 0..2u32 {
        let base = ptx.regs.alloc_b32();
        ptx.add_s32_imm(base, tid_x16, (chunk * 8) as i32);
        b_offsets.push(emit_swizzle_bytes(ptx, base));
    }

    CpAsyncAddrs { a_offsets, b_offsets }
}

/// Registers holding ldmatrix source addresses (relative to buffer A/B start).
pub struct LdmatrixAddrs {
    /// A offsets: [ki][rm] → swizzled byte offset
    pub a_offsets: Vec<Vec<Reg>>,
    /// B offsets: [rn] → byte offset for ldmatrix.trans
    pub b_offsets: Vec<Reg>,
}

/// Emits ldmatrix address precomputation.
pub fn emit_ldmatrix_addrs(
    ptx: &mut PtxBuilder, c: &GemmConfig,
    lane: Reg, warp_id: Reg,
) -> LdmatrixAddrs {
    // A ldmatrix (non-transposed): row = rm*MMA_M + lane%16, col = ki*MMA_K
    let lane_mod16 = ptx.regs.alloc_b32();
    ptx.and_b32(lane_mod16, lane, 15);

    let mut a_offsets = Vec::new();
    for ki in 0..c.k_iters() {
        let mut ki_offsets = Vec::new();
        for rm in 0..c.reg_m() {
            let row = ptx.regs.alloc_b32();
            ptx.add_s32_imm(row, lane_mod16, (rm * c.mma_m) as i32);
            let elem = ptx.regs.alloc_b32();
            // elem = row * BK + ki * MMA_K
            ptx.shl_b32(elem, row, c.bk.trailing_zeros()); // row * BK (BK is power of 2)
            if ki > 0 {
                ptx.add_s32_imm(elem, elem, (ki * c.mma_k) as i32);
            }
            ki_offsets.push(emit_swizzle_bytes(ptx, elem));
        }
        a_offsets.push(ki_offsets);
    }

    // B ldmatrix.trans: addr = ((lane & 7) ^ col_group) << 4 | (lane << 7)
    // col_group = warp_id * 2 + rn (for 1x4 layout, warp_id = wx along N)
    let lane_mod8 = ptx.regs.alloc_b32();
    ptx.and_b32(lane_mod8, lane, 7);
    let lane_col = ptx.regs.alloc_b32();
    ptx.shl_b32(lane_col, lane_mod8, 4);
    let row_byte = ptx.regs.alloc_b32();
    ptx.shl_b32(row_byte, lane, 7);
    let wx2 = ptx.regs.alloc_b32();
    ptx.shl_b32(wx2, warp_id, 1);

    let mut b_offsets = Vec::new();
    for rn in 0..c.reg_n() {
        let g = ptx.regs.alloc_b32();
        ptx.add_s32_imm(g, wx2, rn as i32);
        let g_byte = ptx.regs.alloc_b32();
        ptx.shl_b32(g_byte, g, 4);
        let col_xor = ptx.regs.alloc_b32();
        ptx.xor_b32(col_xor, lane_col, g_byte);
        let addr = ptx.regs.alloc_b32();
        ptx.or_b32(addr, col_xor, row_byte);
        b_offsets.push(addr);
    }

    LdmatrixAddrs { a_offsets, b_offsets }
}

/// Computes swizzled byte offset from element index.
/// swizzled_byte = byte ^ ((byte & 0x380) >> 3) where byte = elem * 2
fn emit_swizzle_bytes(ptx: &mut PtxBuilder, elem: Reg) -> Reg {
    let byte = ptx.regs.alloc_b32();
    ptx.shl_b32(byte, elem, 1);
    let masked = ptx.regs.alloc_b32();
    ptx.and_b32(masked, byte, 0x380);
    let shifted = ptx.regs.alloc_b32();
    ptx.shr_u32(shifted, masked, 3);
    let swizzled = ptx.regs.alloc_b32();
    ptx.xor_b32(swizzled, byte, shifted);
    swizzled
}
