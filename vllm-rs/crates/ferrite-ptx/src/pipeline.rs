use crate::{PtxBuilder, Reg};
use crate::config::GemmConfig;
use crate::atoms::{CopyAtom, TransformAtom, MmaAtom};
use crate::gemm::{GemmSetup, AccumulatorMap};

// Hard-coded register tile dimensions matching the existing GEMM.
const REG_M: u32 = 2;
const REG_N: u32 = 4;

// ═══════════════════════════════════════════════════════════════════════════
// MainloopPipeline — THE single K-loop implementation
//
// All GEMM-based kernels use this. The pipeline handles:
// - Software pipelining with configurable stages (double-buffering)
// - Prologue tile loads
// - K-loop with interleaved: ldmatrix, transform, MMA, async copy
// - Epilogue drain
//
// The atoms don't know about double-buffering, pipeline stages, or sync.
// The pipeline does ALL scheduling.
// ═══════════════════════════════════════════════════════════════════════════

pub struct MainloopPipeline {
    pub stages: u32,
}

/// Result of pipeline emission: accumulators + k_counter register
pub struct PipelineResult {
    pub acc: AccumulatorMap,
    pub k_counter: Reg,
}

impl MainloopPipeline {
    pub fn new(stages: u32) -> Self {
        assert!(stages == 2, "Only 2-stage pipeline supported currently");
        Self { stages }
    }

    /// Emit the complete mainloop: prologue, K-loop, epilogue.
    ///
    /// Returns the accumulator map after the K-loop.
    ///
    /// Parameters:
    /// - `copy_a`, `copy_b`: how tiles are loaded from global to shared memory
    /// - `transform_a`: how A fragments are transformed after ldmatrix
    /// - `mma`: how tensor cores consume fragments
    /// - `ga0, ga1`: A global pointers for chunk 0/1
    /// - `gb0, gb1`: B global pointers for chunk 0/1
    /// - `a_cp_off, b_cp_off`: swizzled smem offsets for A and B
    /// - `n_param, k_param`: matrix dimensions
    /// - `extra_advance`: optional closure for additional per-iteration state advance
    pub fn emit(
        &self,
        ptx: &mut PtxBuilder,
        c: &GemmConfig,
        setup: &GemmSetup,
        // Atoms
        copy_a: &dyn CopyAtom,
        copy_b: &dyn CopyAtom,
        transform_a: &dyn TransformAtom,
        mma: &dyn MmaAtom,
        // Memory addresses
        ga0: Reg, ga1: Reg,
        a_cp_off: Reg,
        gb0: Reg, gb1: Reg,
        b_cp_off: Reg,
        // Params
        n_param: Reg,
        k_param: Reg,
        // Optional extra advance
        extra_advance: Option<&dyn Fn(&mut PtxBuilder)>,
    ) -> PipelineResult {
        let smem_base = setup.smem_base;
        let tid = setup.tid;

        let a_tile_bytes = c.smem_a_bytes();     // 4096
        let buf_stride = a_tile_bytes;           // 4096
        let b_start = (a_tile_bytes * c.num_stages) as i32; // 8192

        let tid_x16 = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_x16, tid, 4);

        // ═══════════════════════════════════════════════════════════════
        // A smem addresses
        // ═══════════════════════════════════════════════════════════════
        let a_st0 = ptx.regs.alloc_b32();
        ptx.add_s32(a_st0, smem_base, a_cp_off);
        let a_st1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(a_st1, a_st0, 2048);

        // B smem addresses
        let b_cp_base_smem = ptx.regs.alloc_b32();
        ptx.add_s32(b_cp_base_smem, smem_base, b_cp_off);
        let b_cp0 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b_cp0, b_cp_base_smem, b_start);
        let b_cp1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b_cp1, b_cp_base_smem, b_start + 2048);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Prologue: load tile 0 into buffer 0
        // ═══════════════════════════════════════════════════════════════
        ptx.comment("Pipeline prologue: load tile 0 into buffer 0");
        let p_tile0 = ptx.regs.alloc_pred();
        ptx.setp_gt_s32_imm(p_tile0, k_param, 0);
        let sz0 = ptx.regs.alloc_b32();
        ptx.selp_b32(sz0, 16, 0, p_tile0);

        // A tile 0
        copy_a.emit_async_copy(ptx, a_st0, ga0, sz0);
        copy_a.emit_async_copy(ptx, a_st1, ga1, sz0);
        copy_a.emit_commit(ptx);

        // B tile 0
        copy_b.emit_async_copy(ptx, b_cp0, gb0, sz0);
        copy_b.emit_async_copy(ptx, b_cp1, gb1, sz0);
        copy_b.emit_commit(ptx);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Advance global pointers for tile 1
        // ═══════════════════════════════════════════════════════════════
        ptx.comment("Advance global pointers for tile 1");
        let ga0_t1 = ptx.regs.alloc_b64();
        ptx.add_s64_imm(ga0_t1, ga0, (c.bk * 2) as i64);
        let ga1_t1 = ptx.regs.alloc_b64();
        ptx.add_s64_imm(ga1_t1, ga1, (c.bk * 2) as i64);

        let b_stride_val = ptx.regs.alloc_b32();
        ptx.shl_b32(b_stride_val, n_param, c.bk.trailing_zeros());
        let two_reg = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(two_reg, 2);
        let b_stride_bytes = ptx.regs.alloc_b64();
        ptx.mul_wide_s32(b_stride_bytes, b_stride_val, two_reg);
        let gb0_t1 = ptx.regs.alloc_b64();
        ptx.add_s64(gb0_t1, gb0, b_stride_bytes);
        let gb1_t1 = ptx.regs.alloc_b64();
        ptx.add_s64(gb1_t1, gb1, b_stride_bytes);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Prologue: load tile 1 into buffer 1
        // ═══════════════════════════════════════════════════════════════
        ptx.comment("Pipeline prologue: load tile 1 into buffer 1");
        let p_tile1 = ptx.regs.alloc_pred();
        ptx.setp_gt_s32_imm(p_tile1, k_param, c.bk as i32);

        // Advance extra state for tile 1
        if let Some(adv) = extra_advance {
            adv(ptx);
        }

        ptx.bar_sync(0);

        // Buffer 1 A smem addresses
        let a_st0_b1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(a_st0_b1, a_st0, buf_stride as i32);
        let a_st1_b1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(a_st1_b1, a_st1, buf_stride as i32);

        let sz1 = ptx.regs.alloc_b32();
        ptx.selp_b32(sz1, 16, 0, p_tile1);

        // A tile 1
        copy_a.emit_async_copy(ptx, a_st0_b1, ga0_t1, sz1);
        copy_a.emit_async_copy(ptx, a_st1_b1, ga1_t1, sz1);
        copy_a.emit_commit(ptx);

        // Buffer 1 B
        let b_cp0_b1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b_cp0_b1, b_cp_base_smem, b_start + buf_stride as i32);
        let b_cp1_b1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b_cp1_b1, b_cp_base_smem, b_start + buf_stride as i32 + 2048);
        copy_b.emit_async_copy(ptx, b_cp0_b1, gb0_t1, sz1);
        copy_b.emit_async_copy(ptx, b_cp1_b1, gb1_t1, sz1);
        copy_b.emit_commit(ptx);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Precompute ldmatrix addresses
        // ═══════════════════════════════════════════════════════════════
        ptx.comment("ldmatrix swizzle offsets");

        let tid_x64 = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_x64, tid, 6);
        let a_ld_r0 = ptx.regs.alloc_b32();
        ptx.and_b32(a_ld_r0, tid_x64, 960);
        let tid_x8 = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_x8, tid, 3);
        let a_ld_r1 = ptx.regs.alloc_b32();
        ptx.and_b32(a_ld_r1, tid_x8, 48);
        let a_ld_swiz = ptx.regs.alloc_b32();
        ptx.and_b32(a_ld_swiz, tid, 16);
        let a_ld_half = ptx.regs.alloc_b32();
        ptx.and_b32(a_ld_half, tid_x16, 1024);
        let a_ld_comb = ptx.regs.alloc_b32();
        ptx.or_b32(a_ld_comb, a_ld_r0, a_ld_r1);
        let a_ld_xor = ptx.regs.alloc_b32();
        ptx.xor_b32(a_ld_xor, a_ld_comb, a_ld_swiz);
        let a_off_ki0 = ptx.regs.alloc_b32();
        ptx.or_b32(a_off_ki0, a_ld_xor, a_ld_half);
        let a_off_ki1 = ptx.regs.alloc_b32();
        ptx.xor_b32_imm(a_off_ki1, a_off_ki0, 32);

        let lane = setup.lane;
        let tid_x128 = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_x128, tid, 7);
        let b_ld_row = ptx.regs.alloc_b32();
        ptx.and_b32(b_ld_row, tid_x128, 3968);
        let lane_and7 = ptx.regs.alloc_b32();
        ptx.and_b32(lane_and7, lane, 7);
        let b_ld_col = ptx.regs.alloc_b32();
        ptx.shl_b32(b_ld_col, lane_and7, 4);
        let tid_shr1 = ptx.regs.alloc_b32();
        ptx.shr_u32(tid_shr1, tid, 1);
        let b_ld_swiz = ptx.regs.alloc_b32();
        ptx.and_b32(b_ld_swiz, tid_shr1, 16);
        let b_col_xor = ptx.regs.alloc_b32();
        ptx.xor_b32(b_col_xor, b_ld_col, b_ld_swiz);
        let b_off_rn0 = ptx.regs.alloc_b32();
        ptx.or_b32(b_off_rn0, b_col_xor, b_ld_row);
        let b_off_rn1 = ptx.regs.alloc_b32();
        ptx.xor_b32_imm(b_off_rn1, b_off_rn0, 32);
        let b_off_rn2 = ptx.regs.alloc_b32();
        ptx.xor_b32_imm(b_off_rn2, b_off_rn0, 64);
        let b_off_rn3 = ptx.regs.alloc_b32();
        ptx.xor_b32_imm(b_off_rn3, b_off_rn0, 96);
        let b_off = [b_off_rn0, b_off_rn1, b_off_rn2, b_off_rn3];
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Advance global ptrs to tile 2 position (for loop's loads)
        // ═══════════════════════════════════════════════════════════════
        ptx.comment("Advance global ptrs to tile 2 position");
        let ga0_loop = ptx.regs.alloc_b64();
        ptx.add_s64_imm(ga0_loop, ga0_t1, (c.bk * 2) as i64);
        let ga1_loop = ptx.regs.alloc_b64();
        ptx.add_s64_imm(ga1_loop, ga1_t1, (c.bk * 2) as i64);
        let gb0_loop = ptx.regs.alloc_b64();
        ptx.add_s64(gb0_loop, gb0_t1, b_stride_bytes);
        let gb1_loop = ptx.regs.alloc_b64();
        ptx.add_s64(gb1_loop, gb1_t1, b_stride_bytes);

        // Advance extra state for tile 2
        if let Some(adv) = extra_advance {
            adv(ptx);
        }
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // K>0 check and loop entry
        // ═══════════════════════════════════════════════════════════════
        ptx.w(&format!("@{p_tile0} bra \t$L_LOOP_ENTRY;"));
        ptx.bra_uni("$L_K0_FALLTHROUGH");
        ptx.blank();

        ptx.label("$L_LOOP_ENTRY");

        let k_minus_2bk = ptx.regs.alloc_b32();
        ptx.add_s32_imm(k_minus_2bk, k_param, -(2 * c.bk as i32));
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Transform prologue (e.g., load norm factors into registers)
        // ═══════════════════════════════════════════════════════════════
        transform_a.emit_prologue(ptx);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Initialize accumulators
        // ═══════════════════════════════════════════════════════════════
        ptx.comment("Initialize accumulators to 0.0f");
        let zero = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(zero, 0x00000000);

        let num_tiles = (REG_M * REG_N) as usize;
        let mut acc_regs: Vec<[Reg; 4]> = Vec::with_capacity(num_tiles);
        for _ in 0..num_tiles {
            let tile = [
                ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
            ];
            for &r in &tile {
                ptx.mov_b32(r, zero);
            }
            acc_regs.push(tile);
        }
        let acc = AccumulatorMap {
            regs: acc_regs,
            reg_m: REG_M,
            reg_n: REG_N,
        };
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Buffer toggle counters
        // ═══════════════════════════════════════════════════════════════
        let read_ctr = ptx.regs.alloc_b32();
        let write_ctr = ptx.regs.alloc_b32();
        ptx.w(&format!("mov.b32 \t{read_ctr}, 1;"));
        ptx.w(&format!("mov.b32 \t{write_ctr}, -1;"));

        let k_counter = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(k_counter, 0);
        ptx.blank();

        // Compute total async groups in flight
        let a_async = copy_a.async_groups_per_tile();
        let b_async = copy_b.async_groups_per_tile();
        let total_async_per_tile = a_async + b_async;
        let wait_count = total_async_per_tile;

        // ═══════════════════════════════════════════════════════════════
        // K-LOOP
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_KLOOP");

        let p_load = ptx.regs.alloc_pred();
        ptx.setp_lt_s32(p_load, k_counter, k_minus_2bk);

        // Toggle read buffer
        let read_next = ptx.regs.alloc_b32();
        ptx.add_s32_imm(read_next, read_ctr, 1);
        let p_reset_r = ptx.regs.alloc_pred();
        ptx.setp_gt_s32_imm(p_reset_r, read_next, 1);
        ptx.selp_b32_imm_reg(read_ctr, 0, read_next, p_reset_r);

        // Wait for async groups + barrier
        ptx.cp_async_wait_group(wait_count);
        ptx.bar_sync(0);

        // Compute read buffer base
        let read_buf_off = ptx.regs.alloc_b32();
        ptx.shl_b32(read_buf_off, read_ctr, 12);
        let buf_base = ptx.regs.alloc_b32();
        ptx.add_s32(buf_base, smem_base, read_buf_off);
        ptx.blank();

        // ── ldmatrix A ──
        // ── ldmatrix.trans B (all 4 loads — B frags used across both ki) ──
        ptx.comment("ldmatrix.trans B -- 4 loads");
        let mut b_frags: Vec<[Reg; 4]> = Vec::new();
        for rn in 0..REG_N as usize {
            let b_addr = ptx.regs.alloc_b32();
            ptx.add_s32(b_addr, buf_base, b_off[rn]);
            let frag = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                        ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];
            ptx.ldmatrix_x4_trans(frag, b_addr, Some(b_start));
            b_frags.push(frag);
        }
        ptx.blank();

        // ── ki=0: load A, transform, MMA (A frags die after MMA) ──
        ptx.comment("ki=0: ldmatrix A + transform + MMA");
        transform_a.emit_k_setup(ptx, 0);
        let a_addr_ki0 = ptx.regs.alloc_b32();
        ptx.add_s32(a_addr_ki0, buf_base, a_off_ki0);

        let mut a_frag_rm0 = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                               ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];
        ptx.ldmatrix_x4(a_frag_rm0, a_addr_ki0, None);
        transform_a.emit_transform(ptx, &mut a_frag_rm0, 0, 0);
        let mut a_frag_rm1 = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                               ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];
        ptx.ldmatrix_x4(a_frag_rm1, a_addr_ki0, Some(2048));
        transform_a.emit_transform(ptx, &mut a_frag_rm1, 0, 1);

        for rn in 0..REG_N as usize {
            let ai = 0 * REG_N as usize + rn;
            let mut acc_tile = acc.regs[ai];
            mma.emit_mma(ptx, a_frag_rm0, [b_frags[rn][0], b_frags[rn][1]], &mut acc_tile);
        }
        for rn in 0..REG_N as usize {
            let ai = 1 * REG_N as usize + rn;
            let mut acc_tile = acc.regs[ai];
            mma.emit_mma(ptx, a_frag_rm1, [b_frags[rn][0], b_frags[rn][1]], &mut acc_tile);
        }
        ptx.blank();

        // ── ki=1: load A, transform, MMA (reuse same reg names — ki=0 A frags are dead) ──
        ptx.comment("ki=1: ldmatrix A + transform + MMA");
        transform_a.emit_k_setup(ptx, 1);
        let a_addr_ki1 = ptx.regs.alloc_b32();
        ptx.add_s32(a_addr_ki1, buf_base, a_off_ki1);

        // Reuse the same fragment register variables (ptxas sees them as separate defs)
        ptx.ldmatrix_x4(a_frag_rm0, a_addr_ki1, None);
        transform_a.emit_transform(ptx, &mut a_frag_rm0, 1, 0);
        ptx.ldmatrix_x4(a_frag_rm1, a_addr_ki1, Some(2048));
        transform_a.emit_transform(ptx, &mut a_frag_rm1, 1, 1);

        for rn in 0..REG_N as usize {
            let ai = 0 * REG_N as usize + rn;
            let mut acc_tile = acc.regs[ai];
            mma.emit_mma(ptx, a_frag_rm0, [b_frags[rn][2], b_frags[rn][3]], &mut acc_tile);
        }
        for rn in 0..REG_N as usize {
            let ai = 1 * REG_N as usize + rn;
            let mut acc_tile = acc.regs[ai];
            mma.emit_mma(ptx, a_frag_rm1, [b_frags[rn][2], b_frags[rn][3]], &mut acc_tile);
        }
        ptx.blank();

        // ── Advance B global pointers ──
        let gb0_next = ptx.regs.alloc_b64();
        ptx.add_s64(gb0_next, gb0_loop, b_stride_bytes);
        let gb1_next = ptx.regs.alloc_b64();
        ptx.add_s64(gb1_next, gb1_loop, b_stride_bytes);

        // ── Toggle write buffer ──
        let write_next = ptx.regs.alloc_b32();
        ptx.add_s32_imm(write_next, write_ctr, 1);
        let p_reset_w = ptx.regs.alloc_pred();
        ptx.setp_gt_s32_imm(p_reset_w, write_next, 1);
        ptx.selp_b32_imm_reg(write_ctr, 0, write_next, p_reset_w);

        let write_buf_off = ptx.regs.alloc_b32();
        ptx.shl_b32(write_buf_off, write_ctr, 12);
        let write_buf_base = ptx.regs.alloc_b32();
        ptx.add_s32(write_buf_base, smem_base, write_buf_off);

        ptx.bar_sync(0);

        // ── Predicated loads for next tile ──
        ptx.comment("Predicated loads for next tile");
        let cp_size = ptx.regs.alloc_b32();
        ptx.selp_b32(cp_size, 16, 0, p_load);

        // A next tile
        let a_dst0 = ptx.regs.alloc_b32();
        ptx.add_s32(a_dst0, write_buf_base, a_cp_off);
        let a_dst1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(a_dst1, a_dst0, 2048);
        copy_a.emit_async_copy(ptx, a_dst0, ga0_loop, cp_size);
        copy_a.emit_async_copy(ptx, a_dst1, ga1_loop, cp_size);
        copy_a.emit_commit(ptx);

        // B next tile
        let b_dst_base = ptx.regs.alloc_b32();
        ptx.add_s32(b_dst_base, write_buf_base, b_cp_off);
        let b_dst0 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b_dst0, b_dst_base, b_start);
        let b_dst1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b_dst1, b_dst_base, b_start + 2048);
        copy_b.emit_async_copy(ptx, b_dst0, gb0_loop, cp_size);
        copy_b.emit_async_copy(ptx, b_dst1, gb1_loop, cp_size);
        copy_b.emit_commit(ptx);
        ptx.blank();

        // ── Advance loop state ──
        ptx.comment("Advance loop state");
        ptx.add_s32_imm(k_counter, k_counter, c.bk as i32);
        ptx.add_s64_imm(ga0_loop, ga0_loop, (c.bk * 2) as i64);
        ptx.add_s64_imm(ga1_loop, ga1_loop, (c.bk * 2) as i64);
        ptx.mov_b64(gb0_loop, gb0_next);
        ptx.mov_b64(gb1_loop, gb1_next);

        // Extra advance (e.g., k_offset for transform)
        if let Some(adv) = extra_advance {
            adv(ptx);
        }

        let p_loop = ptx.regs.alloc_pred();
        ptx.setp_lt_s32(p_loop, k_counter, k_param);
        ptx.w(&format!("@{p_loop} bra \t$L_KLOOP;"));
        ptx.blank();

        ptx.bra_uni("$L_EPILOGUE");
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // K=0 fallthrough
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_K0_FALLTHROUGH");
        let zero2 = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(zero2, 0x00000000);
        for tile in &acc.regs {
            for &r in tile {
                ptx.mov_b32(r, zero2);
            }
        }
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Epilogue
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_EPILOGUE");
        ptx.cp_async_wait_group(0);
        ptx.bar_sync(0);
        ptx.blank();

        PipelineResult { acc, k_counter }
    }
}
