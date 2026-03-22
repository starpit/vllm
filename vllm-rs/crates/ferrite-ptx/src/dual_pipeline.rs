use crate::atoms::{CopyAtom, MmaAtom, TransformAtom};
use crate::config::GemmConfig;
use crate::gemm::{AccumulatorMap, GemmSetup};
use crate::{PtxBuilder, Reg};

// ═══════════════════════════════════════════════════════════════════════════
// DualMainloopPipeline — dual GEMM K-loop following CUTLASS dual_gemm
//
// Shared A fragments, separate B0 (gate) and B1 (up) fragments.
// Triple-buffered smem: A + B0 + B1 in separate regions.
// K-loop: load A+B0+B1 → ALL GEMM0 MMAs → ALL GEMM1 MMAs → cp.async → cycle
// Two accumulator sets: acc0 (gate) and acc1 (up).
// ═══════════════════════════════════════════════════════════════════════════

pub struct DualMainloopPipeline {
    pub stages: u32,
}

/// Result of dual pipeline emission: two accumulator maps + k_counter register
pub struct DualPipelineResult {
    pub acc0: AccumulatorMap, // gate GEMM result
    pub acc1: AccumulatorMap, // up GEMM result
    pub k_counter: Reg,
}

impl DualMainloopPipeline {
    pub fn new(stages: u32) -> Self {
        assert!(stages >= 2, "Pipeline requires at least 2 stages");
        Self { stages }
    }

    /// Emit the complete dual mainloop: prologue, K-loop, epilogue.
    ///
    /// Returns two accumulator maps (gate, up) after the K-loop.
    ///
    /// Parameters:
    /// - `copy_a`, `copy_b0`, `copy_b1`: how tiles are loaded from global to shared memory
    /// - `transform`: how A fragments are transformed after ldmatrix
    /// - `mma`: how tensor cores consume fragments
    /// - `ga_chunks`: global A pointers, one per cp.async chunk
    /// - `a_cp_off`: swizzled smem offset for A
    /// - `gb0_chunks`: global B0 (gate) pointers, one per cp.async chunk
    /// - `b0_cp_off`: swizzled smem offset for B0
    /// - `gb1_chunks`: global B1 (up) pointers, one per cp.async chunk
    /// - `b1_cp_off`: swizzled smem offset for B1
    /// - `n_param, k_param`: matrix dimensions
    /// - `extra_k_advance`: optional closure for additional per-iteration state advance
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &self,
        ptx: &mut PtxBuilder,
        c: &GemmConfig,
        setup: &GemmSetup,
        // Atoms
        copy_a: &dyn CopyAtom,
        copy_b0: &dyn CopyAtom,
        copy_b1: &dyn CopyAtom,
        transform: &dyn TransformAtom,
        mma: &dyn MmaAtom,
        // Memory addresses
        ga_chunks: Vec<Reg>,
        a_cp_off: Reg,
        gb0_chunks: Vec<Reg>,
        b0_cp_off: Reg,
        gb1_chunks: Vec<Reg>,
        b1_cp_off: Reg,
        // Params
        n_param: Reg,
        k_param: Reg,
        // Optional extra advance
        extra_k_advance: Option<&dyn Fn(&mut PtxBuilder)>,
    ) -> DualPipelineResult {
        let smem_base = setup.smem_base;
        let tid = setup.tid;
        let stages = self.stages;

        // kWarpGemmIterations = BK / MMA_K
        let k_warp_iters = c.bk / c.mma_k; // e.g. 32/16=2 or 64/16=4
        assert!(k_warp_iters >= 2, "kWarpGemmIterations must be >= 2");

        let a_tile_bytes = c.smem_a_bytes(); // BM*BK*2
        let b_tile_bytes = c.smem_b_bytes(); // BK*BN*2

        // Dual GEMM smem layout:
        //   A:  offset 0,                     size = a_tile_bytes * stages
        //   B0: offset a_tile_bytes * stages, size = b_tile_bytes * stages
        //   B1: offset (a_tile_bytes + b_tile_bytes) * stages, size = b_tile_bytes * stages
        let a_smem_total = a_tile_bytes * stages;
        let b0_start = a_smem_total as i32;
        let b1_start = (a_smem_total + b_tile_bytes * stages) as i32;

        // Number of 2048-byte cp.async chunks per tile.
        let cp_chunks_a = a_tile_bytes / 2048;
        let cp_chunks_b = b_tile_bytes / 2048;

        // Number of ldmatrix.x4.trans groups for B per tile.
        let b_ld_groups = k_warp_iters / 2; // 1 for BK=32, 2 for BK=64

        let tid_x16 = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_x16, tid, 4);

        assert_eq!(
            ga_chunks.len(),
            cp_chunks_a as usize,
            "Expected {} A global pointers, got {}",
            cp_chunks_a,
            ga_chunks.len()
        );
        assert_eq!(
            gb0_chunks.len(),
            cp_chunks_b as usize,
            "Expected {} B0 global pointers, got {}",
            cp_chunks_b,
            gb0_chunks.len()
        );
        assert_eq!(
            gb1_chunks.len(),
            cp_chunks_b as usize,
            "Expected {} B1 global pointers, got {}",
            cp_chunks_b,
            gb1_chunks.len()
        );

        // ═══════════════════════════════════════════════════════════════
        // A smem base addresses for cp.async (buffer 0, all chunks)
        // ═══════════════════════════════════════════════════════════════
        let mut a_st: Vec<Reg> = Vec::with_capacity(cp_chunks_a as usize);
        let a_st0 = ptx.regs.alloc_b32();
        ptx.add_s32(a_st0, smem_base, a_cp_off);
        a_st.push(a_st0);
        for i in 1..cp_chunks_a {
            let r = ptx.regs.alloc_b32();
            ptx.add_s32_imm(r, a_st0, (i * 2048) as i32);
            a_st.push(r);
        }

        let ga_all = ga_chunks;

        // B0 smem base addresses for cp.async (buffer 0, all chunks)
        let b0_cp_base_smem = ptx.regs.alloc_b32();
        ptx.add_s32(b0_cp_base_smem, smem_base, b0_cp_off);
        let mut b0_cp: Vec<Reg> = Vec::with_capacity(cp_chunks_b as usize);
        let b0_cp0 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b0_cp0, b0_cp_base_smem, b0_start);
        b0_cp.push(b0_cp0);
        for i in 1..cp_chunks_b {
            let r = ptx.regs.alloc_b32();
            ptx.add_s32_imm(r, b0_cp0, (i * 2048) as i32);
            b0_cp.push(r);
        }

        // B1 smem base addresses for cp.async (buffer 0, all chunks)
        let b1_cp_base_smem = ptx.regs.alloc_b32();
        ptx.add_s32(b1_cp_base_smem, smem_base, b1_cp_off);
        let mut b1_cp: Vec<Reg> = Vec::with_capacity(cp_chunks_b as usize);
        let b1_cp0 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b1_cp0, b1_cp_base_smem, b1_start);
        b1_cp.push(b1_cp0);
        for i in 1..cp_chunks_b {
            let r = ptx.regs.alloc_b32();
            ptx.add_s32_imm(r, b1_cp0, (i * 2048) as i32);
            b1_cp.push(r);
        }

        let gb0_all = gb0_chunks;
        let gb1_all = gb1_chunks;
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // B stride for advancing global B pointers between K-tiles
        // b_stride_bytes = BK * N * 2
        // ═══════════════════════════════════════════════════════════════
        let b_stride_val = ptx.regs.alloc_b32();
        ptx.shl_b32(b_stride_val, n_param, c.bk.trailing_zeros());
        let two_reg = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(two_reg, 2);
        let b_stride_bytes = ptx.regs.alloc_b64();
        ptx.mul_wide_s32(b_stride_bytes, b_stride_val, two_reg);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // PROLOGUE: Load (stages-1) tiles into shared memory
        // ═══════════════════════════════════════════════════════════════

        // Allocate loop global pointers
        let mut ga_loop: Vec<Reg> = Vec::new();
        for &g in &ga_all {
            let r = ptx.regs.alloc_b64();
            ptx.mov_b64(r, g);
            ga_loop.push(r);
        }
        let mut gb0_loop: Vec<Reg> = Vec::new();
        for &g in &gb0_all {
            let r = ptx.regs.alloc_b64();
            ptx.mov_b64(r, g);
            gb0_loop.push(r);
        }
        let mut gb1_loop: Vec<Reg> = Vec::new();
        for &g in &gb1_all {
            let r = ptx.regs.alloc_b64();
            ptx.mov_b64(r, g);
            gb1_loop.push(r);
        }

        for stage in 0..stages - 1 {
            ptx.comment(&format!(
                "Dual pipeline prologue: load tile {stage} into buffer {stage}"
            ));

            let p_tile = ptx.regs.alloc_pred();
            ptx.setp_gt_s32_imm(p_tile, k_param, (stage * c.bk) as i32);
            let sz = ptx.regs.alloc_b32();
            ptx.selp_b32(sz, 16, 0, p_tile);

            // A tile into buffer `stage`
            for i in 0..cp_chunks_a as usize {
                let a_dst = ptx.regs.alloc_b32();
                ptx.add_s32_imm(a_dst, a_st[i], (stage * a_tile_bytes) as i32);
                copy_a.emit_async_copy(ptx, a_dst, ga_loop[i], sz);
            }
            copy_a.emit_commit(ptx);

            // B0 tile into buffer `stage`
            for i in 0..cp_chunks_b as usize {
                let b0_dst = ptx.regs.alloc_b32();
                ptx.add_s32_imm(b0_dst, b0_cp[i], (stage * b_tile_bytes) as i32);
                copy_b0.emit_async_copy(ptx, b0_dst, gb0_loop[i], sz);
            }
            copy_b0.emit_commit(ptx);

            // B1 tile into buffer `stage`
            for i in 0..cp_chunks_b as usize {
                let b1_dst = ptx.regs.alloc_b32();
                ptx.add_s32_imm(b1_dst, b1_cp[i], (stage * b_tile_bytes) as i32);
                copy_b1.emit_async_copy(ptx, b1_dst, gb1_loop[i], sz);
            }
            copy_b1.emit_commit(ptx);
            ptx.blank();

            // Advance global pointers to next tile
            ptx.comment(&format!("Advance global pointers for tile {}", stage + 1));
            for g in &ga_loop {
                ptx.add_s64_imm(*g, *g, (c.bk * 2) as i64);
            }
            for g in &gb0_loop {
                ptx.add_s64(*g, *g, b_stride_bytes);
            }
            for g in &gb1_loop {
                ptx.add_s64(*g, *g, b_stride_bytes);
            }
            // Advance extra state
            if let Some(adv) = extra_k_advance {
                adv(ptx);
            }
        }
        ptx.blank();

        let reg_m = c.reg_m();
        let reg_n = c.reg_n();

        // ═══════════════════════════════════════════════════════════════
        // Precompute ldmatrix swizzle offsets (constant across K-loop)
        // ═══════════════════════════════════════════════════════════════
        ptx.comment("ldmatrix swizzle offsets");

        // A ldmatrix offsets
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

        let mut a_off = vec![a_off_ki0];
        for ki in 1..k_warp_iters {
            let off = ptx.regs.alloc_b32();
            let xor_val = (ki % 2) * 32;
            let chunk_off = (ki / 2) * 4096;
            if chunk_off == 0 {
                ptx.xor_b32_imm(off, a_off_ki0, xor_val);
            } else if xor_val == 0 {
                ptx.add_s32_imm(off, a_off_ki0, chunk_off as i32);
            } else {
                let tmp = ptx.regs.alloc_b32();
                ptx.xor_b32_imm(tmp, a_off_ki0, xor_val);
                ptx.add_s32_imm(off, tmp, chunk_off as i32);
            }
            a_off.push(off);
        }

        // B ldmatrix offsets (shared formula for B0 and B1)
        let lane = setup.lane;
        let b_row_shift = (c.bn * 2).trailing_zeros();
        let b_row_mask = 31 * c.bn * 2;
        let tid_shifted_b = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_shifted_b, tid, b_row_shift);
        let b_ld_row = ptx.regs.alloc_b32();
        ptx.and_b32(b_ld_row, tid_shifted_b, b_row_mask);

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
        let b_off_base = ptx.regs.alloc_b32();
        ptx.or_b32(b_off_base, b_col_xor, b_ld_row);

        // Build b_off array for all rn values (same offsets for B0 and B1)
        let b_col_group_stride = c.b_col_group_stride();
        let mut b_off: Vec<Reg> = Vec::with_capacity(reg_n as usize);
        for rn in 0..reg_n {
            let group = rn / 4;
            let within = rn % 4;
            let xor_val = within * 32;
            let group_off = group * b_col_group_stride;
            let r = ptx.regs.alloc_b32();
            if xor_val == 0 && group_off == 0 {
                ptx.mov_b32(r, b_off_base);
            } else if group_off == 0 {
                ptx.xor_b32_imm(r, b_off_base, xor_val);
            } else if xor_val == 0 {
                ptx.add_s32_imm(r, b_off_base, group_off as i32);
            } else {
                let tmp = ptx.regs.alloc_b32();
                ptx.xor_b32_imm(tmp, b_off_base, xor_val);
                ptx.add_s32_imm(r, tmp, group_off as i32);
            }
            b_off.push(r);
        }
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // K>0 check and loop entry
        // ═══════════════════════════════════════════════════════════════
        let p_has_k = ptx.regs.alloc_pred();
        ptx.setp_gt_s32_imm(p_has_k, k_param, 0);
        ptx.w(&format!("@{p_has_k} bra \t$L_DUAL_LOOP_ENTRY;"));
        ptx.bra_uni("$L_DUAL_K0_FALLTHROUGH");
        ptx.blank();

        ptx.label("$L_DUAL_LOOP_ENTRY");

        // k_minus_stages_bk: used for predicated loads
        let k_minus_stages_bk = ptx.regs.alloc_b32();
        ptx.add_s32_imm(k_minus_stages_bk, k_param, -((stages * c.bk) as i32));
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Transform prologue (e.g., load norm factors into registers)
        // ═══════════════════════════════════════════════════════════════
        transform.emit_prologue(ptx);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Initialize TWO accumulator sets
        // ═══════════════════════════════════════════════════════════════
        ptx.comment(&format!(
            "Initialize dual accumulators to 0.0f (REG_M={}, REG_N={}, {} tiles x2)",
            reg_m,
            reg_n,
            reg_m * reg_n
        ));
        let zero = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(zero, 0x00000000);

        let num_tiles = (reg_m * reg_n) as usize;

        // Accumulator 0 (gate GEMM)
        let mut acc0_regs: Vec<[Reg; 4]> = Vec::with_capacity(num_tiles);
        for _ in 0..num_tiles {
            let tile = [
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ];
            for &r in &tile {
                ptx.mov_b32(r, zero);
            }
            acc0_regs.push(tile);
        }
        let acc0 = AccumulatorMap {
            regs: acc0_regs,
            reg_m,
            reg_n,
        };

        // Accumulator 1 (up GEMM)
        let mut acc1_regs: Vec<[Reg; 4]> = Vec::with_capacity(num_tiles);
        for _ in 0..num_tiles {
            let tile = [
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ];
            for &r in &tile {
                ptx.mov_b32(r, zero);
            }
            acc1_regs.push(tile);
        }
        let acc1 = AccumulatorMap {
            regs: acc1_regs,
            reg_m,
            reg_n,
        };
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Circular buffer state
        // ═══════════════════════════════════════════════════════════════
        let read_stage = ptx.regs.alloc_b32();
        let write_stage = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(read_stage, stages - 1);
        ptx.mov_b32_imm(write_stage, stages - 1);

        let k_counter = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(k_counter, 0);
        ptx.blank();

        // Compute total async groups in flight
        let a_async = copy_a.async_groups_per_tile();
        let b0_async = copy_b0.async_groups_per_tile();
        let b1_async = copy_b1.async_groups_per_tile();
        let total_async_per_tile = a_async + b0_async + b1_async;
        let wait_count = total_async_per_tile * (stages - 2);

        // ═══════════════════════════════════════════════════════════════
        // Pre-allocate K-loop temporary registers
        // ═══════════════════════════════════════════════════════════════
        let buf_base = ptx.regs.alloc_b32();
        let read_off = ptx.regs.alloc_b32();
        let write_buf_base = ptx.regs.alloc_b32();
        let write_off = ptx.regs.alloc_b32();
        let b0_addr = ptx.regs.alloc_b32(); // reused for B0 ldmatrix address
        let b1_addr = ptx.regs.alloc_b32(); // reused for B1 ldmatrix address
        let a_addr = ptx.regs.alloc_b32();

        // Pre-allocate cp.async destination registers
        let mut a_dst_regs: Vec<Reg> = Vec::with_capacity(cp_chunks_a as usize);
        for _ in 0..cp_chunks_a {
            a_dst_regs.push(ptx.regs.alloc_b32());
        }
        let mut b0_dst_regs: Vec<Reg> = Vec::with_capacity(cp_chunks_b as usize);
        for _ in 0..cp_chunks_b {
            b0_dst_regs.push(ptx.regs.alloc_b32());
        }
        let mut b1_dst_regs: Vec<Reg> = Vec::with_capacity(cp_chunks_b as usize);
        for _ in 0..cp_chunks_b {
            b1_dst_regs.push(ptx.regs.alloc_b32());
        }

        let cp_size = ptx.regs.alloc_b32();
        let p_load = ptx.regs.alloc_pred();
        let next_read = ptx.regs.alloc_b32();
        let p_wrap_r = ptx.regs.alloc_pred();
        let next_write = ptx.regs.alloc_b32();
        let p_wrap_w = ptx.regs.alloc_pred();
        let p_loop = ptx.regs.alloc_pred();

        // Pre-allocate A fragment registers (shared between GEMM0 and GEMM1)
        let mut a_frag = [
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
        ];

        // Pre-allocate B0 fragment registers (all rn live simultaneously)
        let mut b0_frags: Vec<Vec<Reg>> = Vec::new();
        for _rn in 0..reg_n as usize {
            let mut frag_regs = Vec::new();
            for _grp in 0..b_ld_groups as usize {
                for _ in 0..4 {
                    frag_regs.push(ptx.regs.alloc_b32());
                }
            }
            b0_frags.push(frag_regs);
        }

        // Pre-allocate B1 fragment registers (all rn live simultaneously)
        let mut b1_frags: Vec<Vec<Reg>> = Vec::new();
        for _rn in 0..reg_n as usize {
            let mut frag_regs = Vec::new();
            for _grp in 0..b_ld_groups as usize {
                for _ in 0..4 {
                    frag_regs.push(ptx.regs.alloc_b32());
                }
            }
            b1_frags.push(frag_regs);
        }

        // Pre-allocate A fragment storage for reuse in GEMM1
        // After loading A for each (ki, rm), we store the fragments so GEMM1 can reuse them.
        // a_saved[ki][rm] = [Reg; 4]
        let mut a_saved: Vec<Vec<[Reg; 4]>> = Vec::new();
        for _ki in 0..k_warp_iters as usize {
            let mut rm_frags = Vec::new();
            for _rm in 0..reg_m as usize {
                rm_frags.push([
                    ptx.regs.alloc_b32(),
                    ptx.regs.alloc_b32(),
                    ptx.regs.alloc_b32(),
                    ptx.regs.alloc_b32(),
                ]);
            }
            a_saved.push(rm_frags);
        }

        // ═══════════════════════════════════════════════════════════════
        // K-LOOP
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_DUAL_KLOOP");

        // 1. Advance read stage BEFORE wait
        {
            ptx.add_s32_imm(next_read, read_stage, 1);
            ptx.setp_gt_s32_imm(p_wrap_r, next_read, stages as i32 - 1);
            ptx.selp_b32_imm_reg(read_stage, 0, next_read, p_wrap_r);
        }

        // 2. Wait + barrier
        ptx.cp_async_wait_group(wait_count);
        ptx.bar_sync(0);

        // 3. Compute read buffer base for A region
        ptx.shl_b32(read_off, read_stage, a_tile_bytes.trailing_zeros());
        ptx.add_s32(buf_base, smem_base, read_off);

        // 4. Load gamma BEFORE B (transform k-setup)
        for ki in 0..k_warp_iters as usize {
            transform.emit_k_setup(ptx, ki as u32);
        }
        ptx.blank();

        // ── ldmatrix.trans B0 ──
        ptx.comment(&format!(
            "ldmatrix.trans B0 -- {} loads x {} groups",
            reg_n, b_ld_groups
        ));
        for rn in 0..reg_n as usize {
            for grp in 0..b_ld_groups as usize {
                ptx.add_s32(b0_addr, buf_base, b_off[rn]);
                let grp_off = b0_start + (grp as i32) * (32 * c.bn as i32 * 2);
                let frag_base = grp * 4;
                let frag = [
                    b0_frags[rn][frag_base],
                    b0_frags[rn][frag_base + 1],
                    b0_frags[rn][frag_base + 2],
                    b0_frags[rn][frag_base + 3],
                ];
                ptx.ldmatrix_x4_trans(frag, b0_addr, Some(grp_off));
            }
        }
        ptx.blank();

        // ── ldmatrix.trans B1 ──
        ptx.comment(&format!(
            "ldmatrix.trans B1 -- {} loads x {} groups",
            reg_n, b_ld_groups
        ));
        for rn in 0..reg_n as usize {
            for grp in 0..b_ld_groups as usize {
                ptx.add_s32(b1_addr, buf_base, b_off[rn]);
                let grp_off = b1_start + (grp as i32) * (32 * c.bn as i32 * 2);
                let frag_base = grp * 4;
                let frag = [
                    b1_frags[rn][frag_base],
                    b1_frags[rn][frag_base + 1],
                    b1_frags[rn][frag_base + 2],
                    b1_frags[rn][frag_base + 3],
                ];
                ptx.ldmatrix_x4_trans(frag, b1_addr, Some(grp_off));
            }
        }
        ptx.blank();

        // ── Phase 2: Load A fragments, transform, execute ALL GEMM0 MMAs, then ALL GEMM1 MMAs ──
        // First: load and transform A fragments for all ki/rm, save them
        for ki in 0..k_warp_iters as usize {
            ptx.comment(&format!("ki={ki}: ldmatrix A + transform"));
            ptx.add_s32(a_addr, buf_base, a_off[ki]);

            for rm in 0..reg_m as usize {
                let a_rm_off = if rm == 0 {
                    None
                } else {
                    Some((rm as i32) * 2048_i32)
                };
                ptx.ldmatrix_x4(a_frag, a_addr, a_rm_off);
                transform.emit_transform(ptx, &mut a_frag, ki as u32, rm as u32);

                // Save A fragments for reuse in GEMM1
                for j in 0..4 {
                    ptx.mov_b32(a_saved[ki][rm][j], a_frag[j]);
                }

                // GEMM0 MMA: use saved A with B0 for this (ki, rm)
                for rn in 0..reg_n as usize {
                    let ai = rm * reg_n as usize + rn;
                    let mut acc_tile = acc0.regs[ai];
                    let b_sel = [b0_frags[rn][ki * 2], b0_frags[rn][ki * 2 + 1]];
                    mma.emit_mma(ptx, a_frag, b_sel, &mut acc_tile);
                }
            }
            ptx.blank();
        }

        // GEMM1 MMAs: reuse saved A fragments with B1
        ptx.comment("GEMM1 MMAs using saved A fragments");
        for ki in 0..k_warp_iters as usize {
            for rm in 0..reg_m as usize {
                let saved = a_saved[ki][rm];
                for rn in 0..reg_n as usize {
                    let ai = rm * reg_n as usize + rn;
                    let mut acc_tile = acc1.regs[ai];
                    let b_sel = [b1_frags[rn][ki * 2], b1_frags[rn][ki * 2 + 1]];
                    mma.emit_mma(ptx, saved, b_sel, &mut acc_tile);
                }
            }
        }
        ptx.blank();

        // ── Predicated loads for next tile ──
        ptx.setp_lt_s32(p_load, k_counter, k_minus_stages_bk);

        ptx.comment("Predicated loads for next tile (A + B0 + B1)");
        ptx.selp_b32(cp_size, 16, 0, p_load);

        // Compute write buffer base
        ptx.shl_b32(write_off, write_stage, a_tile_bytes.trailing_zeros());
        ptx.add_s32(write_buf_base, smem_base, write_off);

        ptx.bar_sync(0);

        // A next tile
        ptx.add_s32(a_dst_regs[0], write_buf_base, a_cp_off);
        copy_a.emit_async_copy(ptx, a_dst_regs[0], ga_loop[0], cp_size);
        for i in 1..cp_chunks_a as usize {
            ptx.add_s32_imm(a_dst_regs[i], a_dst_regs[0], (i * 2048) as i32);
            copy_a.emit_async_copy(ptx, a_dst_regs[i], ga_loop[i], cp_size);
        }
        copy_a.emit_commit(ptx);

        // B0 next tile
        ptx.add_s32(b0_dst_regs[0], write_buf_base, b0_cp_off);
        ptx.add_s32_imm(b0_dst_regs[0], b0_dst_regs[0], b0_start);
        copy_b0.emit_async_copy(ptx, b0_dst_regs[0], gb0_loop[0], cp_size);
        for i in 1..cp_chunks_b as usize {
            ptx.add_s32_imm(b0_dst_regs[i], b0_dst_regs[0], i as i32 * 2048);
            copy_b0.emit_async_copy(ptx, b0_dst_regs[i], gb0_loop[i], cp_size);
        }
        copy_b0.emit_commit(ptx);

        // B1 next tile
        ptx.add_s32(b1_dst_regs[0], write_buf_base, b1_cp_off);
        ptx.add_s32_imm(b1_dst_regs[0], b1_dst_regs[0], b1_start);
        copy_b1.emit_async_copy(ptx, b1_dst_regs[0], gb1_loop[0], cp_size);
        for i in 1..cp_chunks_b as usize {
            ptx.add_s32_imm(b1_dst_regs[i], b1_dst_regs[0], i as i32 * 2048);
            copy_b1.emit_async_copy(ptx, b1_dst_regs[i], gb1_loop[i], cp_size);
        }
        copy_b1.emit_commit(ptx);
        ptx.blank();

        // ── Advance circular buffer stage indices ──
        ptx.comment("Advance write buffer stage index");
        ptx.add_s32_imm(next_write, write_stage, 1);
        ptx.setp_gt_s32_imm(p_wrap_w, next_write, stages as i32 - 1);
        ptx.selp_b32_imm_reg(write_stage, 0, next_write, p_wrap_w);

        // ── Advance loop state ──
        ptx.comment("Advance loop state");
        ptx.add_s32_imm(k_counter, k_counter, c.bk as i32);
        for g in &ga_loop {
            ptx.add_s64_imm(*g, *g, (c.bk * 2) as i64);
        }
        for g in &gb0_loop {
            ptx.add_s64(*g, *g, b_stride_bytes);
        }
        for g in &gb1_loop {
            ptx.add_s64(*g, *g, b_stride_bytes);
        }

        // Extra advance
        if let Some(adv) = extra_k_advance {
            adv(ptx);
        }

        ptx.setp_lt_s32(p_loop, k_counter, k_param);
        ptx.w(&format!("@{p_loop} bra \t$L_DUAL_KLOOP;"));
        ptx.blank();

        ptx.bra_uni("$L_DUAL_EPILOGUE");
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // K=0 fallthrough
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_DUAL_K0_FALLTHROUGH");
        let zero2 = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(zero2, 0x00000000);
        for tile in &acc0.regs {
            for &r in tile {
                ptx.mov_b32(r, zero2);
            }
        }
        for tile in &acc1.regs {
            for &r in tile {
                ptx.mov_b32(r, zero2);
            }
        }
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Epilogue
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_DUAL_EPILOGUE");
        ptx.cp_async_wait_group(0);
        ptx.bar_sync(0);
        ptx.blank();

        DualPipelineResult {
            acc0,
            acc1,
            k_counter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{CpAsyncCopy, IdentityTransform, Mma16816};
    use crate::config::GemmConfig;
    use crate::gemm::{
        emit_a_global_addrs_cfg, emit_b_global_addrs_cfg, emit_cpasync_swizzle_cfg,
        emit_gemm_setup, gemm_params,
    };

    /// Helper: build a dual pipeline GEMM with the given config and return the full PTX string.
    fn build_dual_pipeline_ptx(config: &GemmConfig) -> String {
        let mut ptx = PtxBuilder::new(config.clone());
        let c = &ptx.config.clone();

        ptx.comment("Dual GEMM kernel generated by DualMainloopPipeline");
        ptx.blank();

        // Load parameters
        let a_ptr = ptx.regs.alloc_b64();
        let b0_ptr = ptx.regs.alloc_b64();
        let b1_ptr = ptx.regs.alloc_b64();
        let c_ptr = ptx.regs.alloc_b64();
        ptx.ld_param_b64(a_ptr, "param_A");
        ptx.ld_param_b64(b0_ptr, "param_B0");
        ptx.ld_param_b64(b1_ptr, "param_B1");
        ptx.ld_param_b64(c_ptr, "param_C");
        let n_param = ptx.regs.alloc_b32();
        let k_param = ptx.regs.alloc_b32();
        ptx.ld_param_b32(n_param, "param_N");
        ptx.ld_param_b32(k_param, "param_K");
        ptx.blank();

        // Thread/block setup
        let setup = emit_gemm_setup(&mut ptx, c);

        // cp.async swizzle addresses
        let (a_cp_off, b0_cp_off) = emit_cpasync_swizzle_cfg(&mut ptx, c, setup.tid);
        // B1 uses the same swizzle pattern as B0
        let b1_cp_off = ptx.regs.alloc_b32();
        ptx.mov_b32(b1_cp_off, b0_cp_off);

        // Global addresses
        let ga_chunks =
            emit_a_global_addrs_cfg(&mut ptx, c, setup.block_row, k_param, a_ptr, setup.tid);
        let gb0_chunks =
            emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, b0_ptr, setup.tid);
        let gb1_chunks =
            emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, b1_ptr, setup.tid);

        // Pipeline
        let pipeline = DualMainloopPipeline::new(c.num_stages);
        let _result = pipeline.emit(
            &mut ptx,
            c,
            &setup,
            &CpAsyncCopy,
            &CpAsyncCopy,
            &CpAsyncCopy,
            &IdentityTransform,
            &Mma16816,
            ga_chunks,
            a_cp_off,
            gb0_chunks,
            b0_cp_off,
            gb1_chunks,
            b1_cp_off,
            n_param,
            k_param,
            None,
        );

        ptx.blank();
        ptx.ret();

        ptx.finalize("dual_gemm_test", &gemm_params())
    }

    #[test]
    fn test_dual_pipeline_produces_valid_ptx() {
        let config = GemmConfig::default_64x64();
        let ptx = build_dual_pipeline_ptx(&config);
        // Should not panic and should produce non-empty PTX
        assert!(!ptx.is_empty(), "Dual pipeline must produce non-empty PTX");
        assert!(
            ptx.contains("$L_DUAL_KLOOP:"),
            "Must have dual K-loop label"
        );
        assert!(
            ptx.contains("$L_DUAL_EPILOGUE:"),
            "Must have dual epilogue label"
        );
    }

    #[test]
    fn test_dual_pipeline_mma_count_is_2x_single() {
        let config = GemmConfig::default_64x64();
        let ptx = build_dual_pipeline_ptx(&config);

        let kloop_start = ptx.find("$L_DUAL_KLOOP:").expect("K-loop label");
        let kloop_end = ptx
            .find("$L_DUAL_K0_FALLTHROUGH:")
            .expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // 64x64: single pipeline has 16 MMAs. Dual should have 32 (16 GEMM0 + 16 GEMM1).
        let mma_count = kloop.matches("mma.sync.aligned.m16n8k16").count();
        let single_mma_count = 16; // REG_M=4, REG_N=2, k_iters=2 => 4*2*2 = 16
        assert_eq!(
            mma_count,
            single_mma_count * 2,
            "Dual pipeline must have 2x MMAs ({}), got {}",
            single_mma_count * 2,
            mma_count
        );
    }

    #[test]
    fn test_dual_pipeline_cp_async_count_a_b0_b1() {
        let config = GemmConfig::default_64x64();
        let ptx = build_dual_pipeline_ptx(&config);

        let kloop_start = ptx.find("$L_DUAL_KLOOP:").expect("K-loop label");
        let kloop_end = ptx
            .find("$L_DUAL_K0_FALLTHROUGH:")
            .expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // 64x64: 2 chunks for A + 2 chunks for B0 + 2 chunks for B1 = 6 cp.async per K-iter
        let cp_count = kloop.matches("cp.async.cg.shared.global").count();
        assert_eq!(
            cp_count, 6,
            "Dual 64x64 pipeline must have 6 cp.async per K-iteration (2 A + 2 B0 + 2 B1), got {}",
            cp_count
        );
    }

    #[test]
    fn test_dual_pipeline_two_accumulators_correct_dimensions() {
        let config = GemmConfig::default_64x64();
        let mut ptx = PtxBuilder::new(config.clone());
        let c = &ptx.config.clone();

        let a_ptr = ptx.regs.alloc_b64();
        let b0_ptr = ptx.regs.alloc_b64();
        let b1_ptr = ptx.regs.alloc_b64();
        ptx.ld_param_b64(a_ptr, "param_A");
        ptx.ld_param_b64(b0_ptr, "param_B0");
        ptx.ld_param_b64(b1_ptr, "param_B1");
        let n_param = ptx.regs.alloc_b32();
        let k_param = ptx.regs.alloc_b32();
        ptx.ld_param_b32(n_param, "param_N");
        ptx.ld_param_b32(k_param, "param_K");

        let setup = emit_gemm_setup(&mut ptx, c);
        let (a_cp_off, b0_cp_off) = emit_cpasync_swizzle_cfg(&mut ptx, c, setup.tid);
        let b1_cp_off = ptx.regs.alloc_b32();
        ptx.mov_b32(b1_cp_off, b0_cp_off);

        let ga_chunks =
            emit_a_global_addrs_cfg(&mut ptx, c, setup.block_row, k_param, a_ptr, setup.tid);
        let gb0_chunks =
            emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, b0_ptr, setup.tid);
        let gb1_chunks =
            emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, b1_ptr, setup.tid);

        let pipeline = DualMainloopPipeline::new(c.num_stages);
        let result = pipeline.emit(
            &mut ptx,
            c,
            &setup,
            &CpAsyncCopy,
            &CpAsyncCopy,
            &CpAsyncCopy,
            &IdentityTransform,
            &Mma16816,
            ga_chunks,
            a_cp_off,
            gb0_chunks,
            b0_cp_off,
            gb1_chunks,
            b1_cp_off,
            n_param,
            k_param,
            None,
        );

        // Both accumulators should have the same dimensions
        assert_eq!(result.acc0.reg_m, c.reg_m());
        assert_eq!(result.acc0.reg_n, c.reg_n());
        assert_eq!(result.acc1.reg_m, c.reg_m());
        assert_eq!(result.acc1.reg_n, c.reg_n());

        // Both should have reg_m * reg_n tiles
        let expected_tiles = (c.reg_m() * c.reg_n()) as usize;
        assert_eq!(result.acc0.regs.len(), expected_tiles);
        assert_eq!(result.acc1.regs.len(), expected_tiles);
    }

    #[test]
    fn test_dual_pipeline_ldmatrix_count_a_b0_b1() {
        let config = GemmConfig::default_64x64();
        let ptx = build_dual_pipeline_ptx(&config);

        let kloop_start = ptx.find("$L_DUAL_KLOOP:").expect("K-loop label");
        let kloop_end = ptx
            .find("$L_DUAL_K0_FALLTHROUGH:")
            .expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // For 64x64 (BK=32):
        //   A ldmatrix (non-transposed): REG_M=4 * k_warp_iters=2 = 8
        //   B0 ldmatrix.trans: REG_N=2 * b_ld_groups=1 = 2
        //   B1 ldmatrix.trans: REG_N=2 * b_ld_groups=1 = 2
        //   Total ldmatrix: 8 + 2 + 2 = 12
        let ldm_count = kloop
            .matches("ldmatrix.sync.aligned.m8n8.x4")
            .count();

        // A: non-transposed ldmatrix
        let ldm_a = kloop
            .matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
            .count();
        // B: transposed ldmatrix
        let ldm_b = kloop
            .matches("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")
            .count();

        let expected_a = (config.reg_m() * (config.bk / config.mma_k)) as usize; // 4*2=8
        let expected_b = 2 * (config.reg_n() * (config.bk / config.mma_k / 2)) as usize; // 2*(2*1)=4

        assert_eq!(
            ldm_a, expected_a,
            "Expected {} A ldmatrix calls, got {}",
            expected_a, ldm_a
        );
        assert_eq!(
            ldm_b, expected_b,
            "Expected {} B ldmatrix.trans calls (B0+B1), got {}",
            expected_b, ldm_b
        );
        assert_eq!(
            ldm_count,
            expected_a + expected_b,
            "Total ldmatrix must be {} (A={} + B0+B1={}), got {}",
            expected_a + expected_b,
            expected_a,
            expected_b,
            ldm_count
        );
    }

    #[test]
    fn test_dual_pipeline_has_prologue_and_epilogue() {
        let config = GemmConfig::default_64x64();
        let ptx = build_dual_pipeline_ptx(&config);

        assert!(
            ptx.contains("Dual pipeline prologue: load tile 0"),
            "Must have prologue tile 0"
        );
        assert!(
            ptx.contains("$L_DUAL_EPILOGUE:"),
            "Must have epilogue label"
        );
        assert!(
            ptx.contains("$L_DUAL_KLOOP:"),
            "Must have K-loop label"
        );
        assert!(
            ptx.contains("$L_DUAL_K0_FALLTHROUGH:"),
            "Must have K0 fallthrough"
        );
    }

    #[test]
    fn test_dual_pipeline_new_requires_at_least_2_stages() {
        let _p = DualMainloopPipeline::new(2);
        let _p = DualMainloopPipeline::new(3);
        let _p = DualMainloopPipeline::new(4);
    }

    #[test]
    #[should_panic(expected = "Pipeline requires at least 2 stages")]
    fn test_dual_pipeline_new_panics_on_1_stage() {
        let _p = DualMainloopPipeline::new(1);
    }

    #[test]
    fn test_dual_pipeline_128x128_mma_count() {
        let config = GemmConfig::default_128x128();
        let ptx = build_dual_pipeline_ptx(&config);

        let kloop_start = ptx.find("$L_DUAL_KLOOP:").expect("K-loop label");
        let kloop_end = ptx
            .find("$L_DUAL_K0_FALLTHROUGH:")
            .expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // 128x128: single pipeline has 64 MMAs. Dual should have 128.
        let mma_count = kloop.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 128,
            "128x128 dual pipeline must have 128 MMA instructions (64*2), got {}",
            mma_count
        );
    }

    #[test]
    fn test_dual_pipeline_3_stage_wait_group() {
        let config = GemmConfig {
            num_stages: 3,
            ..GemmConfig::default_64x64()
        };
        let ptx = build_dual_pipeline_ptx(&config);

        // With 3 stages and 3 async groups per tile (A=1, B0=1, B1=1):
        // wait_count = 3 * (3 - 2) = 3
        assert!(
            ptx.contains("cp.async.wait_group \t3"),
            "3-stage dual pipeline must wait for group 3 in K-loop"
        );
    }

    #[test]
    fn test_dual_pipeline_gemm0_before_gemm1() {
        // Verify that GEMM0 MMAs come before GEMM1 MMAs in the K-loop
        let config = GemmConfig::default_64x64();
        let ptx = build_dual_pipeline_ptx(&config);

        let kloop_start = ptx.find("$L_DUAL_KLOOP:").expect("K-loop label");
        let kloop_end = ptx
            .find("$L_DUAL_K0_FALLTHROUGH:")
            .expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // The "GEMM1 MMAs using saved A fragments" comment should appear AFTER ldmatrix A
        let gemm1_comment = kloop
            .find("GEMM1 MMAs using saved A fragments")
            .expect("GEMM1 comment");
        let first_ldmatrix_a = kloop
            .find("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
            .expect("first ldmatrix A");

        assert!(
            gemm1_comment > first_ldmatrix_a,
            "GEMM1 MMAs must come after A fragment loads (GEMM0 MMAs)"
        );
    }
}
