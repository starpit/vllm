use crate::{PtxBuilder, Reg};
use crate::config::GemmConfig;
use crate::atoms::{CopyAtom, TransformAtom, MmaAtom};
use crate::gemm::{GemmSetup, AccumulatorMap};

// REG_M and REG_N are now derived from the GemmConfig:
//   REG_M = config.reg_m() = wm / mma_m
//   REG_N = config.reg_n() = wn / mma_n
// For 64×64:  REG_M=4, REG_N=2 (wm=64,wn=16)
// For 128×128: REG_M=4, REG_N=8 (wm=64,wn=64)

// ═══════════════════════════════════════════════════════════════════════════
// MainloopPipeline — THE single K-loop implementation
//
// All GEMM-based kernels use this. The pipeline handles:
// - Software pipelining with configurable stages (2 or 3)
// - Prologue tile loads
// - K-loop with interleaved: ldmatrix, transform, MMA, async copy
// - Epilogue drain
//
// Supports BK=32 (k_warp_iters=2) and BK=64 (k_warp_iters=4).
// For BK=64, the pipeline automatically:
// - Issues 4 cp.async chunks per tile instead of 2
// - Issues 2 ldmatrix.x4.trans calls per rn (rows 0-31 and 32-63)
// - Unrolls 4 ki iterations instead of 2
//
// The atoms don't know about double-buffering, pipeline stages, or sync.
// The pipeline does ALL scheduling.
//
// Compiler optimizations applied:
// 1. Register reuse: K-loop temporaries (buf_base, b_addr, a_addr, cp dst
//    regs) are allocated ONCE before the loop and reused each iteration.
// 2. CSE: cp.async smem destinations use pre-swizzled offsets (a_cp_off,
//    b_cp_off) added to write_buf_base only once, not per-chunk.
// 3. LICM: ldmatrix swizzle offsets, B column group strides, and all
//    constant address components are computed before the loop.
// 4. Constant folding: b_start, chunk offsets, and rm offsets are folded
//    into immediate operands of add/ldmatrix instructions.
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
        assert!(stages >= 2, "Pipeline requires at least 2 stages");
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
    /// - `ga_chunks`: global A pointers, one per cp.async chunk (e.g., 2 for BM=64/BK=32, 4 for BM=128/BK=32)
    /// - `gb_chunks`: global B pointers, one per cp.async chunk
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
        ga_chunks: Vec<Reg>,
        a_cp_off: Reg,
        gb_chunks: Vec<Reg>,
        b_cp_off: Reg,
        // Params
        n_param: Reg,
        k_param: Reg,
        // Optional extra advance
        extra_advance: Option<&dyn Fn(&mut PtxBuilder)>,
    ) -> PipelineResult {
        let smem_base = setup.smem_base;
        let tid = setup.tid;
        let stages = self.stages;

        // kWarpGemmIterations = BK / MMA_K
        let k_warp_iters = c.bk / c.mma_k;  // e.g. 32/16=2 or 64/16=4
        assert!(k_warp_iters >= 2, "kWarpGemmIterations must be >= 2");

        let a_tile_bytes = c.smem_a_bytes();     // BM*BK*2 (4096 for BK=32, 8192 for BK=64)
        let b_tile_bytes = c.smem_b_bytes();     // BK*BN*2 (4096 for BK=32, 8192 for BK=64)
        let b_start = (a_tile_bytes * stages) as i32;  // B region starts after all A stages
        // The buffer indexing assumes a_tile_bytes == b_tile_bytes (BM == BN).
        // This is true for all our configs (BM=BN=64) but we assert it.
        assert_eq!(a_tile_bytes, b_tile_bytes,
            "Pipeline assumes a_tile_bytes == b_tile_bytes (BM*BK == BK*BN => BM == BN)");

        // Number of 2048-byte cp.async chunks per tile.
        // Each cp.async = 128 threads * 16 bytes = 2048 bytes.
        let cp_chunks_a = a_tile_bytes / 2048;   // 2 for BK=32, 4 for BK=64
        let cp_chunks_b = b_tile_bytes / 2048;   // 2 for BK=32, 4 for BK=64

        // Number of ldmatrix.x4.trans groups for B per tile.
        // Each ldmatrix.x4.trans covers 32 K-rows (giving 4 regs = 2 ki values).
        let b_ld_groups = k_warp_iters / 2;  // 1 for BK=32, 2 for BK=64

        let tid_x16 = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_x16, tid, 4);

        assert_eq!(ga_chunks.len(), cp_chunks_a as usize,
            "Expected {} A global pointers, got {}", cp_chunks_a, ga_chunks.len());
        assert_eq!(gb_chunks.len(), cp_chunks_b as usize,
            "Expected {} B global pointers, got {}", cp_chunks_b, gb_chunks.len());

        // ═══════════════════════════════════════════════════════════════
        // A smem base addresses for cp.async (buffer 0, all chunks)
        // Each chunk is 2048 bytes apart in smem.
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

        // ga_all: use the caller-provided global pointers directly
        let ga_all = ga_chunks;

        // B smem base addresses for cp.async (buffer 0, all chunks)
        let b_cp_base_smem = ptx.regs.alloc_b32();
        ptx.add_s32(b_cp_base_smem, smem_base, b_cp_off);
        let mut b_cp: Vec<Reg> = Vec::with_capacity(cp_chunks_b as usize);
        let b_cp0 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(b_cp0, b_cp_base_smem, b_start);
        b_cp.push(b_cp0);
        for i in 1..cp_chunks_b {
            let r = ptx.regs.alloc_b32();
            ptx.add_s32_imm(r, b_cp0, (i * 2048) as i32);
            b_cp.push(r);
        }

        // gb_all: use the caller-provided global pointers directly
        let gb_all = gb_chunks;
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
        //
        // OPTIMIZATION: Instead of creating separate ga_cur/gb_cur copies
        // and then ga_loop/gb_loop copies, we allocate the loop pointers
        // first and use them directly for prologue loads. This eliminates
        // cp_chunks_a + cp_chunks_b extra b64 register allocations.
        // ═══════════════════════════════════════════════════════════════

        // Allocate the loop global pointers (these will be advanced through
        // prologue and then used in the K-loop)
        let mut ga_loop: Vec<Reg> = Vec::new();
        for &g in &ga_all {
            let r = ptx.regs.alloc_b64();
            ptx.mov_b64(r, g);
            ga_loop.push(r);
        }
        let mut gb_loop: Vec<Reg> = Vec::new();
        for &g in &gb_all {
            let r = ptx.regs.alloc_b64();
            ptx.mov_b64(r, g);
            gb_loop.push(r);
        }

        for stage in 0..stages - 1 {
            ptx.comment(&format!("Pipeline prologue: load tile {stage} into buffer {stage}"));

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

            // B tile into buffer `stage`
            for i in 0..cp_chunks_b as usize {
                let b_dst = ptx.regs.alloc_b32();
                ptx.add_s32_imm(b_dst, b_cp[i], (stage * b_tile_bytes) as i32);
                copy_b.emit_async_copy(ptx, b_dst, gb_loop[i], sz);
            }
            copy_b.emit_commit(ptx);
            ptx.blank();

            // Advance global pointers to next tile
            ptx.comment(&format!("Advance global pointers for tile {}", stage + 1));
            for g in &ga_loop {
                ptx.add_s64_imm(*g, *g, (c.bk * 2) as i64);
            }
            for g in &gb_loop {
                ptx.add_s64(*g, *g, b_stride_bytes);
            }
            // Advance extra state
            if let Some(adv) = extra_advance {
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

        // A ldmatrix offsets: same for 64×64 and 128×128
        // The formula only depends on BK (controls ki XOR) and BM (controls rm chunk stride).
        // For BM=128, we have 4 rm groups at +0, +2048, +4096, +6144 within each ki chunk-pair.
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
        // Build a_off for all ki values.
        // Each 2048-byte chunk covers 32 K-columns (2 ki values).
        // Within a chunk: ki%2 == 0 uses base, ki%2 == 1 uses base XOR 32.
        // Across chunks: ki/2 * 4096 jumps to the next chunk-pair.
        // (chunk-pair = two 2048-byte chunks for rm=0 and rm=1)
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

        // B ldmatrix offsets: depend on BN.
        // b_ld_row shift = log2(BN*2), mask = 31 * BN * 2
        // For BN=64:  shift=7, mask=3968
        // For BN=128: shift=8, mask=7936
        let lane = setup.lane;
        let b_row_shift = (c.bn * 2).trailing_zeros();
        let b_row_mask = 31 * c.bn * 2;
        let tid_shifted_b = ptx.regs.alloc_b32();
        ptx.shl_b32(tid_shifted_b, tid, b_row_shift);
        let b_ld_row = ptx.regs.alloc_b32();
        ptx.and_b32(b_ld_row, tid_shifted_b, b_row_mask);

        // b_ld_col: same formula for all configs
        // For BN=64:  (lane & 7) << 4   (using lane)
        // For BN=128: (tid << 4) & 112   (equivalent to (tid & 7) << 4 = (lane & 7) << 4)
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

        // Build b_off array for all rn values.
        // Within each column group of 4, use XOR 0,32,64,96.
        // Across column groups, add b_col_group_stride (128 bytes).
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
        ptx.w(&format!("@{p_has_k} bra \t$L_LOOP_ENTRY;"));
        ptx.bra_uni("$L_K0_FALLTHROUGH");
        ptx.blank();

        ptx.label("$L_LOOP_ENTRY");

        // k_minus_stages_bk: used for predicated loads
        let k_minus_stages_bk = ptx.regs.alloc_b32();
        ptx.add_s32_imm(k_minus_stages_bk, k_param, -((stages * c.bk) as i32));
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Transform prologue (e.g., load norm factors into registers)
        // ═══════════════════════════════════════════════════════════════
        transform_a.emit_prologue(ptx);
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Initialize accumulators
        // ═══════════════════════════════════════════════════════════════
        ptx.comment(&format!("Initialize accumulators to 0.0f (REG_M={}, REG_N={}, {} tiles)", reg_m, reg_n, reg_m * reg_n));
        let zero = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(zero, 0x00000000);

        let num_tiles = (reg_m * reg_n) as usize;
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
            reg_m,
            reg_n,
        };
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Circular buffer state
        // ═══════════════════════════════════════════════════════════════
        let read_stage = ptx.regs.alloc_b32();
        let write_stage = ptx.regs.alloc_b32();
        // Start: read from stage 0, write to stage (stages-1)
        // The prologue loaded stages 0..stages-2. The first loop iteration
        // reads stage 0 and writes the next tile into stage (stages-1).
        ptx.mov_b32_imm(read_stage, 0);
        ptx.mov_b32_imm(write_stage, stages - 1);

        let k_counter = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(k_counter, 0);
        ptx.blank();

        // Compute total async groups in flight
        let a_async = copy_a.async_groups_per_tile();
        let b_async = copy_b.async_groups_per_tile();
        let total_async_per_tile = a_async + b_async;
        // Wait for oldest stage: keep (stages-2) groups in flight
        let wait_count = total_async_per_tile * (stages - 2);

        // ═══════════════════════════════════════════════════════════════
        // Pre-allocate K-loop temporary registers (OPTIMIZATION: register reuse)
        //
        // Instead of allocating new registers inside the K-loop body
        // (which inflates the register declaration and prevents reuse),
        // we allocate them once here and reuse them every iteration.
        // This matches the hand-written PTX pattern where a fixed set of
        // registers is reused across iterations.
        // ═══════════════════════════════════════════════════════════════
        let buf_base = ptx.regs.alloc_b32();       // read buffer base (smem)
        let read_off = ptx.regs.alloc_b32();        // read_stage * tile_bytes
        let write_buf_base = ptx.regs.alloc_b32();  // write buffer base (smem)
        let write_off = ptx.regs.alloc_b32();       // write_stage * tile_bytes
        let b_addr = ptx.regs.alloc_b32();          // reused for each B ldmatrix address
        let a_addr = ptx.regs.alloc_b32();          // reused for each A ldmatrix address

        // Pre-allocate cp.async destination registers (one per chunk, reused each iter)
        let mut a_dst_regs: Vec<Reg> = Vec::with_capacity(cp_chunks_a as usize);
        for _ in 0..cp_chunks_a {
            a_dst_regs.push(ptx.regs.alloc_b32());
        }
        let mut b_dst_regs: Vec<Reg> = Vec::with_capacity(cp_chunks_b as usize);
        for _ in 0..cp_chunks_b {
            b_dst_regs.push(ptx.regs.alloc_b32());
        }

        // Pre-allocate cp_size and predicates for the loop
        let cp_size = ptx.regs.alloc_b32();
        let p_load = ptx.regs.alloc_pred();
        let next_read = ptx.regs.alloc_b32();
        let p_wrap_r = ptx.regs.alloc_pred();
        let next_write = ptx.regs.alloc_b32();
        let p_wrap_w = ptx.regs.alloc_pred();
        let p_loop = ptx.regs.alloc_pred();

        // Pre-allocate A fragment registers (4 regs per ldmatrix, reused each ki/rm)
        let mut a_frag = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                          ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];

        // Pre-allocate B fragment registers.
        // Each rn needs 4*b_ld_groups registers. These are all live simultaneously
        // during the MMA phase, so they cannot be reused across rn values.
        // However, they CAN be reused across K-loop iterations.
        let mut b_frags: Vec<Vec<Reg>> = Vec::new();
        for _rn in 0..reg_n as usize {
            let mut frag_regs = Vec::new();
            for _grp in 0..b_ld_groups as usize {
                for _ in 0..4 {
                    frag_regs.push(ptx.regs.alloc_b32());
                }
            }
            b_frags.push(frag_regs);
        }

        // ═══════════════════════════════════════════════════════════════
        // K-LOOP
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_KLOOP");

        // Wait for the oldest async group + barrier
        ptx.cp_async_wait_group(wait_count);
        ptx.bar_sync(0);

        // Compute read buffer base: smem_base + read_stage * a_tile_bytes
        // (OPTIMIZATION: reuses pre-allocated buf_base and read_off)
        ptx.shl_b32(read_off, read_stage, a_tile_bytes.trailing_zeros());
        ptx.add_s32(buf_base, smem_base, read_off);
        ptx.blank();

        // ── ldmatrix.trans B ──
        // For each rn, issue b_ld_groups ldmatrix.x4.trans calls.
        // Group 0: rows 0-31 (4 regs), Group 1: rows 32-63 (4 more regs).
        // b_frags[rn] has 4*b_ld_groups registers total.
        //
        // OPTIMIZATION: reuse single b_addr register for all rn values.
        // Each b_addr computation (buf_base + b_off[rn]) produces a value
        // that is consumed immediately by ldmatrix, then dead.
        ptx.comment(&format!("ldmatrix.trans B -- {} loads x {} groups", reg_n, b_ld_groups));
        for rn in 0..reg_n as usize {
            for grp in 0..b_ld_groups as usize {
                ptx.add_s32(b_addr, buf_base, b_off[rn]);
                // Add offset for additional K-row groups:
                // Group 0 is at b_start (rows 0-31).
                // Group 1 is at b_start + 32*BN*2 (rows 32-63).
                let grp_off = b_start + (grp as i32) * (32 * c.bn as i32 * 2);
                let frag_base = grp * 4;
                let frag = [b_frags[rn][frag_base], b_frags[rn][frag_base + 1],
                            b_frags[rn][frag_base + 2], b_frags[rn][frag_base + 3]];
                ptx.ldmatrix_x4_trans(frag, b_addr, Some(grp_off));
            }
        }
        ptx.blank();

        // ── Unrolled ki loop: for each ki, load A, transform, MMA ──
        // OPTIMIZATION: reuse a_addr and a_frag registers across ki/rm iterations.
        // a_frag values are consumed by MMA before the next ldmatrix overwrites them.
        for ki in 0..k_warp_iters as usize {
            ptx.comment(&format!("ki={ki}: ldmatrix A + transform + MMA"));
            transform_a.emit_k_setup(ptx, ki as u32);
            ptx.add_s32(a_addr, buf_base, a_off[ki]);

            // A fragments for each rm, immediately consumed by MMA
            for rm in 0..reg_m as usize {
                // Each rm occupies a 2048-byte chunk within the A tile.
                // rm=0 at +0, rm=1 at +2048, rm=2 at +4096, rm=3 at +6144.
                let a_rm_off = if rm == 0 { None } else { Some((rm as i32) * 2048_i32) };
                ptx.ldmatrix_x4(a_frag, a_addr, a_rm_off);
                transform_a.emit_transform(ptx, &mut a_frag, ki as u32, rm as u32);

                // MMA with all rn for this rm
                for rn in 0..reg_n as usize {
                    let ai = rm * reg_n as usize + rn;
                    let mut acc_tile = acc.regs[ai];
                    let b_sel = [b_frags[rn][ki * 2], b_frags[rn][ki * 2 + 1]];
                    mma.emit_mma(ptx, a_frag, b_sel, &mut acc_tile);
                }
            }
            ptx.blank();
        }

        // ── Predicated loads for next tile ──
        // OPTIMIZATION: reuse pre-allocated p_load, cp_size, write_buf_base,
        // a_dst_regs, b_dst_regs, next_read/write, p_wrap registers
        ptx.setp_lt_s32(p_load, k_counter, k_minus_stages_bk);

        ptx.comment("Predicated loads for next tile");
        ptx.selp_b32(cp_size, 16, 0, p_load);

        // Compute write buffer base: smem_base + write_stage * a_tile_bytes
        ptx.shl_b32(write_off, write_stage, a_tile_bytes.trailing_zeros());
        ptx.add_s32(write_buf_base, smem_base, write_off);

        ptx.bar_sync(0);

        // A next tile
        // OPTIMIZATION: compute write_buf_base + a_cp_off once into a_dst_regs[0],
        // then offset subsequent chunks from it.
        ptx.add_s32(a_dst_regs[0], write_buf_base, a_cp_off);
        copy_a.emit_async_copy(ptx, a_dst_regs[0], ga_loop[0], cp_size);
        for i in 1..cp_chunks_a as usize {
            ptx.add_s32_imm(a_dst_regs[i], a_dst_regs[0], (i * 2048) as i32);
            copy_a.emit_async_copy(ptx, a_dst_regs[i], ga_loop[i], cp_size);
        }
        copy_a.emit_commit(ptx);

        // B next tile
        // OPTIMIZATION: compute write_buf_base + b_cp_off + b_start once into b_dst_regs[0],
        // then offset subsequent chunks from it.
        ptx.add_s32(b_dst_regs[0], write_buf_base, b_cp_off);
        ptx.add_s32_imm(b_dst_regs[0], b_dst_regs[0], b_start);
        copy_b.emit_async_copy(ptx, b_dst_regs[0], gb_loop[0], cp_size);
        for i in 1..cp_chunks_b as usize {
            ptx.add_s32_imm(b_dst_regs[i], b_dst_regs[0], i as i32 * 2048);
            copy_b.emit_async_copy(ptx, b_dst_regs[i], gb_loop[i], cp_size);
        }
        copy_b.emit_commit(ptx);
        ptx.blank();

        // ── Advance circular buffer stage indices ──
        ptx.comment("Advance circular buffer stage indices");
        ptx.add_s32_imm(next_read, read_stage, 1);
        ptx.setp_gt_s32_imm(p_wrap_r, next_read, stages as i32 - 1);
        ptx.selp_b32_imm_reg(read_stage, 0, next_read, p_wrap_r);

        ptx.add_s32_imm(next_write, write_stage, 1);
        ptx.setp_gt_s32_imm(p_wrap_w, next_write, stages as i32 - 1);
        ptx.selp_b32_imm_reg(write_stage, 0, next_write, p_wrap_w);

        // ── Advance loop state ──
        ptx.comment("Advance loop state");
        ptx.add_s32_imm(k_counter, k_counter, c.bk as i32);
        for g in &ga_loop {
            ptx.add_s64_imm(*g, *g, (c.bk * 2) as i64);
        }
        for g in &gb_loop {
            ptx.add_s64(*g, *g, b_stride_bytes);
        }

        // Extra advance (e.g., k_offset for transform)
        if let Some(adv) = extra_advance {
            adv(ptx);
        }

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
