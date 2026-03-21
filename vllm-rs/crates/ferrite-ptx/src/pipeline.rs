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
    /// - `ga0, ga1`: A global pointers for chunk 0/1 (rows 0-31, 32-63 of first 32 K-cols)
    /// - `gb0, gb1`: B global pointers for chunk 0/1 (rows 0-15, 16-31)
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

        // ═══════════════════════════════════════════════════════════════
        // A smem base addresses for cp.async (buffer 0, all chunks)
        //
        // For BK=32: chunks at +0, +2048 (ga0 -> rows 0-31, ga1 -> rows 32-63)
        // For BK=64: chunks at +0, +2048, +4096, +6144
        //   ga0 -> chunk 0 (rows 0-31, K-cols 0-31)
        //   ga1 -> chunk 1 (rows 32-63, K-cols 0-31)
        //   ga2 -> chunk 2 (rows 0-31, K-cols 32-63) = ga0 + 64 bytes
        //   ga3 -> chunk 3 (rows 32-63, K-cols 32-63) = ga1 + 64 bytes
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

        // Build ga_all: all global A source pointers for cp.async chunks.
        // For BK=32: [ga0, ga1]
        // For BK=64: [ga0, ga1, ga0+64, ga1+64]
        let mut ga_all: Vec<Reg> = vec![ga0, ga1];
        for i in 2..cp_chunks_a {
            let r = ptx.regs.alloc_b64();
            // Extra chunks cover additional K-columns of the same rows.
            // Chunk i corresponds to the same row-half as chunk (i%2), but
            // offset by ((i/2) * 32) K-columns, where each f16 element = 2 bytes.
            // But global A is row-major with stride K (full), not BK.
            // Since each thread reads contiguous 16 bytes from its row position,
            // advancing by 32 K-cols means adding 32*2 = 64 bytes to the
            // thread's starting address.
            let base = if i % 2 == 0 { ga0 } else { ga1 };
            ptx.add_s64_imm(r, base, ((i / 2) * 64) as i64);
            ga_all.push(r);
        }

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

        // Build gb_all: all global B source pointers for cp.async chunks.
        // For BK=32: [gb0, gb1] (rows 0-15, 16-31)
        // For BK=64: [gb0, gb1, gb2, gb3] (rows 0-15, 16-31, 32-47, 48-63)
        //   gb2 = gb0 + 32*N*2 (advance 32 rows in B, each row is N elements)
        //   gb3 = gb1 + 32*N*2
        let mut gb_all: Vec<Reg> = vec![gb0, gb1];
        if cp_chunks_b > 2 {
            // Compute 32*N*2 = 64*N as byte offset for extra B rows
            let b_row32_stride = ptx.regs.alloc_b64();
            {
                let n_x64 = ptx.regs.alloc_b32();
                ptx.shl_b32(n_x64, n_param, 6); // N * 64
                let two_r = ptx.regs.alloc_b32();
                ptx.mov_b32_imm(two_r, 1);
                ptx.mul_wide_s32(b_row32_stride, n_x64, two_r);
            }
            for i in 2..cp_chunks_b {
                let r = ptx.regs.alloc_b64();
                let base = if i % 2 == 0 { gb0 } else { gb1 };
                // For chunk i, add (i/2)*32 rows worth of stride.
                // Since chunks 0,1 are rows 0-15 and 16-31 (first 32 rows),
                // chunks 2,3 are rows 32-47 and 48-63.
                // The offset is (i/2) * 32*N*2 but we already computed 32*N*2.
                // For i=2,3: (i/2)=1, so add 1 * b_row32_stride.
                if i / 2 == 1 {
                    ptx.add_s64(r, base, b_row32_stride);
                } else {
                    // For even larger BK, would need i/2 * stride. Not needed for BK=64.
                    panic!("BK > 64 not supported (would need multi-stride B offsets)");
                }
                gb_all.push(r);
            }
        }
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

        // Track current global pointers (will advance through prologue)
        let mut ga_cur: Vec<Reg> = Vec::new();
        for &g in &ga_all {
            let r = ptx.regs.alloc_b64();
            ptx.mov_b64(r, g);
            ga_cur.push(r);
        }
        let mut gb_cur: Vec<Reg> = Vec::new();
        for &g in &gb_all {
            let r = ptx.regs.alloc_b64();
            ptx.mov_b64(r, g);
            gb_cur.push(r);
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
                copy_a.emit_async_copy(ptx, a_dst, ga_cur[i], sz);
            }
            copy_a.emit_commit(ptx);

            // B tile into buffer `stage`
            for i in 0..cp_chunks_b as usize {
                let b_dst = ptx.regs.alloc_b32();
                ptx.add_s32_imm(b_dst, b_cp[i], (stage * b_tile_bytes) as i32);
                copy_b.emit_async_copy(ptx, b_dst, gb_cur[i], sz);
            }
            copy_b.emit_commit(ptx);
            ptx.blank();

            // Advance global pointers to next tile (if not last prologue stage)
            if stage < stages - 2 {
                ptx.comment(&format!("Advance global pointers for tile {}", stage + 1));
                for g in &ga_cur {
                    ptx.add_s64_imm(*g, *g, (c.bk * 2) as i64);
                }
                for g in &gb_cur {
                    ptx.add_s64(*g, *g, b_stride_bytes);
                }
                // Advance extra state
                if let Some(adv) = extra_advance {
                    adv(ptx);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // Advance global ptrs past prologue (for the loop's async copies)
        // ga_cur currently points at tile (stages-2). Advance once more.
        // ═══════════════════════════════════════════════════════════════
        ptx.comment(&format!("Advance global ptrs to tile {} position (for loop's loads)", stages - 1));
        let mut ga_loop: Vec<Reg> = Vec::new();
        for &g in &ga_cur {
            let r = ptx.regs.alloc_b64();
            ptx.add_s64_imm(r, g, (c.bk * 2) as i64);
            ga_loop.push(r);
        }
        let mut gb_loop: Vec<Reg> = Vec::new();
        for &g in &gb_cur {
            let r = ptx.regs.alloc_b64();
            ptx.add_s64(r, g, b_stride_bytes);
            gb_loop.push(r);
        }
        // Advance extra state for the last prologue-to-loop transition
        if let Some(adv) = extra_advance {
            adv(ptx);
        }
        ptx.blank();

        // ═══════════════════════════════════════════════════════════════
        // Precompute ldmatrix swizzle offsets (constant across K-loop)
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
        // K-LOOP
        // ═══════════════════════════════════════════════════════════════
        ptx.label("$L_KLOOP");

        // Wait for the oldest async group + barrier
        ptx.cp_async_wait_group(wait_count);
        ptx.bar_sync(0);

        // Compute read buffer base: smem_base + read_stage * a_tile_bytes
        let buf_base = ptx.regs.alloc_b32();
        {
            let read_off = ptx.regs.alloc_b32();
            ptx.shl_b32(read_off, read_stage, a_tile_bytes.trailing_zeros());
            ptx.add_s32(buf_base, smem_base, read_off);
        }
        ptx.blank();

        // ── ldmatrix.trans B ──
        // For each rn, issue b_ld_groups ldmatrix.x4.trans calls.
        // Group 0: rows 0-31 (4 regs), Group 1: rows 32-63 (4 more regs).
        // b_frags[rn] has 4*b_ld_groups registers total.
        ptx.comment(&format!("ldmatrix.trans B -- {} loads x {} groups", REG_N, b_ld_groups));
        let mut b_frags: Vec<Vec<Reg>> = Vec::new();
        for rn in 0..REG_N as usize {
            let mut frag_regs = Vec::new();
            for grp in 0..b_ld_groups as usize {
                let b_addr = ptx.regs.alloc_b32();
                ptx.add_s32(b_addr, buf_base, b_off[rn]);
                // Add offset for additional K-row groups:
                // Group 0 is at b_start (rows 0-31).
                // Group 1 is at b_start + 32*BN*2 = b_start + 4096 (rows 32-63).
                let grp_off = b_start + (grp as i32) * (32 * c.bn as i32 * 2);
                let frag = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                            ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];
                ptx.ldmatrix_x4_trans(frag, b_addr, Some(grp_off));
                frag_regs.extend_from_slice(&frag);
            }
            b_frags.push(frag_regs);
        }
        ptx.blank();

        // ── Unrolled ki loop: for each ki, load A, transform, MMA ──
        for ki in 0..k_warp_iters as usize {
            ptx.comment(&format!("ki={ki}: ldmatrix A + transform + MMA"));
            transform_a.emit_k_setup(ptx, ki as u32);
            let a_addr = ptx.regs.alloc_b32();
            ptx.add_s32(a_addr, buf_base, a_off[ki]);

            // A fragments for each rm, immediately consumed by MMA
            for rm in 0..REG_M as usize {
                let mut a_frag = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                                  ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];
                // rm=1 data is in the next 2048-byte chunk from rm=0 within
                // the same K-column group. This is always +2048 regardless of BK.
                // (Chunks are laid out as: rm=0-K0, rm=1-K0, rm=0-K1, rm=1-K1, ...)
                let a_rm_off = if rm == 0 { None } else { Some(2048_i32) };
                ptx.ldmatrix_x4(a_frag, a_addr, a_rm_off);
                transform_a.emit_transform(ptx, &mut a_frag, ki as u32, rm as u32);

                // MMA with all rn for this rm
                for rn in 0..REG_N as usize {
                    let ai = rm * REG_N as usize + rn;
                    let mut acc_tile = acc.regs[ai];
                    let b_sel = [b_frags[rn][ki * 2], b_frags[rn][ki * 2 + 1]];
                    mma.emit_mma(ptx, a_frag, b_sel, &mut acc_tile);
                }
            }
            ptx.blank();
        }

        // ── Predicated loads for next tile ──
        let p_load = ptx.regs.alloc_pred();
        ptx.setp_lt_s32(p_load, k_counter, k_minus_stages_bk);

        ptx.comment("Predicated loads for next tile");
        let cp_size = ptx.regs.alloc_b32();
        ptx.selp_b32(cp_size, 16, 0, p_load);

        // Compute write buffer base: smem_base + write_stage * a_tile_bytes
        let write_buf_base = ptx.regs.alloc_b32();
        {
            let write_off = ptx.regs.alloc_b32();
            ptx.shl_b32(write_off, write_stage, a_tile_bytes.trailing_zeros());
            ptx.add_s32(write_buf_base, smem_base, write_off);
        }

        ptx.bar_sync(0);

        // A next tile
        for i in 0..cp_chunks_a as usize {
            let a_dst = ptx.regs.alloc_b32();
            ptx.add_s32(a_dst, write_buf_base, a_cp_off);
            if i > 0 {
                ptx.add_s32_imm(a_dst, a_dst, (i * 2048) as i32);
            }
            copy_a.emit_async_copy(ptx, a_dst, ga_loop[i], cp_size);
        }
        copy_a.emit_commit(ptx);

        // B next tile: write_stage * b_tile_bytes offset into B region
        for i in 0..cp_chunks_b as usize {
            let b_dst = ptx.regs.alloc_b32();
            ptx.add_s32(b_dst, write_buf_base, b_cp_off);
            // B region offset: b_start + write_stage * b_tile_bytes + chunk * 2048
            // But write_buf_base already has write_stage * a_tile_bytes from smem_base.
            // Since a_tile_bytes == b_tile_bytes (BM==BN), the write_stage offset is
            // correct for both A and B with the same shift. But b_start accounts for
            // the A/B region separation.
            ptx.add_s32_imm(b_dst, b_dst, b_start + (i as i32 * 2048));
            copy_b.emit_async_copy(ptx, b_dst, gb_loop[i], cp_size);
        }
        copy_b.emit_commit(ptx);
        ptx.blank();

        // ── Advance circular buffer stage indices ──
        ptx.comment("Advance circular buffer stage indices");
        {
            let next_read = ptx.regs.alloc_b32();
            ptx.add_s32_imm(next_read, read_stage, 1);
            let p_wrap_r = ptx.regs.alloc_pred();
            ptx.setp_gt_s32_imm(p_wrap_r, next_read, stages as i32 - 1);
            ptx.selp_b32_imm_reg(read_stage, 0, next_read, p_wrap_r);
        }
        {
            let next_write = ptx.regs.alloc_b32();
            ptx.add_s32_imm(next_write, write_stage, 1);
            let p_wrap_w = ptx.regs.alloc_pred();
            ptx.setp_gt_s32_imm(p_wrap_w, next_write, stages as i32 - 1);
            ptx.selp_b32_imm_reg(write_stage, 0, next_write, p_wrap_w);
        }

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
