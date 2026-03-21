use super::config::GemmConfig;
use super::{PtxBuilder, Reg};

/// Accumulator register map -- the fusion interface.
/// After the K-loop, these hold the partial results.
/// An activation can transform them in-place before the store.
pub struct AccumulatorMap {
    /// regs[rm * reg_n + rn] = [d0, d1, d2, d3]
    pub regs: Vec<[Reg; 4]>,
    pub reg_m: u32,
    pub reg_n: u32,
}

// Hard-coded register tile dimensions matching the hand-written PTX.
const REG_M: u32 = 2;
const REG_N: u32 = 4;

// ═══════════════════════════════════════════════════════════════════════════
// TileLoader trait — the abstraction over HOW tiles arrive in shared memory.
// ═══════════════════════════════════════════════════════════════════════════

/// Describes how to load one tile (A or B) into shared memory.
/// The GEMM infrastructure calls these methods instead of hardcoding cp.async.
///
/// There are two "chunks" per tile (each 2048 bytes = 128 threads * 16 bytes).
/// Chunk 0 covers rows 0..31, chunk 1 covers rows 32..63 (for A).
pub trait TileLoader {
    /// Emit the load for prologue tile `tile_idx` (0 or 1).
    /// `smem_dst0/1`: smem addresses for chunk 0 and chunk 1.
    /// `g_ptr0/1`: global memory pointers for chunk 0 and chunk 1.
    /// `cp_size`: predicated size register (16 or 0).
    fn emit_prologue_load(
        &self,
        ptx: &mut PtxBuilder,
        smem_dst0: Reg,
        smem_dst1: Reg,
        g_ptr0: Reg,
        g_ptr1: Reg,
        cp_size: Reg,
    );

    /// Emit the load in the K-loop body (for next tile, predicated).
    /// Same signature as prologue but with an additional predicate.
    fn emit_loop_load(
        &self,
        ptx: &mut PtxBuilder,
        smem_dst0: Reg,
        smem_dst1: Reg,
        g_ptr0: Reg,
        g_ptr1: Reg,
        cp_size: Reg,
        p_load: Reg,
    );

    /// Emit a cp.async.commit_group after tile loads (if needed).
    fn emit_commit(&self, ptx: &mut PtxBuilder);

    /// The number of cp.async groups in flight from this loader.
    /// CpAsync returns 1 (we commit after each tile).
    /// Normalized returns 0 (writes directly, no async).
    fn async_groups_per_tile(&self) -> u32;

    /// Does this loader need a bar.sync after loading to ensure smem writes are visible?
    fn needs_barrier_after_load(&self) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════════
// CpAsyncLoader — wraps cp.async hardware DMA
// ═══════════════════════════════════════════════════════════════════════════

pub struct CpAsyncLoader;

impl TileLoader for CpAsyncLoader {
    fn emit_prologue_load(
        &self,
        ptx: &mut PtxBuilder,
        smem_dst0: Reg,
        smem_dst1: Reg,
        g_ptr0: Reg,
        g_ptr1: Reg,
        cp_size: Reg,
    ) {
        ptx.cp_async_cg(smem_dst0, 0, g_ptr0, 0, cp_size);
        ptx.cp_async_cg(smem_dst1, 0, g_ptr1, 0, cp_size);
    }

    fn emit_loop_load(
        &self,
        ptx: &mut PtxBuilder,
        smem_dst0: Reg,
        smem_dst1: Reg,
        g_ptr0: Reg,
        g_ptr1: Reg,
        cp_size: Reg,
        _p_load: Reg,
    ) {
        // cp.async uses cp_size = selp(16, 0, p_load) for predication
        ptx.cp_async_cg(smem_dst0, 0, g_ptr0, 0, cp_size);
        ptx.cp_async_cg(smem_dst1, 0, g_ptr1, 0, cp_size);
    }

    fn emit_commit(&self, ptx: &mut PtxBuilder) {
        ptx.cp_async_commit();
    }

    fn async_groups_per_tile(&self) -> u32 {
        1
    }
    fn needs_barrier_after_load(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NormalizedLoader — loads from global, normalizes, writes to smem
// ═══════════════════════════════════════════════════════════════════════════

/// NormalizedLoader context. The GEMM builder must set these up before
/// calling the K-loop.
pub struct NormalizedLoader {
    /// .b64 register: RMSNorm weight pointer (current K position).
    pub wnorm_ptr: Reg,
    /// .b32 register: base of per-row norm factors in smem.
    pub norm_factors_base: Reg,
    /// .b32 register: global row for chunk 0 (block_row + a_tid_row).
    pub row_id_chunk0: Reg,
    /// .b32 register: global row for chunk 1 (chunk0 + 32).
    pub row_id_chunk1: Reg,
    /// .b32 register: block_row.
    pub block_row: Reg,
}

impl NormalizedLoader {
    fn emit_chunk_store(
        &self,
        ptx: &mut PtxBuilder,
        g_ptr: Reg,
        smem_dst: Reg,
        row_id: Reg,
        predicate: Option<Reg>,
    ) {
        super::tile::emit_normalized_chunk_store(
            ptx,
            g_ptr,
            self.wnorm_ptr,
            smem_dst,
            row_id,
            self.norm_factors_base,
            self.block_row,
            predicate,
        );
    }
}

impl TileLoader for NormalizedLoader {
    fn emit_prologue_load(
        &self,
        ptx: &mut PtxBuilder,
        smem_dst0: Reg,
        smem_dst1: Reg,
        g_ptr0: Reg,
        g_ptr1: Reg,
        _cp_size: Reg,
    ) {
        self.emit_chunk_store(ptx, g_ptr0, smem_dst0, self.row_id_chunk0, None);
        self.emit_chunk_store(ptx, g_ptr1, smem_dst1, self.row_id_chunk1, None);
    }

    fn emit_loop_load(
        &self,
        ptx: &mut PtxBuilder,
        smem_dst0: Reg,
        smem_dst1: Reg,
        g_ptr0: Reg,
        g_ptr1: Reg,
        _cp_size: Reg,
        p_load: Reg,
    ) {
        self.emit_chunk_store(ptx, g_ptr0, smem_dst0, self.row_id_chunk0, Some(p_load));
        self.emit_chunk_store(ptx, g_ptr1, smem_dst1, self.row_id_chunk1, Some(p_load));
    }

    fn emit_commit(&self, _ptx: &mut PtxBuilder) {
        // NormalizedLoader writes directly via st.shared -- no async commit needed.
    }

    fn async_groups_per_tile(&self) -> u32 {
        0
    }
    fn needs_barrier_after_load(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reusable GEMM infrastructure
// ═══════════════════════════════════════════════════════════════════════════

/// State computed during GEMM setup that callers may need for store phase.
pub struct GemmSetup {
    pub block_row: Reg,
    pub block_col: Reg,
    pub warp_id: Reg,
    pub lane: Reg,
    pub group: Reg,
    pub tg: Reg,
    pub smem_base: Reg,
    pub tid: Reg,
}

/// Emit the complete GEMM K-loop using the given tile loaders.
///
/// Returns the accumulator map. This function handles:
/// - Prologue (load tiles 0 and 1)
/// - K-loop (ldmatrix, MMA, predicated next-tile loads)
/// - Epilogue (wait for async, barrier)
///
/// Callers provide:
/// - Global A/B pointers (chunk 0 and chunk 1 for each)
/// - TileLoader implementations for A and B
/// - Additional A-loader state (wnorm_ptr advancement etc.) via closures
pub fn emit_gemm_with_loaders(
    ptx: &mut PtxBuilder,
    c: &GemmConfig,
    setup: &GemmSetup,
    // A tile loading
    a_loader: &dyn TileLoader,
    ga0: Reg,
    ga1: Reg,      // A global ptrs for chunk 0/1
    a_cp_off: Reg, // A swizzled smem offset
    // B tile loading
    b_loader: &dyn TileLoader,
    gb0: Reg,
    gb1: Reg,      // B global ptrs for chunk 0/1
    b_cp_off: Reg, // B swizzled smem offset
    // B advancement
    n_param: Reg,
    k_param: Reg,
    // Extra per-iteration advance callback for A (e.g., wnorm advance)
    extra_advance: Option<&dyn Fn(&mut PtxBuilder)>,
) -> AccumulatorMap {
    let smem_base = setup.smem_base;
    let tid = setup.tid;

    let a_tile_bytes = c.smem_a_bytes(); // 4096
    let buf_stride = a_tile_bytes; // 4096
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
    ptx.comment("Prologue: load tile 0 into buffer 0");
    let p_tile0 = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_tile0, k_param, 0);
    let sz0 = ptx.regs.alloc_b32();
    ptx.selp_b32(sz0, 16, 0, p_tile0);

    // A tile 0
    a_loader.emit_prologue_load(ptx, a_st0, a_st1, ga0, ga1, sz0);
    if a_loader.needs_barrier_after_load() {
        ptx.bar_sync(0);
    }
    a_loader.emit_commit(ptx);

    // B tile 0
    b_loader.emit_prologue_load(ptx, b_cp0, b_cp1, gb0, gb1, sz0);
    b_loader.emit_commit(ptx);
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
    ptx.comment("Prologue: load tile 1 into buffer 1");
    let p_tile1 = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_tile1, k_param, c.bk as i32);

    // Advance extra state for tile 1 (e.g., wnorm pointer)
    if let Some(adv) = extra_advance {
        adv(ptx);
    }

    if !a_loader.needs_barrier_after_load() {
        // Only need barrier here if A didn't already do one
        ptx.bar_sync(0);
    }

    // Buffer 1 A smem addresses
    let a_st0_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_st0_b1, a_st0, buf_stride as i32);
    let a_st1_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_st1_b1, a_st1, buf_stride as i32);

    let sz1 = ptx.regs.alloc_b32();
    ptx.selp_b32(sz1, 16, 0, p_tile1);

    // A tile 1
    a_loader.emit_prologue_load(ptx, a_st0_b1, a_st1_b1, ga0_t1, ga1_t1, sz1);
    if a_loader.needs_barrier_after_load() {
        ptx.bar_sync(0);
    }
    a_loader.emit_commit(ptx);

    // Buffer 1 B
    let b_cp0_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp0_b1, b_cp_base_smem, b_start + buf_stride as i32);
    let b_cp1_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp1_b1, b_cp_base_smem, b_start + buf_stride as i32 + 2048);
    b_loader.emit_prologue_load(ptx, b_cp0_b1, b_cp1_b1, gb0_t1, gb1_t1, sz1);
    b_loader.emit_commit(ptx);
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
    // Initialize accumulators
    // ═══════════════════════════════════════════════════════════════
    ptx.comment("Initialize accumulators to 0.0f");
    let zero = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(zero, 0x00000000);

    let num_tiles = (REG_M * REG_N) as usize;
    let mut acc_regs: Vec<[Reg; 4]> = Vec::with_capacity(num_tiles);
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

    // Compute total async groups in flight from both loaders per tile pair
    let a_async = a_loader.async_groups_per_tile();
    let b_async = b_loader.async_groups_per_tile();
    let total_async_per_tile = a_async + b_async;
    // Wait group count: we have 2 tiles in flight, so wait for (total * 2 - tiles_consumed)
    // Actually the pattern is: wait_group = number of outstanding groups we want to keep
    // For pure cp.async (A+B each commit once per tile, 2 tiles in flight) = wait_group 2
    // For mixed (A normalized + B cp.async): B has 1 group per tile, so wait_group 1
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
    ptx.comment("ldmatrix A -- 4 loads");
    let a_addr_ki0 = ptx.regs.alloc_b32();
    ptx.add_s32(a_addr_ki0, buf_base, a_off_ki0);
    let a_frag_ki0_rm0 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki0_rm0, a_addr_ki0, None);

    let a_frag_ki0_rm1 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki0_rm1, a_addr_ki0, Some(2048));

    let a_addr_ki1 = ptx.regs.alloc_b32();
    ptx.add_s32(a_addr_ki1, buf_base, a_off_ki1);
    let a_frag_ki1_rm0 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki1_rm0, a_addr_ki1, None);

    let a_frag_ki1_rm1 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki1_rm1, a_addr_ki1, Some(2048));
    ptx.blank();

    // ── ldmatrix.trans B ──
    ptx.comment("ldmatrix.trans B -- 4 loads");
    let mut b_frags: Vec<[Reg; 4]> = Vec::new();
    for rn in 0..REG_N as usize {
        let b_addr = ptx.regs.alloc_b32();
        ptx.add_s32(b_addr, buf_base, b_off[rn]);
        let frag = [
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
        ];
        ptx.ldmatrix_x4_trans(frag, b_addr, Some(b_start));
        b_frags.push(frag);
    }
    ptx.blank();

    // ── MMA instructions ──
    ptx.comment("MMA -- 16 total");
    for rn in 0..REG_N as usize {
        let ai = rn;
        ptx.mma_m16n8k16(
            acc.regs[ai],
            a_frag_ki0_rm0,
            [b_frags[rn][0], b_frags[rn][1]],
            acc.regs[ai],
        );
    }
    for rn in 0..REG_N as usize {
        let ai = 1 * REG_N as usize + rn;
        ptx.mma_m16n8k16(
            acc.regs[ai],
            a_frag_ki0_rm1,
            [b_frags[rn][0], b_frags[rn][1]],
            acc.regs[ai],
        );
    }
    for rn in 0..REG_N as usize {
        let ai = rn;
        ptx.mma_m16n8k16(
            acc.regs[ai],
            a_frag_ki1_rm0,
            [b_frags[rn][2], b_frags[rn][3]],
            acc.regs[ai],
        );
    }
    for rn in 0..REG_N as usize {
        let ai = 1 * REG_N as usize + rn;
        ptx.mma_m16n8k16(
            acc.regs[ai],
            a_frag_ki1_rm1,
            [b_frags[rn][2], b_frags[rn][3]],
            acc.regs[ai],
        );
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
    a_loader.emit_loop_load(ptx, a_dst0, a_dst1, ga0_loop, ga1_loop, cp_size, p_load);
    if a_loader.needs_barrier_after_load() {
        // Need to ensure normalized writes are visible before B cp.async reads smem
        // Actually normalized A writes to A region, cp.async B writes to B region,
        // they don't conflict. But we need bar before ldmatrix reads in next iteration.
        // The bar at the top of the loop handles this.
    }
    a_loader.emit_commit(ptx);

    // B next tile
    let b_dst_base = ptx.regs.alloc_b32();
    ptx.add_s32(b_dst_base, write_buf_base, b_cp_off);
    let b_dst0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_dst0, b_dst_base, b_start);
    let b_dst1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_dst1, b_dst_base, b_start + 2048);
    b_loader.emit_loop_load(ptx, b_dst0, b_dst1, gb0_loop, gb1_loop, cp_size, p_load);
    b_loader.emit_commit(ptx);
    ptx.blank();

    // ── Advance loop state ──
    ptx.comment("Advance loop state");
    ptx.add_s32_imm(k_counter, k_counter, c.bk as i32);
    ptx.add_s64_imm(ga0_loop, ga0_loop, (c.bk * 2) as i64);
    ptx.add_s64_imm(ga1_loop, ga1_loop, (c.bk * 2) as i64);
    ptx.mov_b64(gb0_loop, gb0_next);
    ptx.mov_b64(gb1_loop, gb1_next);

    // Extra advance (e.g., wnorm pointer)
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

    acc
}

/// Emit the C matrix store phase. Shared between standalone GEMM and fused kernels.
pub fn emit_store_c(
    ptx: &mut PtxBuilder,
    c: &GemmConfig,
    acc: &AccumulatorMap,
    setup: &GemmSetup,
    output_ptr: Reg,
    n_param: Reg,
) {
    ptx.comment("Store C");

    let tg2 = ptx.regs.alloc_b32();
    ptx.shl_b32(tg2, setup.tg, 1);
    let tg2p1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(tg2p1, tg2, 1);
    let group8 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(group8, setup.group, 8);

    let warp_col_base = ptx.regs.alloc_b32();
    let warp_x16 = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_x16, setup.warp_id, 4);
    ptx.add_s32(warp_col_base, setup.block_col, warp_x16);

    let row_a0 = ptx.regs.alloc_b32();
    ptx.add_s32(row_a0, setup.block_row, setup.group);
    let row_b0 = ptx.regs.alloc_b32();
    ptx.add_s32(row_b0, setup.block_row, group8);

    let row_a0_x_n = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(row_a0_x_n, row_a0, n_param);
    let row_b0_x_n = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(row_b0_x_n, row_b0, n_param);

    for rm in 0..acc.reg_m {
        let (ra_x_n, rb_x_n) = if rm == 0 {
            (row_a0_x_n, row_b0_x_n)
        } else {
            let ra = ptx.regs.alloc_b32();
            ptx.add_s32_imm(ra, row_a0, (rm * c.mma_m * 2) as i32);
            let rb = ptx.regs.alloc_b32();
            ptx.add_s32_imm(rb, row_b0, (rm * c.mma_m * 2) as i32);
            let ra_n = ptx.regs.alloc_b32();
            ptx.mul_lo_s32(ra_n, ra, n_param);
            let rb_n = ptx.regs.alloc_b32();
            ptx.mul_lo_s32(rb_n, rb, n_param);
            (ra_n, rb_n)
        };

        for rn in 0..acc.reg_n {
            let ai = (rm * acc.reg_n + rn) as usize;
            let col_base = ptx.regs.alloc_b32();
            if rn > 0 {
                ptx.add_s32_imm(col_base, warp_col_base, (rn * c.mma_n) as i32);
            } else {
                ptx.mov_b32(col_base, warp_col_base);
            }
            let col0 = ptx.regs.alloc_b32();
            ptx.add_s32(col0, col_base, tg2);

            let idx0 = ptx.regs.alloc_b32();
            ptx.add_s32(idx0, ra_x_n, col0);
            let idx1 = ptx.regs.alloc_b32();
            ptx.add_s32_imm(idx1, idx0, 1);
            let idx2 = ptx.regs.alloc_b32();
            ptx.add_s32(idx2, rb_x_n, col0);
            let idx3 = ptx.regs.alloc_b32();
            ptx.add_s32_imm(idx3, idx2, 1);

            let four = ptx.regs.alloc_b32();
            ptx.mov_b32_imm(four, 4);

            let addr0 = ptx.regs.alloc_b64();
            ptx.mad_wide_s32(addr0, idx0, four, output_ptr);
            let addr1 = ptx.regs.alloc_b64();
            ptx.mad_wide_s32(addr1, idx1, four, output_ptr);
            let addr2 = ptx.regs.alloc_b64();
            ptx.mad_wide_s32(addr2, idx2, four, output_ptr);
            let addr3 = ptx.regs.alloc_b64();
            ptx.mad_wide_s32(addr3, idx3, four, output_ptr);

            ptx.st_global_b32(addr0, 0, acc.regs[ai][0]);
            ptx.st_global_b32(addr1, 0, acc.regs[ai][1]);
            ptx.st_global_b32(addr2, 0, acc.regs[ai][2]);
            ptx.st_global_b32(addr3, 0, acc.regs[ai][3]);
        }
    }
}

/// Emit GEMM setup: thread indices, block coordinates, smem base.
pub fn emit_gemm_setup(ptx: &mut PtxBuilder, c: &GemmConfig) -> GemmSetup {
    let block_x = ptx.regs.alloc_b32();
    let block_y = ptx.regs.alloc_b32();
    let tid = ptx.regs.alloc_b32();
    ptx.mov_b32_name(block_x, "%ctaid.x");
    ptx.mov_b32_name(block_y, "%ctaid.y");
    ptx.mov_b32_name(tid, "%tid.x");

    let block_row = ptx.regs.alloc_b32();
    ptx.shl_b32(block_row, block_y, c.bm.trailing_zeros());
    let block_col = ptx.regs.alloc_b32();
    ptx.shl_b32(block_col, block_x, c.bn.trailing_zeros());

    let warp_id = ptx.regs.alloc_b32();
    ptx.bfe_u32(warp_id, tid, 5, 2);
    let lane = ptx.regs.alloc_b32();
    ptx.and_b32(lane, tid, 31);

    let group = ptx.regs.alloc_b32();
    ptx.shr_u32(group, lane, 2);
    let tg = ptx.regs.alloc_b32();
    ptx.and_b32(tg, lane, 3);

    let smem_base = ptx.regs.alloc_b32();
    ptx.mov_b32_name(smem_base, "global_smem");
    ptx.blank();

    GemmSetup {
        block_row,
        block_col,
        warp_id,
        lane,
        group,
        tg,
        smem_base,
        tid,
    }
}

/// Emit cp.async swizzle address computation for A and B.
/// Returns (a_cp_off, b_cp_off).
pub fn emit_cpasync_swizzle(ptx: &mut PtxBuilder, tid: Reg) -> (Reg, Reg) {
    ptx.comment("cp.async swizzle addresses");
    let tid_x16 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x16, tid, 4);
    let cp_base = ptx.regs.alloc_b32();
    ptx.and_b32(cp_base, tid_x16, 2032);
    let tid_x2 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x2, tid, 1);

    let a_swiz = ptx.regs.alloc_b32();
    ptx.and_b32(a_swiz, tid_x2, 48);
    let a_cp_off = ptx.regs.alloc_b32();
    ptx.xor_b32(a_cp_off, cp_base, a_swiz);

    let b_swiz = ptx.regs.alloc_b32();
    ptx.and_b32(b_swiz, tid_x2, 112);
    let b_cp_off = ptx.regs.alloc_b32();
    ptx.xor_b32(b_cp_off, cp_base, b_swiz);
    ptx.blank();

    (a_cp_off, b_cp_off)
}

/// Emit A global address computation.
/// Returns (ga0, ga1, a_tid_row, a_tid_col).
pub fn emit_a_global_addrs(
    ptx: &mut PtxBuilder,
    block_row: Reg,
    k_param: Reg,
    a_ptr: Reg,
    tid: Reg,
) -> (Reg, Reg, Reg, Reg) {
    ptx.comment("Global addresses for A");
    let a_tid_and3 = ptx.regs.alloc_b32();
    ptx.and_b32(a_tid_and3, tid, 3);
    let a_tid_col = ptx.regs.alloc_b32();
    ptx.shl_b32(a_tid_col, a_tid_and3, 3);
    let a_tid_row = ptx.regs.alloc_b32();
    ptx.shr_u32(a_tid_row, tid, 2);

    let grow_a0 = ptx.regs.alloc_b32();
    ptx.add_s32(grow_a0, block_row, a_tid_row);
    let grow_a1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(grow_a1, grow_a0, 32);

    let a_gidx0 = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(a_gidx0, grow_a0, k_param);
    ptx.add_s32(a_gidx0, a_gidx0, a_tid_col);
    let ga0 = ptx.regs.alloc_b64();
    ptx.mad_wide_s32_imm(ga0, a_gidx0, 2, a_ptr);

    let a_gidx1 = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(a_gidx1, grow_a1, k_param);
    ptx.add_s32(a_gidx1, a_gidx1, a_tid_col);
    let ga1 = ptx.regs.alloc_b64();
    ptx.mad_wide_s32_imm(ga1, a_gidx1, 2, a_ptr);
    ptx.blank();

    (ga0, ga1, a_tid_row, a_tid_col)
}

/// Emit B global address computation.
/// Returns (gb0, gb1).
pub fn emit_b_global_addrs(
    ptx: &mut PtxBuilder,
    block_col: Reg,
    n_param: Reg,
    b_ptr: Reg,
    tid: Reg,
) -> (Reg, Reg) {
    ptx.comment("Global addresses for B");
    let b_tid_and7 = ptx.regs.alloc_b32();
    ptx.and_b32(b_tid_and7, tid, 7);
    let b_tid_col = ptx.regs.alloc_b32();
    ptx.shl_b32(b_tid_col, b_tid_and7, 3);
    let b_tid_row = ptx.regs.alloc_b32();
    ptx.shr_u32(b_tid_row, tid, 3);

    let brow0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(brow0, b_tid_row, 0);
    let brow1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(brow1, b_tid_row, 16);

    let gcol_b = ptx.regs.alloc_b32();
    ptx.add_s32(gcol_b, block_col, b_tid_col);

    let b_gidx0 = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(b_gidx0, brow0, n_param);
    ptx.add_s32(b_gidx0, b_gidx0, gcol_b);
    let gb0 = ptx.regs.alloc_b64();
    ptx.mad_wide_s32_imm(gb0, b_gidx0, 2, b_ptr);

    let b_gidx1 = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(b_gidx1, brow1, n_param);
    ptx.add_s32(b_gidx1, b_gidx1, gcol_b);
    let gb1 = ptx.regs.alloc_b64();
    ptx.mad_wide_s32_imm(gb1, b_gidx1, 2, b_ptr);
    ptx.blank();

    (gb0, gb1)
}

// ═══════════════════════════════════════════════════════════════════════════
// build_gemm — standalone GEMM kernel using CpAsyncLoader for both A and B
// ═══════════════════════════════════════════════════════════════════════════

/// Build a complete GEMM kernel PTX string.
/// Uses CpAsyncLoader for both A and B tiles — identical behavior to original.
pub fn build_gemm(config: &GemmConfig) -> String {
    let mut ptx = PtxBuilder::new(config.clone());
    let c = &ptx.config.clone();

    ptx.comment("GEMM kernel generated by PtxBuilder (TileLoader abstraction)");
    ptx.blank();

    // Phase 1: Load parameters
    let a_ptr = ptx.regs.alloc_b64();
    let b_ptr = ptx.regs.alloc_b64();
    let c_ptr = ptx.regs.alloc_b64();
    ptx.ld_param_b64(a_ptr, "param_A");
    ptx.ld_param_b64(b_ptr, "param_B");
    ptx.ld_param_b64(c_ptr, "param_C");
    let n_param = ptx.regs.alloc_b32();
    let k_param = ptx.regs.alloc_b32();
    ptx.ld_param_b32(n_param, "param_N");
    ptx.ld_param_b32(k_param, "param_K");
    ptx.blank();

    // Phase 2: Thread/block setup
    let setup = emit_gemm_setup(&mut ptx, c);

    // Phase 3: cp.async swizzle addresses
    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle(&mut ptx, setup.tid);

    // Phase 4: Global addresses
    let (ga0, ga1, _a_tid_row, _a_tid_col) =
        emit_a_global_addrs(&mut ptx, setup.block_row, k_param, a_ptr, setup.tid);
    let (gb0, gb1) = emit_b_global_addrs(&mut ptx, setup.block_col, n_param, b_ptr, setup.tid);

    // Phase 5-13: K-loop with CpAsyncLoader for both A and B
    let a_loader = CpAsyncLoader;
    let b_loader = CpAsyncLoader;

    let acc = emit_gemm_with_loaders(
        &mut ptx, c, &setup, &a_loader, ga0, ga1, a_cp_off, &b_loader, gb0, gb1, b_cp_off, n_param,
        k_param, None, // no extra advance
    );

    // Phase 14: Store C
    emit_store_c(&mut ptx, c, &acc, &setup, c_ptr, n_param);
    ptx.blank();
    ptx.ret();

    ptx.finalize("triton_style_gemm", &gemm_params())
}

fn gemm_params() -> Vec<(&'static str, &'static str)> {
    vec![
        (".u64 .ptr .global .align 16", "param_A"),
        (".u64 .ptr .global .align 16", "param_B"),
        (".u64 .ptr .global .align 16", "param_C"),
        (".u32", "param_M"),
        (".u32", "param_N"),
        (".u32", "param_K"),
    ]
}
