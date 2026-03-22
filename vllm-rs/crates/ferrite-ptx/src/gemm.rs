use crate::config::GemmConfig;
use crate::{PtxBuilder, Reg};

// ═══════════════════════════════════════════════════════════════════════════
// FragmentTransform — transforms A fragment registers between ldmatrix & MMA
//
// This is the general hook for fusing elementwise operations into the GEMM
// K-loop without changing the tile loading strategy:
//
//   ldmatrix A → 4 b32 registers (packed f16 pairs)
//   OPTIONAL: fragment_transform(A_regs)  ← THIS HOOK
//   ldmatrix.trans B → 4 b32 registers
//   MMA(A_regs, B_regs, accumulators)
//
// Each b32 register holds a pair of f16 values. The transform operates
// on 4 such registers (= 8 packed f16) produced by one ldmatrix.x4.
// ═══════════════════════════════════════════════════════════════════════════

/// Transforms A fragment registers in-place between ldmatrix and MMA.
/// Each call transforms one ldmatrix output (4 b32 registers = 8 packed f16).
/// The transform knows which K-iteration and register-tile it's operating on.
pub trait FragmentTransform {
    /// One-time setup: emit any prologue code (e.g., preload gamma to smem).
    /// Called once before the K-loop starts.
    fn emit_setup(&self, _ptx: &mut PtxBuilder) {}

    /// Per-K-iteration setup (e.g., load gamma values for this K chunk).
    /// Called once per K-loop iteration, before any ldmatrix calls.
    /// `ki_base` is the K-counter value at the start of this iteration.
    fn emit_k_iter_setup(&self, _ptx: &mut PtxBuilder, _ki_base: Reg) {}

    /// Transform 4 b32 registers in-place. Called once per ldmatrix.
    /// `ki` = which K-iteration within BK (0 or 1 for BK=32, corresponding to k16 halves)
    /// `rm` = which M-tile (0..REG_M-1)
    fn emit_transform(&self, ptx: &mut PtxBuilder, regs: &mut [Reg; 4], ki: u32, rm: u32);
}

/// No-op transform (standalone GEMM uses this).
/// Produces zero additional PTX instructions.
pub struct IdentityTransform;

impl FragmentTransform for IdentityTransform {
    fn emit_transform(&self, _ptx: &mut PtxBuilder, _regs: &mut [Reg; 4], _ki: u32, _rm: u32) {}
}

/// Scratch registers for the RmsNorm transform. Pre-allocated ONCE, reused for every call.
pub struct RmsNormScratch {
    // Norm factor lookup
    pub nf_off: Reg,     // b32
    pub nf_addr: Reg,    // b32
    pub nf_b32: Reg,     // b32: raw f32 norm_factor from smem
    pub nf_packed: Reg,  // b32: norm_factor packed as f16x2 (same value in both halves)
    pub nf_addr2: Reg,   // b32
    pub nf_b32_2: Reg,   // b32
    pub nf_packed2: Reg, // b32: second norm_factor packed as f16x2
    pub nf_f16: Reg,     // b32: temp for f32→f16 conversion
    // Gamma lookup
    pub gamma_k: Reg,       // b32
    pub gamma_byte: Reg,    // b32
    pub gamma_addr: Reg,    // b32
    pub gamma_packed: Reg,  // b32: gamma[k], gamma[k+1] as packed f16x2
    pub gamma_addr2: Reg,   // b32
    pub gamma_packed2: Reg, // b32: gamma[k+8], gamma[k+9] as packed f16x2
    pub tg2: Reg,           // b32 (tg * 2, precomputed)
    pub tmp: Reg,           // b32: general temp
}

/// Multiplies each f16 element by norm_factor (per-row) * gamma (per-k-column).
/// Uses pre-allocated scratch registers to avoid register bloat.
pub struct RmsNormTransform {
    pub norm_factors_smem: Reg,
    pub gamma_smem: Reg,
    pub group_id: Reg,
    pub tg: Reg,
    pub k_counter: Reg,
    pub scratch: RmsNormScratch,
}

impl RmsNormTransform {
    pub fn new(
        ptx: &mut PtxBuilder,
        norm_factors_smem: Reg,
        gamma_smem: Reg,
        group_id: Reg,
        tg: Reg,
        k_counter: Reg,
    ) -> Self {
        let scratch = RmsNormScratch {
            nf_off: ptx.regs.alloc_b32(),
            nf_addr: ptx.regs.alloc_b32(),
            nf_b32: ptx.regs.alloc_b32(),
            nf_packed: ptx.regs.alloc_b32(),
            nf_addr2: ptx.regs.alloc_b32(),
            nf_b32_2: ptx.regs.alloc_b32(),
            nf_packed2: ptx.regs.alloc_b32(),
            nf_f16: ptx.regs.alloc_b32(),
            gamma_k: ptx.regs.alloc_b32(),
            gamma_byte: ptx.regs.alloc_b32(),
            gamma_addr: ptx.regs.alloc_b32(),
            gamma_packed: ptx.regs.alloc_b32(),
            gamma_addr2: ptx.regs.alloc_b32(),
            gamma_packed2: ptx.regs.alloc_b32(),
            tg2: ptx.regs.alloc_b32(),
            tmp: ptx.regs.alloc_b32(),
        };
        ptx.shl_b32(scratch.tg2, tg, 1);
        Self {
            norm_factors_smem,
            gamma_smem,
            group_id,
            tg,
            k_counter,
            scratch,
        }
    }
}

impl FragmentTransform for RmsNormTransform {
    fn emit_transform(&self, ptx: &mut PtxBuilder, regs: &mut [Reg; 4], ki: u32, rm: u32) {
        let s = &self.scratch;
        ptx.comment(&format!("RmsNorm f16x2 transform ki={ki} rm={rm}"));

        // Load norm_factor for a0/a1 rows, pack as f16x2
        ptx.shl_b32(s.nf_off, self.group_id, 2);
        if rm > 0 {
            ptx.add_s32_imm(s.nf_off, s.nf_off, (rm * 32 * 4) as i32);
        }
        ptx.add_s32(s.nf_addr, self.norm_factors_smem, s.nf_off);
        ptx.ld_shared_b32(s.nf_b32, s.nf_addr, 0);
        // Pack f32 norm_factor as f16x2 (same value in both halves)
        ptx.mov_b32_to_f32(s.nf_packed, s.nf_b32); // temporary reinterpret
        ptx.pack_f16x2_from_f32(s.nf_packed, s.nf_packed, s.nf_f16);

        // Norm factor for a2/a3 rows (group_id + 8)
        ptx.add_s32_imm(s.nf_addr2, s.nf_addr, 8 * 4);
        ptx.ld_shared_b32(s.nf_b32_2, s.nf_addr2, 0);
        ptx.mov_b32_to_f32(s.nf_packed2, s.nf_b32_2);
        ptx.pack_f16x2_from_f32(s.nf_packed2, s.nf_packed2, s.nf_f16);

        // Gamma for a0/a2: gamma[k_counter + ki*16 + tg*2] as packed f16x2
        // (already packed in smem as consecutive f16 pair)
        ptx.add_s32(s.gamma_k, self.k_counter, s.tg2);
        if ki > 0 {
            ptx.add_s32_imm(s.gamma_k, s.gamma_k, (ki * 16) as i32);
        }
        ptx.shl_b32(s.gamma_byte, s.gamma_k, 1);
        ptx.add_s32(s.gamma_addr, self.gamma_smem, s.gamma_byte);
        ptx.ld_shared_b32(s.gamma_packed, s.gamma_addr, 0); // {gamma[k], gamma[k+1]} packed

        // Gamma for a1/a3: gamma[k+8] as packed f16x2
        ptx.add_s32_imm(s.gamma_addr2, s.gamma_addr, 8 * 2);
        ptx.ld_shared_b32(s.gamma_packed2, s.gamma_addr2, 0);

        // Transform: 2 packed f16x2 multiplies per register!
        // a0: x * norm_factor_01 * gamma_01
        ptx.mul_rn_f16x2(regs[0], regs[0], s.nf_packed);
        ptx.mul_rn_f16x2(regs[0], regs[0], s.gamma_packed);
        // a1: x * norm_factor_01 * gamma_89
        ptx.mul_rn_f16x2(regs[1], regs[1], s.nf_packed);
        ptx.mul_rn_f16x2(regs[1], regs[1], s.gamma_packed2);
        // a2: x * norm_factor_23 * gamma_01 (same k-cols as a0, different row)
        ptx.mul_rn_f16x2(regs[2], regs[2], s.nf_packed2);
        ptx.mul_rn_f16x2(regs[2], regs[2], s.gamma_packed);
        // a3: x * norm_factor_23 * gamma_89
        ptx.mul_rn_f16x2(regs[3], regs[3], s.nf_packed2);
        ptx.mul_rn_f16x2(regs[3], regs[3], s.gamma_packed2);
    }
}

/// Accumulator register map -- the fusion interface.
/// After the K-loop, these hold the partial results.
/// An activation can transform them in-place before the store.
pub struct AccumulatorMap {
    /// regs[rm * reg_n + rn] = [d0, d1, d2, d3] (f32 accum) or [d0, d1] (f16 accum)
    pub regs: Vec<[Reg; 4]>,
    pub reg_m: u32,
    pub reg_n: u32,
}

/// Accumulator map for f16 accumulators (2 regs per MMA output tile).
/// Used by the dual GEMM pipeline to halve register pressure.
pub struct AccumulatorMapF16 {
    /// regs[rm * reg_n + rn] = [d0, d1] (packed f16x2 pairs)
    pub regs: Vec<[Reg; 2]>,
    pub reg_m: u32,
    pub reg_n: u32,
}

// Legacy hard-coded register tile dimensions for the emit_gemm_with_loaders path.
// These match the original 64×64 hand-written kernel constants.
// The new pipeline-based path (MainloopPipeline) derives REG_M/REG_N from GemmConfig.
const REG_M: u32 = 2;
const REG_N: u32 = 4;

// ═══════════════════════════════════════════════════════════════════════════
// TileLoader trait — the abstraction over HOW tiles arrive in shared memory.
// ═══════════════════════════════════════════════════════════════════════════

/// Pipeline schedule for the K-loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// Both A and B use cp.async -- standard async overlap.
    FullAsync,
    /// A uses register-based transform (load early, process after MMA).
    /// B uses cp.async. This hides NormalizedLoader's ld.global latency
    /// behind MMA compute.
    RegisterTransformA,
}

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

    /// Does this loader support split-phase operation?
    /// If true, `emit_issue_loads` and `emit_process_loads` can be called
    /// separately to overlap global loads with MMA compute.
    fn supports_split_phase(&self) -> bool {
        false
    }

    /// Phase 1: Issue global loads only. Returns register handles to in-flight data.
    /// Called BEFORE MMA to start loads early. The returned Vec contains
    /// 8 registers: [chunk0_v4[0..4], chunk1_v4[0..4]].
    fn emit_issue_loads(
        &self,
        _ptx: &mut PtxBuilder,
        _g_ptr0: Reg,
        _g_ptr1: Reg,
        _p_load: Reg,
    ) -> Vec<Reg> {
        Vec::new()
    }

    /// Phase 1 (unpredicated): Issue global loads for prologue tiles.
    fn emit_issue_loads_unpredicated(
        &self,
        _ptx: &mut PtxBuilder,
        _g_ptr0: Reg,
        _g_ptr1: Reg,
    ) -> Vec<Reg> {
        Vec::new()
    }

    /// Phase 2: Process previously issued loads (ALU normalize + st.shared).
    /// Called AFTER MMA, when loaded data has arrived.
    fn emit_process_loads(
        &self,
        _ptx: &mut PtxBuilder,
        _loaded_regs: &[Reg],
        _smem_dst0: Reg,
        _smem_dst1: Reg,
        _predicate: Option<Reg>,
    ) {
    }
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
        crate::tile::emit_normalized_chunk_store(
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

    /// Phase 1 helper: issue ld.global.v4.b32 for one chunk, plus weight load.
    /// Returns 8 registers: [input_v4[0..4], weight_v4[0..4]].
    fn emit_issue_chunk_loads(
        ptx: &mut PtxBuilder,
        g_ptr: Reg,
        wnorm_ptr: Reg,
        predicate: Option<Reg>,
    ) -> [Reg; 8] {
        let raw_in = [
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
        ];
        let raw_wt = [
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
        ];
        if let Some(pred) = predicate {
            ptx.pred_ld_global_v4_b32(pred, raw_in, g_ptr, 0);
            ptx.pred_ld_global_v4_b32(pred, raw_wt, wnorm_ptr, 0);
        } else {
            ptx.ld_global_v4_b32(raw_in, g_ptr, 0);
            ptx.ld_global_v4_b32(raw_wt, wnorm_ptr, 0);
        }
        [
            raw_in[0], raw_in[1], raw_in[2], raw_in[3], raw_wt[0], raw_wt[1], raw_wt[2], raw_wt[3],
        ]
    }

    /// Phase 2 helper: process loaded data for one chunk.
    /// Takes 8 regs [input[0..4], weight[0..4]], normalizes, stores to smem.
    fn emit_process_chunk(
        ptx: &mut PtxBuilder,
        loaded: &[Reg; 8],
        smem_dst: Reg,
        row_id: Reg,
        norm_factors_base: Reg,
        block_row: Reg,
        predicate: Option<Reg>,
    ) {
        // Load norm factor for this row from smem
        let local_row = ptx.regs.alloc_b32();
        ptx.sub_s32(local_row, row_id, block_row);
        let factor_off = ptx.regs.alloc_b32();
        ptx.shl_b32(factor_off, local_row, 2);
        let factor_addr = ptx.regs.alloc_b32();
        ptx.add_s32(factor_addr, norm_factors_base, factor_off);
        let norm_factor_b32 = ptx.regs.alloc_b32();
        ptx.ld_shared_b32(norm_factor_b32, factor_addr, 0);
        let norm_factor = ptx.regs.alloc_f32();
        ptx.mov_b32_to_f32(norm_factor, norm_factor_b32);

        // Process each pair: extract lo/hi f16, normalize, pack back, store
        for pair_idx in 0..4u32 {
            let inp = loaded[pair_idx as usize];
            let wgt = loaded[(pair_idx + 4) as usize];

            let x_lo = ptx.regs.alloc_f32();
            ptx.cvt_f32_f16(x_lo, inp);
            let w_lo = ptx.regs.alloc_f32();
            ptx.cvt_f32_f16(w_lo, wgt);
            let t_lo = ptx.regs.alloc_f32();
            ptx.mul_f32(t_lo, x_lo, norm_factor);
            let y_lo = ptx.regs.alloc_f32();
            ptx.mul_f32(y_lo, t_lo, w_lo);

            let hi_input = ptx.regs.alloc_b32();
            ptx.shr_u32(hi_input, inp, 16);
            let x_hi = ptx.regs.alloc_f32();
            ptx.cvt_f32_f16(x_hi, hi_input);
            let hi_weight = ptx.regs.alloc_b32();
            ptx.shr_u32(hi_weight, wgt, 16);
            let w_hi = ptx.regs.alloc_f32();
            ptx.cvt_f32_f16(w_hi, hi_weight);
            let t_hi = ptx.regs.alloc_f32();
            ptx.mul_f32(t_hi, x_hi, norm_factor);
            let y_hi = ptx.regs.alloc_f32();
            ptx.mul_f32(y_hi, t_hi, w_hi);

            let h_lo = ptx.regs.alloc_b32();
            ptx.cvt_rn_f16_f32(h_lo, y_lo);
            let h_hi = ptx.regs.alloc_b32();
            ptx.cvt_rn_f16_f32(h_hi, y_hi);
            let h_hi_shifted = ptx.regs.alloc_b32();
            ptx.shl_b32(h_hi_shifted, h_hi, 16);
            let packed = ptx.regs.alloc_b32();
            ptx.or_b32(packed, h_lo, h_hi_shifted);

            if let Some(pred) = predicate {
                ptx.pred_st_shared_b32(pred, smem_dst, (pair_idx * 4) as i32, packed);
            } else {
                ptx.st_shared_b32(smem_dst, (pair_idx * 4) as i32, packed);
            }
        }
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

    fn supports_split_phase(&self) -> bool {
        true
    }

    fn emit_issue_loads(
        &self,
        ptx: &mut PtxBuilder,
        g_ptr0: Reg,
        g_ptr1: Reg,
        p_load: Reg,
    ) -> Vec<Reg> {
        ptx.comment("Split-phase: issue ld.global for NEXT A tile (early)");
        let chunk0 =
            NormalizedLoader::emit_issue_chunk_loads(ptx, g_ptr0, self.wnorm_ptr, Some(p_load));
        let chunk1 =
            NormalizedLoader::emit_issue_chunk_loads(ptx, g_ptr1, self.wnorm_ptr, Some(p_load));
        // Return 16 regs: [chunk0_in[4], chunk0_wt[4], chunk1_in[4], chunk1_wt[4]]
        let mut result = Vec::with_capacity(16);
        result.extend_from_slice(&chunk0);
        result.extend_from_slice(&chunk1);
        result
    }

    fn emit_issue_loads_unpredicated(
        &self,
        ptx: &mut PtxBuilder,
        g_ptr0: Reg,
        g_ptr1: Reg,
    ) -> Vec<Reg> {
        ptx.comment("Split-phase: issue ld.global for A tile (prologue, unpredicated)");
        let chunk0 = NormalizedLoader::emit_issue_chunk_loads(ptx, g_ptr0, self.wnorm_ptr, None);
        let chunk1 = NormalizedLoader::emit_issue_chunk_loads(ptx, g_ptr1, self.wnorm_ptr, None);
        let mut result = Vec::with_capacity(16);
        result.extend_from_slice(&chunk0);
        result.extend_from_slice(&chunk1);
        result
    }

    fn emit_process_loads(
        &self,
        ptx: &mut PtxBuilder,
        loaded_regs: &[Reg],
        smem_dst0: Reg,
        smem_dst1: Reg,
        predicate: Option<Reg>,
    ) {
        ptx.comment("Split-phase: process loaded A data (normalize + st.shared)");
        // loaded_regs layout: [chunk0_in[4], chunk0_wt[4], chunk1_in[4], chunk1_wt[4]]
        let chunk0: [Reg; 8] = [
            loaded_regs[0],
            loaded_regs[1],
            loaded_regs[2],
            loaded_regs[3],
            loaded_regs[4],
            loaded_regs[5],
            loaded_regs[6],
            loaded_regs[7],
        ];
        let chunk1: [Reg; 8] = [
            loaded_regs[8],
            loaded_regs[9],
            loaded_regs[10],
            loaded_regs[11],
            loaded_regs[12],
            loaded_regs[13],
            loaded_regs[14],
            loaded_regs[15],
        ];
        NormalizedLoader::emit_process_chunk(
            ptx,
            &chunk0,
            smem_dst0,
            self.row_id_chunk0,
            self.norm_factors_base,
            self.block_row,
            predicate,
        );
        NormalizedLoader::emit_process_chunk(
            ptx,
            &chunk1,
            smem_dst1,
            self.row_id_chunk1,
            self.norm_factors_base,
            self.block_row,
            predicate,
        );
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
    // Optional fragment transform applied to A registers after ldmatrix
    a_transform: Option<&dyn FragmentTransform>,
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
    let mut a_frag_ki0_rm0 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki0_rm0, a_addr_ki0, None);
    if let Some(transform) = a_transform {
        transform.emit_transform(ptx, &mut a_frag_ki0_rm0, 0, 0);
    }

    let mut a_frag_ki0_rm1 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki0_rm1, a_addr_ki0, Some(2048));
    if let Some(transform) = a_transform {
        transform.emit_transform(ptx, &mut a_frag_ki0_rm1, 0, 1);
    }

    let a_addr_ki1 = ptx.regs.alloc_b32();
    ptx.add_s32(a_addr_ki1, buf_base, a_off_ki1);
    let mut a_frag_ki1_rm0 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki1_rm0, a_addr_ki1, None);
    if let Some(transform) = a_transform {
        transform.emit_transform(ptx, &mut a_frag_ki1_rm0, 1, 0);
    }

    let mut a_frag_ki1_rm1 = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];
    ptx.ldmatrix_x4(a_frag_ki1_rm1, a_addr_ki1, Some(2048));
    if let Some(transform) = a_transform {
        transform.emit_transform(ptx, &mut a_frag_ki1_rm1, 1, 1);
    }
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

/// Emit the GEMM K-loop using RegisterTransformA pipeline schedule.
///
/// This schedule hides the NormalizedLoader's ld.global latency behind MMA compute:
///   1. Wait for CURRENT tile (barrier)
///   2. Issue ld.global for NEXT A tile (early — data arrives during MMA)
///   3. ldmatrix + MMA on CURRENT tile (MMA hides ld.global latency)
///   4. Process loaded A data (normalize ALU + st.shared into NEXT buffer)
///   5. B cp.async for NEXT tile + commit
///   6. Barrier (ensure st.shared visible)
///   7. Loop
pub fn emit_gemm_register_transform_a(
    ptx: &mut PtxBuilder,
    c: &GemmConfig,
    setup: &GemmSetup,
    a_loader: &dyn TileLoader,
    ga0: Reg,
    ga1: Reg,
    a_cp_off: Reg,
    b_loader: &dyn TileLoader,
    gb0: Reg,
    gb1: Reg,
    b_cp_off: Reg,
    n_param: Reg,
    k_param: Reg,
    extra_advance: Option<&dyn Fn(&mut PtxBuilder)>,
) -> AccumulatorMap {
    assert!(
        a_loader.supports_split_phase(),
        "RegisterTransformA requires a split-phase A loader"
    );

    let smem_base = setup.smem_base;
    let tid = setup.tid;

    let a_tile_bytes = c.smem_a_bytes(); // 4096
    let buf_stride = a_tile_bytes; // 4096
    let b_start = (a_tile_bytes * c.num_stages) as i32; // 8192

    let tid_x16 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x16, tid, 4);

    // ═══════════════════════════════════════════════════════════════
    // A smem addresses (buffer 0)
    // ═══════════════════════════════════════════════════════════════
    let a_st0 = ptx.regs.alloc_b32();
    ptx.add_s32(a_st0, smem_base, a_cp_off);
    let a_st1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_st1, a_st0, 2048);

    // B smem addresses (buffer 0)
    let b_cp_base_smem = ptx.regs.alloc_b32();
    ptx.add_s32(b_cp_base_smem, smem_base, b_cp_off);
    let b_cp0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp0, b_cp_base_smem, b_start);
    let b_cp1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp1, b_cp_base_smem, b_start + 2048);
    ptx.blank();

    // ═══════════════════════════════════════════════════════════════
    // Prologue: load tile 0 into buffer 0 (split-phase for A)
    // ═══════════════════════════════════════════════════════════════
    ptx.comment("Prologue: load tile 0 into buffer 0 (split-phase A)");
    let p_tile0 = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_tile0, k_param, 0);
    let sz0 = ptx.regs.alloc_b32();
    ptx.selp_b32(sz0, 16, 0, p_tile0);

    // A tile 0: issue loads, then process immediately (no MMA to overlap yet)
    let a_t0_regs = a_loader.emit_issue_loads_unpredicated(ptx, ga0, ga1);
    a_loader.emit_process_loads(ptx, &a_t0_regs, a_st0, a_st1, None);
    ptx.bar_sync(0);

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
    // Prologue: load tile 1 into buffer 1 (split-phase for A)
    // ═══════════════════════════════════════════════════════════════
    ptx.comment("Prologue: load tile 1 into buffer 1 (split-phase A)");
    let p_tile1 = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_tile1, k_param, c.bk as i32);

    // Advance extra state for tile 1 (e.g., wnorm pointer)
    if let Some(adv) = extra_advance {
        adv(ptx);
    }

    // Buffer 1 A smem addresses
    let a_st0_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_st0_b1, a_st0, buf_stride as i32);
    let a_st1_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_st1_b1, a_st1, buf_stride as i32);

    // A tile 1: issue loads, then process immediately
    let a_t1_regs = a_loader.emit_issue_loads_unpredicated(ptx, ga0_t1, ga1_t1);
    a_loader.emit_process_loads(ptx, &a_t1_regs, a_st0_b1, a_st1_b1, None);
    ptx.bar_sync(0);

    // Buffer 1 B
    let sz1 = ptx.regs.alloc_b32();
    ptx.selp_b32(sz1, 16, 0, p_tile1);
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

    // B uses cp.async: 1 group per tile. Wait count = 1 (keep one tile in flight).
    let b_async = b_loader.async_groups_per_tile();
    let wait_count = b_async; // Only B contributes async groups

    // ═══════════════════════════════════════════════════════════════
    // K-LOOP (RegisterTransformA schedule)
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

    // ── Step 1: Wait for CURRENT tile (B async + A already in smem) ──
    ptx.cp_async_wait_group(wait_count);
    ptx.bar_sync(0);

    // Compute read buffer base
    let read_buf_off = ptx.regs.alloc_b32();
    ptx.shl_b32(read_buf_off, read_ctr, 12);
    let buf_base = ptx.regs.alloc_b32();
    ptx.add_s32(buf_base, smem_base, read_buf_off);
    ptx.blank();

    // ── Step 2: Issue early ld.global for NEXT A tile ──
    ptx.comment("Step 2: Issue early ld.global for NEXT A tile");
    let a_loaded_regs = a_loader.emit_issue_loads(ptx, ga0_loop, ga1_loop, p_load);
    ptx.blank();

    // ── Step 3: Compute on CURRENT tile (MMA hides ld.global latency) ──
    ptx.comment("Step 3: ldmatrix + MMA on CURRENT tile");

    // ldmatrix A
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

    // ldmatrix.trans B
    ptx.comment("ldmatrix.trans B");
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

    // MMA instructions
    ptx.comment("MMA -- 16 total (hides ld.global latency from step 2)");
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

    // ── Step 4: Process early-loaded A data (normalize + st.shared) ──
    // Toggle write buffer
    let write_next = ptx.regs.alloc_b32();
    ptx.add_s32_imm(write_next, write_ctr, 1);
    let p_reset_w = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_reset_w, write_next, 1);
    ptx.selp_b32_imm_reg(write_ctr, 0, write_next, p_reset_w);

    let write_buf_off = ptx.regs.alloc_b32();
    ptx.shl_b32(write_buf_off, write_ctr, 12);
    let write_buf_base = ptx.regs.alloc_b32();
    ptx.add_s32(write_buf_base, smem_base, write_buf_off);

    let a_dst0 = ptx.regs.alloc_b32();
    ptx.add_s32(a_dst0, write_buf_base, a_cp_off);
    let a_dst1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_dst1, a_dst0, 2048);

    ptx.comment("Step 4: Process early-loaded A data (normalize + st.shared)");
    a_loader.emit_process_loads(ptx, &a_loaded_regs, a_dst0, a_dst1, Some(p_load));
    ptx.blank();

    // ── Step 5: B cp.async for NEXT tile + commit ──
    ptx.comment("Step 5: B cp.async for NEXT tile");
    let gb0_next = ptx.regs.alloc_b64();
    ptx.add_s64(gb0_next, gb0_loop, b_stride_bytes);
    let gb1_next = ptx.regs.alloc_b64();
    ptx.add_s64(gb1_next, gb1_loop, b_stride_bytes);

    let cp_size = ptx.regs.alloc_b32();
    ptx.selp_b32(cp_size, 16, 0, p_load);

    let b_dst_base = ptx.regs.alloc_b32();
    ptx.add_s32(b_dst_base, write_buf_base, b_cp_off);
    let b_dst0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_dst0, b_dst_base, b_start);
    let b_dst1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_dst1, b_dst_base, b_start + 2048);
    b_loader.emit_loop_load(ptx, b_dst0, b_dst1, gb0_loop, gb1_loop, cp_size, p_load);
    b_loader.emit_commit(ptx);
    ptx.blank();

    // ── Step 6: Barrier (ensure A st.shared writes visible) ──
    ptx.comment("Step 6: Barrier (ensure A st.shared writes visible)");
    ptx.bar_sync(0);

    // ── Step 7: Advance loop state ──
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

// ═══════════════════════════════════════════════════════════════════════════
// CpAsyncTransformA — CUTLASS-style cp.async-then-transform pipeline
//
// Uses hardware DMA (cp.async) to load raw input into smem_A_temp,
// then transforms in shared memory: smem_A_temp → registers → smem_A.
// This keeps cp.async's full bandwidth for A loads while fusing normalization.
// ═══════════════════════════════════════════════════════════════════════════

/// Context for the cp.async transform approach.
pub struct CpAsyncTransformCtx {
    /// Byte offset of smem_A_temp buffer 0 relative to smem_base.
    pub a_temp_offset: u32,
    /// .b64: global pointer to norm weights (wnorm_ptr).
    pub wnorm_ptr: Reg,
    /// Byte offset of norm factors in smem.
    pub norm_factors_base: Reg,
    /// .b32: block_row register.
    pub block_row: Reg,
    /// Hidden size (for computing weight indices).
    pub hidden_size: u32,
}

/// Pre-allocated scratch registers for the transform phase.
/// Reused across all transform calls to avoid register pressure explosion.
pub struct TransformScratch {
    pub tid_row: Reg,
    pub tid_col: Reg,
    pub tid_and3: Reg,
    pub row_in_tile: [Reg; 2], // one per chunk
    pub temp_addr: [Reg; 2],
    pub dest_addr: [Reg; 2],
    pub raw: [Reg; 4],
    pub wgt: [Reg; 4],
    pub result: [Reg; 4],
    pub factor_off: Reg,
    pub factor_addr: Reg,
    pub norm_factor_b32: Reg,
    pub norm_factor: Reg,
    pub w_elem_off: Reg,
    pub w_byte_off: Reg,
    pub w_addr: Reg,   // reused as b64 for global weight address
    pub w_addr64: Reg, // b64 for global weight address
    // ALU scratch for each pair
    pub x_lo: Reg,
    pub w_lo: Reg,
    pub t_lo: Reg,
    pub y_lo: Reg,
    pub hi_input: Reg,
    pub x_hi: Reg,
    pub hi_weight: Reg,
    pub w_hi: Reg,
    pub t_hi: Reg,
    pub y_hi: Reg,
    pub h_lo: Reg,
    pub h_hi: Reg,
    pub h_hi_shifted: Reg,
    pub packed: Reg,
}

impl TransformScratch {
    pub fn alloc(ptx: &mut PtxBuilder) -> Self {
        Self {
            tid_row: ptx.regs.alloc_b32(),
            tid_col: ptx.regs.alloc_b32(),
            tid_and3: ptx.regs.alloc_b32(),
            row_in_tile: [ptx.regs.alloc_b32(), ptx.regs.alloc_b32()],
            temp_addr: [ptx.regs.alloc_b32(), ptx.regs.alloc_b32()],
            dest_addr: [ptx.regs.alloc_b32(), ptx.regs.alloc_b32()],
            raw: [
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ],
            wgt: [
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ],
            result: [
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ],
            factor_off: ptx.regs.alloc_b32(),
            factor_addr: ptx.regs.alloc_b32(),
            norm_factor_b32: ptx.regs.alloc_b32(),
            norm_factor: ptx.regs.alloc_f32(),
            w_elem_off: ptx.regs.alloc_b32(),
            w_byte_off: ptx.regs.alloc_b32(),
            w_addr: ptx.regs.alloc_b32(),
            w_addr64: ptx.regs.alloc_b64(),
            x_lo: ptx.regs.alloc_f32(),
            w_lo: ptx.regs.alloc_f32(),
            t_lo: ptx.regs.alloc_f32(),
            y_lo: ptx.regs.alloc_f32(),
            hi_input: ptx.regs.alloc_b32(),
            x_hi: ptx.regs.alloc_f32(),
            hi_weight: ptx.regs.alloc_b32(),
            w_hi: ptx.regs.alloc_f32(),
            t_hi: ptx.regs.alloc_f32(),
            y_hi: ptx.regs.alloc_f32(),
            h_lo: ptx.regs.alloc_b32(),
            h_hi: ptx.regs.alloc_b32(),
            h_hi_shifted: ptx.regs.alloc_b32(),
            packed: ptx.regs.alloc_b32(),
        }
    }
}

/// Emit the smem transform phase: read from smem_A_temp, apply norm_factor * norm_weight,
/// write to smem_A. Both use the SAME swizzle pattern, just different base addresses.
///
/// `a_buf_base` and `a_temp_buf_base` are registers holding the absolute smem addresses
/// of the current buffer's A tile and A_temp tile respectively (already include buffer offset).
fn emit_smem_transform_dynamic(
    ptx: &mut PtxBuilder,
    _smem_base: Reg,
    a_cp_off: Reg,          // swizzled cp.async offset for this thread
    a_buf_base: Reg,        // .b32: smem_base + read_buf_off (A tile base)
    a_temp_buf_base: Reg,   // .b32: smem_base + read_buf_off + a_temp_start (A_temp base)
    norm_factors_base: Reg, // .b32: smem base of norm factors
    wnorm_ptr: Reg,         // .b64: global pointer to norm weights
    tid: Reg,
    k_offset: Reg, // current K offset (in elements) for norm weight lookup
    scratch: &TransformScratch,
) {
    ptx.comment("Transform: smem_A_temp -> normalize -> smem_A");

    let s = scratch;

    // Compute this thread's col within the tile
    ptx.shr_u32(s.tid_row, tid, 2);
    ptx.and_b32(s.tid_and3, tid, 3);
    ptx.shl_b32(s.tid_col, s.tid_and3, 3); // col = (tid & 3) * 8

    // Process chunk 0 (rows 0..31) and chunk 1 (rows 32..63)
    for chunk in 0..2u32 {
        ptx.add_s32_imm(
            s.row_in_tile[chunk as usize],
            s.tid_row,
            (chunk * 32) as i32,
        );

        // smem_A_temp source address
        ptx.add_s32(s.temp_addr[chunk as usize], a_temp_buf_base, a_cp_off);
        if chunk > 0 {
            ptx.add_s32_imm(
                s.temp_addr[chunk as usize],
                s.temp_addr[chunk as usize],
                (chunk * 2048) as i32,
            );
        }

        // smem_A destination address
        ptx.add_s32(s.dest_addr[chunk as usize], a_buf_base, a_cp_off);
        if chunk > 0 {
            ptx.add_s32_imm(
                s.dest_addr[chunk as usize],
                s.dest_addr[chunk as usize],
                (chunk * 2048) as i32,
            );
        }

        // Load 16 bytes from temp
        ptx.ld_shared_v4_b32(s.raw, s.temp_addr[chunk as usize], 0);

        // Load norm factor for this row
        ptx.shl_b32(s.factor_off, s.row_in_tile[chunk as usize], 2);
        ptx.add_s32(s.factor_addr, norm_factors_base, s.factor_off);
        ptx.ld_shared_b32(s.norm_factor_b32, s.factor_addr, 0);
        ptx.mov_b32_to_f32(s.norm_factor, s.norm_factor_b32);

        // Load norm weights from global memory
        ptx.add_s32(s.w_elem_off, k_offset, s.tid_col);
        ptx.shl_b32(s.w_byte_off, s.w_elem_off, 1); // * 2 bytes per f16
        let two_b32 = s.w_addr; // reuse
        ptx.mov_b32_imm(two_b32, 2);
        ptx.mad_wide_s32(s.w_addr64, s.w_elem_off, two_b32, wnorm_ptr);
        ptx.ld_global_v4_b32(s.wgt, s.w_addr64, 0);

        // Transform each pair and store directly
        for pair_idx in 0..4u32 {
            let inp = s.raw[pair_idx as usize];
            let w = s.wgt[pair_idx as usize];

            ptx.cvt_f32_f16(s.x_lo, inp);
            ptx.cvt_f32_f16(s.w_lo, w);
            ptx.mul_f32(s.t_lo, s.x_lo, s.norm_factor);
            ptx.mul_f32(s.y_lo, s.t_lo, s.w_lo);

            ptx.shr_u32(s.hi_input, inp, 16);
            ptx.cvt_f32_f16(s.x_hi, s.hi_input);
            ptx.shr_u32(s.hi_weight, w, 16);
            ptx.cvt_f32_f16(s.w_hi, s.hi_weight);
            ptx.mul_f32(s.t_hi, s.x_hi, s.norm_factor);
            ptx.mul_f32(s.y_hi, s.t_hi, s.w_hi);

            ptx.cvt_rn_f16_f32(s.h_lo, s.y_lo);
            ptx.cvt_rn_f16_f32(s.h_hi, s.y_hi);
            ptx.shl_b32(s.h_hi_shifted, s.h_hi, 16);
            ptx.or_b32(s.result[pair_idx as usize], s.h_lo, s.h_hi_shifted);
        }

        // Store to smem_A
        ptx.st_shared_v4_b32(s.dest_addr[chunk as usize], 0, s.result);
    }
}

/// Emit the GEMM K-loop using CpAsync-then-Transform pipeline for A.
///
/// CUTLASS MmaLayernormMainloopFusionMultistage approach:
///   Prologue: cp.async A_temp + B for tiles 0,1; transform both
///   K-loop:
///     1. ldmatrix + MMA from smem_A[read] + smem_B[read]
///     2. cp.async NEXT A_temp[write] + B[write]
///     3. commit + wait + barrier
///     4. Transform A_temp[write] -> A[write]
///     5. barrier
///     6. Advance, loop
///
/// Shared memory layout:
///   [0..8191]       Double-buffered A tiles (for ldmatrix)
///   [8192..16383]   Double-buffered B tiles
///   [16384..24575]  Double-buffered A_temp tiles (cp.async targets)
///   [24576..]       Norm factors + scratch + preloaded norm weights
pub fn emit_gemm_cpasync_transform_a(
    ptx: &mut PtxBuilder,
    c: &GemmConfig,
    setup: &GemmSetup,
    ga0: Reg,
    ga1: Reg,
    a_cp_off: Reg,
    gb0: Reg,
    gb1: Reg,
    b_cp_off: Reg,
    n_param: Reg,
    k_param: Reg,
    transform_ctx: &CpAsyncTransformCtx,
) -> AccumulatorMap {
    let smem_base = setup.smem_base;
    let tid = setup.tid;

    let a_tile_bytes = c.smem_a_bytes(); // 4096
    let buf_stride = a_tile_bytes; // 4096
    let b_start = (a_tile_bytes * c.num_stages) as i32; // 8192
    let a_temp_start = transform_ctx.a_temp_offset; // 16384

    let tid_x16 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x16, tid, 4);

    // ═══════════════════════════════════════════════════════════════
    // A_temp smem addresses (for cp.async — points to temp region)
    // ═══════════════════════════════════════════════════════════════
    let a_temp_st0 = ptx.regs.alloc_b32();
    ptx.add_s32(a_temp_st0, smem_base, a_cp_off);
    ptx.add_s32_imm(a_temp_st0, a_temp_st0, a_temp_start as i32);
    let a_temp_st1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_temp_st1, a_temp_st0, 2048);

    // B smem addresses
    let b_cp_base_smem = ptx.regs.alloc_b32();
    ptx.add_s32(b_cp_base_smem, smem_base, b_cp_off);
    let b_cp0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp0, b_cp_base_smem, b_start);
    let b_cp1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp1, b_cp_base_smem, b_start + 2048);
    ptx.blank();

    // K offset tracking (in elements, for norm weight lookup)
    let k_elem_offset = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(k_elem_offset, 0);

    // Allocate transform scratch registers once (reused across all transform calls)
    let transform_scratch = TransformScratch::alloc(ptx);

    // ═══════════════════════════════════════════════════════════════
    // Prologue: load tile 0 via cp.async into temp, then transform
    // ═══════════════════════════════════════════════════════════════
    ptx.comment("Prologue: cp.async tile 0 A_temp + B, then transform A");
    let p_tile0 = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_tile0, k_param, 0);
    let sz0 = ptx.regs.alloc_b32();
    ptx.selp_b32(sz0, 16, 0, p_tile0);

    // cp.async A -> temp buf 0, B -> buf 0
    ptx.cp_async_cg(a_temp_st0, 0, ga0, 0, sz0);
    ptx.cp_async_cg(a_temp_st1, 0, ga1, 0, sz0);
    ptx.cp_async_cg(b_cp0, 0, gb0, 0, sz0);
    ptx.cp_async_cg(b_cp1, 0, gb1, 0, sz0);
    ptx.cp_async_commit();
    ptx.cp_async_wait_group(0);
    ptx.bar_sync(0);

    // Transform tile 0: A_temp buf 0 -> A buf 0
    {
        let a_buf0 = ptx.regs.alloc_b32();
        ptx.mov_b32(a_buf0, smem_base); // buf 0 starts at smem_base + 0
        let a_temp_buf0 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(a_temp_buf0, smem_base, a_temp_start as i32);
        emit_smem_transform_dynamic(
            ptx,
            smem_base,
            a_cp_off,
            a_buf0,
            a_temp_buf0,
            transform_ctx.norm_factors_base,
            transform_ctx.wnorm_ptr,
            tid,
            k_elem_offset,
            &transform_scratch,
        );
    }
    ptx.bar_sync(0);
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

    ptx.add_s32_imm(k_elem_offset, k_elem_offset, c.bk as i32);
    ptx.blank();

    // ═══════════════════════════════════════════════════════════════
    // Prologue: load tile 1 via cp.async into temp, then transform
    // ═══════════════════════════════════════════════════════════════
    ptx.comment("Prologue: cp.async tile 1 A_temp + B, then transform A");
    let p_tile1 = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_tile1, k_param, c.bk as i32);
    let sz1 = ptx.regs.alloc_b32();
    ptx.selp_b32(sz1, 16, 0, p_tile1);

    // A_temp buf 1 addresses
    let a_temp_st0_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_temp_st0_b1, a_temp_st0, buf_stride as i32);
    let a_temp_st1_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_temp_st1_b1, a_temp_st1, buf_stride as i32);

    // cp.async A -> temp buf 1, B -> buf 1
    ptx.cp_async_cg(a_temp_st0_b1, 0, ga0_t1, 0, sz1);
    ptx.cp_async_cg(a_temp_st1_b1, 0, ga1_t1, 0, sz1);
    let b_cp0_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp0_b1, b_cp_base_smem, b_start + buf_stride as i32);
    let b_cp1_b1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_cp1_b1, b_cp_base_smem, b_start + buf_stride as i32 + 2048);
    ptx.cp_async_cg(b_cp0_b1, 0, gb0_t1, 0, sz1);
    ptx.cp_async_cg(b_cp1_b1, 0, gb1_t1, 0, sz1);
    ptx.cp_async_commit();
    ptx.cp_async_wait_group(0);
    ptx.bar_sync(0);

    // Transform tile 1: A_temp buf 1 -> A buf 1
    {
        let a_buf1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(a_buf1, smem_base, buf_stride as i32);
        let a_temp_buf1 = ptx.regs.alloc_b32();
        ptx.add_s32_imm(a_temp_buf1, smem_base, (a_temp_start + buf_stride) as i32);
        emit_smem_transform_dynamic(
            ptx,
            smem_base,
            a_cp_off,
            a_buf1,
            a_temp_buf1,
            transform_ctx.norm_factors_base,
            transform_ctx.wnorm_ptr,
            tid,
            k_elem_offset,
            &transform_scratch,
        );
    }
    ptx.bar_sync(0);
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

    ptx.add_s32_imm(k_elem_offset, k_elem_offset, c.bk as i32);
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

    // ═══════════════════════════════════════════════════════════════
    // K-LOOP (CpAsync-then-Transform schedule)
    //
    // Both A and B tiles are already transformed/loaded in the prologue.
    // The loop body:
    //   1. ldmatrix from smem_A[read] + smem_B[read] -> MMA
    //   2. cp.async NEXT A_temp[write] + B[write]
    //   3. commit, wait, barrier
    //   4. Transform NEXT A_temp[write] -> A[write]
    //   5. barrier
    //   6. Advance, loop
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

    // Compute read buffer base (A is already transformed, B is already loaded)
    let read_buf_off = ptx.regs.alloc_b32();
    ptx.shl_b32(read_buf_off, read_ctr, 12);
    let buf_base = ptx.regs.alloc_b32();
    ptx.add_s32(buf_base, smem_base, read_buf_off);
    ptx.blank();

    // ── Step 1: ldmatrix + MMA on CURRENT tile ──
    ptx.comment("Step 1: ldmatrix + MMA on CURRENT tile (already transformed)");

    // ldmatrix A
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

    // ldmatrix.trans B
    ptx.comment("ldmatrix.trans B");
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

    // MMA instructions
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

    // ── Step 2: cp.async NEXT tile's A_temp + B (predicated) ──
    ptx.comment("Step 2: cp.async NEXT A_temp + B (predicated)");

    // Toggle write buffer
    let write_next = ptx.regs.alloc_b32();
    ptx.add_s32_imm(write_next, write_ctr, 1);
    let p_reset_w = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_reset_w, write_next, 1);
    ptx.selp_b32_imm_reg(write_ctr, 0, write_next, p_reset_w);

    let write_buf_off = ptx.regs.alloc_b32();
    ptx.shl_b32(write_buf_off, write_ctr, 12);

    let cp_size = ptx.regs.alloc_b32();
    ptx.selp_b32(cp_size, 16, 0, p_load);

    // A_temp write destination
    let a_temp_w0 = ptx.regs.alloc_b32();
    ptx.add_s32(a_temp_w0, smem_base, a_cp_off);
    ptx.add_s32(a_temp_w0, a_temp_w0, write_buf_off);
    ptx.add_s32_imm(a_temp_w0, a_temp_w0, a_temp_start as i32);
    let a_temp_w1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(a_temp_w1, a_temp_w0, 2048);

    ptx.cp_async_cg(a_temp_w0, 0, ga0_loop, 0, cp_size);
    ptx.cp_async_cg(a_temp_w1, 0, ga1_loop, 0, cp_size);

    // B write destination
    let b_dst_base = ptx.regs.alloc_b32();
    ptx.add_s32(b_dst_base, smem_base, b_cp_off);
    ptx.add_s32(b_dst_base, b_dst_base, write_buf_off);
    let b_dst0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_dst0, b_dst_base, b_start);
    let b_dst1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b_dst1, b_dst_base, b_start + 2048);
    ptx.cp_async_cg(b_dst0, 0, gb0_loop, 0, cp_size);
    ptx.cp_async_cg(b_dst1, 0, gb1_loop, 0, cp_size);
    ptx.cp_async_commit();
    ptx.blank();

    // ── Step 3: Wait for cp.async + barrier ──
    ptx.comment("Step 3: Wait for cp.async, barrier");
    ptx.cp_async_wait_group(0);
    ptx.bar_sync(0);

    // ── Step 4: Transform NEXT A_temp -> A (in write buffer) ──
    ptx.comment("Step 4: Transform NEXT A_temp -> normalize -> A");
    {
        let write_a_base = ptx.regs.alloc_b32();
        ptx.add_s32(write_a_base, smem_base, write_buf_off);
        let write_a_temp_base = ptx.regs.alloc_b32();
        ptx.add_s32(write_a_temp_base, smem_base, write_buf_off);
        ptx.add_s32_imm(write_a_temp_base, write_a_temp_base, a_temp_start as i32);
        emit_smem_transform_dynamic(
            ptx,
            smem_base,
            a_cp_off,
            write_a_base,
            write_a_temp_base,
            transform_ctx.norm_factors_base,
            transform_ctx.wnorm_ptr,
            tid,
            k_elem_offset,
            &transform_scratch,
        );
    }
    ptx.blank();

    // ── Step 5: Barrier (ensure transform writes visible) ──
    ptx.comment("Step 5: Barrier (ensure transform writes visible)");
    ptx.bar_sync(0);

    // ── Step 6: Advance loop state ──
    ptx.comment("Advance loop state");

    // Advance B global pointers
    let gb0_next = ptx.regs.alloc_b64();
    ptx.add_s64(gb0_next, gb0_loop, b_stride_bytes);
    let gb1_next = ptx.regs.alloc_b64();
    ptx.add_s64(gb1_next, gb1_loop, b_stride_bytes);

    ptx.add_s32_imm(k_counter, k_counter, c.bk as i32);
    ptx.add_s64_imm(ga0_loop, ga0_loop, (c.bk * 2) as i64);
    ptx.add_s64_imm(ga1_loop, ga1_loop, (c.bk * 2) as i64);
    ptx.mov_b64(gb0_loop, gb0_next);
    ptx.mov_b64(gb1_loop, gb1_next);

    ptx.add_s32_imm(k_elem_offset, k_elem_offset, c.bk as i32);

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
    let group8 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(group8, setup.group, 8);

    // Warp layout decomposition.
    // For 1×4 (warps_n=4): warp_m = 0, warp_n = warp_id
    // For 2×2 (warps_n=2): warp_m = warp_id/2, warp_n = warp_id%2
    let warps_n = c.warps_n();
    let warps_n_shift = warps_n.trailing_zeros();

    let warp_m = ptx.regs.alloc_b32();
    ptx.shr_u32(warp_m, setup.warp_id, warps_n_shift);
    let warp_n = ptx.regs.alloc_b32();
    ptx.and_b32(warp_n, setup.warp_id, warps_n - 1);

    // Column base: block_col + warp_n * WN + tg*2
    let warp_col_base = ptx.regs.alloc_b32();
    let warp_n_off = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_n_off, warp_n, c.wn.trailing_zeros());
    ptx.add_s32(warp_col_base, setup.block_col, warp_n_off);

    // Row base: block_row + warp_m * MMA_M + group
    //
    // For the MMA m16n8k16 accumulator layout:
    //   d0 = C[group_id, tg*2]         (group_id = lane / 4, 0..7)
    //   d1 = C[group_id, tg*2+1]
    //   d2 = C[group_id+8, tg*2]       (+8 rows in the m16 tile)
    //   d3 = C[group_id+8, tg*2+1]
    //
    // warp_m * MMA_M gives the base row offset within the warp's M-region.
    // Each rm adds MMA_M * 2 = 32 rows (one m16 tile covers rows [base..base+15]).
    let warp_m_off = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_m_off, warp_m, c.mma_m.trailing_zeros());

    let row_a0 = ptx.regs.alloc_b32();
    ptx.add_s32(row_a0, setup.block_row, warp_m_off);
    ptx.add_s32(row_a0, row_a0, setup.group);
    let row_b0 = ptx.regs.alloc_b32();
    ptx.add_s32(row_b0, setup.block_row, warp_m_off);
    ptx.add_s32(row_b0, row_b0, group8);

    let row_a0_x_n = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(row_a0_x_n, row_a0, n_param);
    let row_b0_x_n = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(row_b0_x_n, row_b0, n_param);

    // Hoist constant "4" register out of the inner loops (constant folding).
    let four = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(four, 4);

    // Pre-allocate reusable registers for store address computation.
    // These are dead-after-use and can be reused across (rm, rn) iterations.
    let col_base = ptx.regs.alloc_b32();
    let col0 = ptx.regs.alloc_b32();
    let idx0 = ptx.regs.alloc_b32();
    let idx2 = ptx.regs.alloc_b32();
    // Reusable b64 address registers: one for row_a, one for row_b.
    // We use st.global.v2.b32 to store (d0,d1) and (d2,d3) together,
    // since they are at adjacent columns (tg*2 and tg*2+1) in the same row.
    let addr_a = ptx.regs.alloc_b64();
    let addr_b = ptx.regs.alloc_b64();
    // Row stride in bytes for advancing between rm groups: N * 4 bytes (f32 output)
    let n_stride = ptx.regs.alloc_b64();
    ptx.mul_wide_u32(n_stride, n_param, four);

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
            if rn > 0 {
                ptx.add_s32_imm(col_base, warp_col_base, (rn * c.mma_n) as i32);
            } else {
                ptx.mov_b32(col_base, warp_col_base);
            }
            ptx.add_s32(col0, col_base, tg2);

            // row_a address: output_ptr + (ra_x_n + col0) * 4
            ptx.add_s32(idx0, ra_x_n, col0);
            ptx.mad_wide_s32(addr_a, idx0, four, output_ptr);
            // Store d0,d1 as v2 (adjacent columns tg*2 and tg*2+1)
            ptx.st_global_v2_b32(addr_a, 0, acc.regs[ai][0], acc.regs[ai][1]);

            // row_b address: output_ptr + (rb_x_n + col0) * 4
            ptx.add_s32(idx2, rb_x_n, col0);
            ptx.mad_wide_s32(addr_b, idx2, four, output_ptr);
            // Store d2,d3 as v2 (adjacent columns tg*2 and tg*2+1, row+8)
            ptx.st_global_v2_b32(addr_b, 0, acc.regs[ai][2], acc.regs[ai][3]);
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
// Config-aware address computation for the pipeline path
// ═══════════════════════════════════════════════════════════════════════════

/// Emit cp.async swizzle address computation, config-aware.
/// For BN=128, the B swizzle mask differs from BN=64.
pub fn emit_cpasync_swizzle_cfg(ptx: &mut PtxBuilder, c: &GemmConfig, tid: Reg) -> (Reg, Reg) {
    ptx.comment("cp.async swizzle addresses (config-aware)");
    let tid_x16 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x16, tid, 4);
    let cp_base = ptx.regs.alloc_b32();
    ptx.and_b32(cp_base, tid_x16, 2032);

    // A swizzle: (tid & (BK-8)) << 1, masked to 48 for BK=32
    // For BK=32: a_swiz = (tid << 1) & 48 = ((tid & 24) << 1)
    // This is equivalent to: tid & (BK-8) gives the 2-bit K-group index, shift left for byte offset
    let tid_x2 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x2, tid, 1);
    let a_swiz = ptx.regs.alloc_b32();
    ptx.and_b32(a_swiz, tid_x2, 48);
    let a_cp_off = ptx.regs.alloc_b32();
    ptx.xor_b32(a_cp_off, cp_base, a_swiz);

    // B swizzle depends on BN:
    // For BN=64:  b_swiz = (tid << 1) & 112   (threads_per_row=8, 3-bit row index)
    // For BN=128: b_swiz = tid & 112           (threads_per_row=16, different mapping)
    let b_threads_per_row = c.bn / 8;
    let b_swiz = ptx.regs.alloc_b32();
    if b_threads_per_row <= 8 {
        // BN <= 64: use (tid << 1) & 112
        ptx.and_b32(b_swiz, tid_x2, 112);
    } else {
        // BN >= 128: use tid & 112 directly
        ptx.and_b32(b_swiz, tid, 112);
    }
    let b_cp_off = ptx.regs.alloc_b32();
    ptx.xor_b32(b_cp_off, cp_base, b_swiz);
    ptx.blank();

    (a_cp_off, b_cp_off)
}

/// Emit A global address computation, returning one pointer per cp.async chunk.
///
/// For BM=64: returns 2 pointers (rows 0-31, 32-63)
/// For BM=128: returns 4 pointers (rows 0-31, 32-63, 64-95, 96-127)
/// For BK=64: returns 4 pointers (rows 0-31/K0-31, rows 32-63/K0-31, rows 0-31/K32-63, rows 32-63/K32-63)
pub fn emit_a_global_addrs_cfg(
    ptx: &mut PtxBuilder,
    c: &GemmConfig,
    block_row: Reg,
    k_param: Reg,
    a_ptr: Reg,
    tid: Reg,
) -> Vec<Reg> {
    ptx.comment(&format!(
        "Global addresses for A ({}x{}, {} chunks)",
        c.bm,
        c.bk,
        c.cp_chunks_a()
    ));
    let a_tid_and3 = ptx.regs.alloc_b32();
    ptx.and_b32(a_tid_and3, tid, 3);
    let a_tid_col = ptx.regs.alloc_b32();
    ptx.shl_b32(a_tid_col, a_tid_and3, 3);
    let a_tid_row = ptx.regs.alloc_b32();
    ptx.shr_u32(a_tid_row, tid, 2);

    let row_chunks = c.bm / 32; // 2 for BM=64, 4 for BM=128
    let kcol_chunks = c.bk / 32; // 1 for BK=32, 2 for BK=64

    // First chunk: grow = block_row + a_tid_row, gidx = grow * K + a_tid_col
    let grow_base = ptx.regs.alloc_b32();
    ptx.add_s32(grow_base, block_row, a_tid_row);

    // Compute row stride for A: 32 rows * K * 2 bytes
    let a_row_stride = ptx.regs.alloc_b64();
    {
        let k_x32 = ptx.regs.alloc_b32();
        ptx.shl_b32(k_x32, k_param, 5); // K * 32
        let two_r = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(two_r, 2);
        ptx.mul_wide_s32(a_row_stride, k_x32, two_r); // K * 32 * 2 bytes
    }

    let mut ga_all = Vec::new();
    for rc in 0..row_chunks {
        let grow = ptx.regs.alloc_b32();
        if rc == 0 {
            ptx.mov_b32(grow, grow_base);
        } else {
            ptx.add_s32_imm(grow, grow_base, (rc * 32) as i32);
        }

        let gidx = ptx.regs.alloc_b32();
        ptx.mul_lo_s32(gidx, grow, k_param);
        ptx.add_s32(gidx, gidx, a_tid_col);
        let ga = ptx.regs.alloc_b64();
        ptx.mad_wide_s32_imm(ga, gidx, 2, a_ptr);
        ga_all.push(ga);

        // Additional K-column chunks for this row group
        for kc in 1..kcol_chunks {
            let ga_k = ptx.regs.alloc_b64();
            ptx.add_s64_imm(ga_k, ga, (kc * 64) as i64);
            ga_all.push(ga_k);
        }
    }
    ptx.blank();

    assert_eq!(ga_all.len(), c.cp_chunks_a() as usize);
    ga_all
}

/// Emit B global address computation, returning one pointer per cp.async chunk.
///
/// For BN=64: returns 2 pointers (B rows 0-15, 16-31)
/// For BN=128: returns 4 pointers (B rows 0-7, 8-15, 16-23, 24-31)
pub fn emit_b_global_addrs_cfg(
    ptx: &mut PtxBuilder,
    c: &GemmConfig,
    block_col: Reg,
    n_param: Reg,
    b_ptr: Reg,
    tid: Reg,
) -> Vec<Reg> {
    let threads = c.threads();
    let b_threads_per_row = c.bn / 8; // 8 for BN=64, 16 for BN=128
    let b_rows_per_chunk = threads / b_threads_per_row; // 16 for BN=64, 8 for BN=128
    let cp_chunks = c.cp_chunks_b();

    ptx.comment(&format!(
        "Global addresses for B ({} rows/chunk, {} chunks)",
        b_rows_per_chunk, cp_chunks
    ));

    let b_tid_col_idx = ptx.regs.alloc_b32();
    ptx.and_b32(b_tid_col_idx, tid, b_threads_per_row - 1);
    let b_tid_col = ptx.regs.alloc_b32();
    ptx.shl_b32(b_tid_col, b_tid_col_idx, 3);
    let b_tid_row = ptx.regs.alloc_b32();
    ptx.shr_u32(b_tid_row, tid, b_threads_per_row.trailing_zeros());

    let gcol_b = ptx.regs.alloc_b32();
    ptx.add_s32(gcol_b, block_col, b_tid_col);

    let mut gb_all = Vec::new();
    for chunk in 0..cp_chunks {
        let brow = ptx.regs.alloc_b32();
        if chunk == 0 {
            ptx.add_s32_imm(brow, b_tid_row, 0);
        } else {
            ptx.add_s32_imm(brow, b_tid_row, (chunk * b_rows_per_chunk) as i32);
        }

        let gidx = ptx.regs.alloc_b32();
        ptx.mul_lo_s32(gidx, brow, n_param);
        ptx.add_s32(gidx, gidx, gcol_b);
        let gb = ptx.regs.alloc_b64();
        ptx.mad_wide_s32_imm(gb, gidx, 2, b_ptr);
        gb_all.push(gb);
    }
    ptx.blank();

    gb_all
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
        None, // no fragment transform — standalone GEMM
    );

    // Phase 14: Store C
    emit_store_c(&mut ptx, c, &acc, &setup, c_ptr, n_param);
    ptx.blank();
    ptx.ret();

    ptx.finalize("triton_style_gemm", &gemm_params())
}

pub(crate) fn gemm_params() -> Vec<(&'static str, &'static str)> {
    vec![
        (".u64 .ptr .global .align 16", "param_A"),
        (".u64 .ptr .global .align 16", "param_B"),
        (".u64 .ptr .global .align 16", "param_C"),
        (".u32", "param_M"),
        (".u32", "param_N"),
        (".u32", "param_K"),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// build_gemm_pipeline — standalone GEMM using the MainloopPipeline
//
// This is the THIN WRAPPER. The pipeline handles all scheduling.
// ═══════════════════════════════════════════════════════════════════════════

/// Build a complete GEMM kernel PTX string using the MainloopPipeline.
/// This produces identical behavior to build_gemm() but uses the
/// pluggable atom architecture. Supports any config (64×64, 128×128, etc).
pub fn build_gemm_pipeline(config: &GemmConfig) -> String {
    use crate::atoms::{CpAsyncCopy, EpilogueAtom, IdentityEpilogue, IdentityTransform, Mma16816};
    use crate::pipeline::MainloopPipeline;

    let mut ptx = PtxBuilder::new(config.clone());
    let c = &ptx.config.clone();

    ptx.comment("GEMM kernel generated by MainloopPipeline");
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

    // Phase 3: cp.async swizzle addresses (config-aware)
    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle_cfg(&mut ptx, c, setup.tid);

    // Phase 4: Global addresses (config-aware, returns Vec)
    let ga_chunks =
        emit_a_global_addrs_cfg(&mut ptx, c, setup.block_row, k_param, a_ptr, setup.tid);
    let gb_chunks =
        emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, b_ptr, setup.tid);

    // Phase 5: Pipeline with Identity atoms (standalone GEMM)
    let pipeline = MainloopPipeline::new(c.num_stages);
    let result = pipeline.emit(
        &mut ptx,
        c,
        &setup,
        &CpAsyncCopy,       // copy_a
        &CpAsyncCopy,       // copy_b
        &IdentityTransform, // no transform
        &Mma16816,          // tensor core MMA
        ga_chunks,
        a_cp_off,
        gb_chunks,
        b_cp_off,
        n_param,
        k_param,
        None, // no extra advance
    );

    // Phase 6: Identity epilogue + store
    let mut acc = result.acc;
    IdentityEpilogue.emit_epilogue(&mut ptx, &mut acc);
    emit_store_c(&mut ptx, c, &acc, &setup, c_ptr, n_param);
    ptx.blank();
    ptx.ret();

    ptx.finalize("triton_style_gemm", &gemm_params())
}
