use crate::gemm::AccumulatorMap;
use crate::{PtxBuilder, Reg};

// ═══════════════════════════════════════════════════════════════════════════
// Atom traits — the pluggable components of the MainloopPipeline
// ═══════════════════════════════════════════════════════════════════════════

/// How tiles move from global memory to shared memory.
/// SM89: cp.async.cg (hardware DMA, 16 bytes/thread)
pub trait CopyAtom {
    /// Emit async copy of one chunk (2048 bytes) from global to shared memory.
    /// `smem_dst`: shared memory destination address register
    /// `glob_src`: global memory source address register
    /// `pred_size`: register containing 16 (copy) or 0 (skip)
    fn emit_async_copy(&self, ptx: &mut PtxBuilder, smem_dst: Reg, glob_src: Reg, pred_size: Reg);

    /// Emit cp.async.commit_group after tile loads.
    fn emit_commit(&self, ptx: &mut PtxBuilder);

    /// Number of cp.async groups issued per tile load.
    fn async_groups_per_tile(&self) -> u32;
}

/// How A fragments are transformed between ldmatrix and MMA.
/// This is WHERE FUSION HAPPENS for pre-GEMM operations.
///
/// CRITICAL: operates on PACKED f16x2 registers.
/// Must use f16x2 arithmetic (mul.rn.f16x2, fma.rn.f16x2).
/// NEVER unpack to f32 and repack.
pub trait TransformAtom {
    /// One-time setup before the K-loop (e.g., load norm factors into registers).
    fn emit_prologue(&self, _ptx: &mut PtxBuilder) {}

    /// Per-K-iteration setup (e.g., load gamma from smem for this K chunk).
    fn emit_k_setup(&self, _ptx: &mut PtxBuilder, _ki: u32) {}

    /// Transform 4 b32 registers (8 packed f16) in-place. Called per ldmatrix.
    /// `ki` = which K-iteration within BK (0 or 1 for BK=32)
    /// `rm` = which M-tile (0..REG_M-1)
    fn emit_transform(&self, ptx: &mut PtxBuilder, frag: &mut [Reg; 4], ki: u32, rm: u32);
}

/// How tensor cores consume fragments.
/// SM89: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
pub trait MmaAtom {
    fn emit_mma(&self, ptx: &mut PtxBuilder, a: [Reg; 4], b: [Reg; 2], acc: &mut [Reg; 4]);
}

/// How accumulators are post-processed before store.
/// This is WHERE FUSION HAPPENS for post-GEMM operations.
pub trait EpilogueAtom {
    fn emit_epilogue(&self, ptx: &mut PtxBuilder, acc: &mut AccumulatorMap);
}

// ═══════════════════════════════════════════════════════════════════════════
// CpAsyncCopy — wraps cp.async.cg.shared.global (SM89 path)
// ═══════════════════════════════════════════════════════════════════════════

pub struct CpAsyncCopy;

impl CopyAtom for CpAsyncCopy {
    fn emit_async_copy(&self, ptx: &mut PtxBuilder, smem_dst: Reg, glob_src: Reg, pred_size: Reg) {
        ptx.cp_async_cg(smem_dst, 0, glob_src, 0, pred_size);
    }

    fn emit_commit(&self, ptx: &mut PtxBuilder) {
        ptx.cp_async_commit();
    }

    fn async_groups_per_tile(&self) -> u32 {
        1
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IdentityTransform — no-op (standalone GEMM uses this)
// ═══════════════════════════════════════════════════════════════════════════

pub struct IdentityTransform;

impl TransformAtom for IdentityTransform {
    fn emit_transform(&self, _ptx: &mut PtxBuilder, _frag: &mut [Reg; 4], _ki: u32, _rm: u32) {
        // No-op: zero additional PTX instructions
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mma16816 — wraps mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// ═══════════════════════════════════════════════════════════════════════════

pub struct Mma16816;

impl MmaAtom for Mma16816 {
    fn emit_mma(&self, ptx: &mut PtxBuilder, a: [Reg; 4], b: [Reg; 2], acc: &mut [Reg; 4]) {
        ptx.mma_m16n8k16(*acc, a, b, *acc);
        // mma_m16n8k16 writes to `d` which is the first arg. We update acc in place.
        // Note: PtxBuilder::mma_m16n8k16(d, a, b, c) emits d = mma(a, b, c).
        // Since we passed *acc as both d and c, acc is updated.
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IdentityEpilogue — just passes through f32 accumulators
// ═══════════════════════════════════════════════════════════════════════════

pub struct IdentityEpilogue;

impl EpilogueAtom for IdentityEpilogue {
    fn emit_epilogue(&self, _ptx: &mut PtxBuilder, _acc: &mut AccumulatorMap) {
        // No-op: accumulators are stored as-is
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SiLuEpilogue — x * sigmoid(x) on accumulators (6 ALU ops per element)
// ═══════════════════════════════════════════════════════════════════════════

pub struct SiLuEpilogue;

impl EpilogueAtom for SiLuEpilogue {
    fn emit_epilogue(&self, ptx: &mut PtxBuilder, acc: &mut AccumulatorMap) {
        crate::silu::emit_silu_phase(ptx, acc);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GeluEpilogue — x * sigmoid(1.702 * x) on accumulators (7 ALU ops per element)
//
// Fast GELU approximation used by many frameworks (GELU_FAST / GELU_PYTORCH_TANH):
//   GELU(x) ≈ x * sigmoid(1.702 * x)
// ═══════════════════════════════════════════════════════════════════════════

pub struct GeluEpilogue;

impl EpilogueAtom for GeluEpilogue {
    fn emit_epilogue(&self, ptx: &mut PtxBuilder, acc: &mut AccumulatorMap) {
        crate::gelu::emit_gelu_phase(ptx, acc);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RmsNormTransform — 2x mul.rn.f16x2 per b32 register
//
// Norm factor in REGISTERS (loaded once in prologue, kept entire K-loop).
// Gamma from smem (preloaded, looked up per K-iter).
// ═══════════════════════════════════════════════════════════════════════════

/// Scratch registers for the RmsNorm transform. Pre-allocated ONCE, reused every call.
///
/// Register-pressure-optimized design:
/// - Norm factors loaded once in prologue (2*REG_M regs, constant across K-loop)
/// - Gamma loaded once per K-iteration for ALL ki values (batch load optimization)
/// - Multi-purpose scratch regs shared across computation phases
///   instead of separate named regs for each intermediate value
pub struct RmsNormAtomScratch {
    // Norm factor registers (computed in prologue, kept for entire K-loop)
    // nf_packed[rm][0] = norm_factor for group_id rows, nf_packed[rm][1] = group_id+8 rows
    pub nf_packed: Vec<[Reg; 2]>, // [REG_M] × 2 regs each
    // Gamma values — one pair per ki, all loaded at once in k_setup(ki=0).
    // gamma_all[ki] = [gamma_packed, gamma_packed2] for that ki value.
    // This matches the hand-written PTX which loads all gamma values before
    // any ldmatrix, enabling better instruction scheduling.
    pub gamma_all: Vec<[Reg; 2]>, // [k_warp_iters] × 2 regs each
    // Multi-purpose scratch (reused across all computation phases)
    pub tmp0: Reg,   // b32: address/offset computation
    pub nf_f16: Reg, // b32: f32→f16 conversion temp
    // Precomputed
    pub tg2: Reg, // b32: tg * 2
}

/// RmsNorm transform atom.
///
/// Multiplies each f16 element by norm_factor (per-row, in REGISTERS) * gamma (per-k-column, from smem).
/// Uses packed f16x2 ops only — 2x mul.rn.f16x2 per b32 register.
pub struct RmsNormAtom {
    pub norm_factors_smem: Reg, // smem base of norm factors array
    pub gamma_smem: Reg,        // smem base of preloaded gamma weights
    pub group_id: Reg,          // lane group within warp (lane >> 2)
    pub tg: Reg,                // thread group (lane & 3)
    pub k_counter: Reg,         // k-loop counter register
    pub reg_m: u32,             // number of M register tiles (from config)
    pub k_warp_iters: u32,      // BK/MMA_K (2 for BK=32, 4 for BK=64)
    /// For 128×128 with 2×2 warp layout, each warp handles 64 rows.
    /// The warp_m_offset (0 or 64) needs to be added to norm factor lookups.
    pub warp_m_offset: Reg, // b32: warp_m * WM (e.g., 0 or 64)
    pub scratch: RmsNormAtomScratch,
}

impl RmsNormAtom {
    /// Create a new RmsNormAtom, allocating scratch registers.
    ///
    /// `reg_m`: number of M register tiles (config.reg_m())
    /// `k_warp_iters`: BK / MMA_K (number of ki values per tile)
    /// `warp_m_offset`: register containing warp_m * WM (0 for 64×64 with 1 warp in M)
    pub fn new(
        ptx: &mut PtxBuilder,
        norm_factors_smem: Reg,
        gamma_smem: Reg,
        group_id: Reg,
        tg: Reg,
        k_counter: Reg,
        reg_m: u32,
        k_warp_iters: u32,
        warp_m_offset: Reg,
    ) -> Self {
        let mut nf_packed = Vec::with_capacity(reg_m as usize);
        for _ in 0..reg_m {
            nf_packed.push([ptx.regs.alloc_b32(), ptx.regs.alloc_b32()]);
        }
        // Allocate gamma registers for ALL ki values at once.
        // This enables batch-loading all gamma values before any ldmatrix,
        // matching the hand-written PTX's scheduling pattern.
        let mut gamma_all = Vec::with_capacity(k_warp_iters as usize);
        for _ in 0..k_warp_iters {
            gamma_all.push([ptx.regs.alloc_b32(), ptx.regs.alloc_b32()]);
        }
        let scratch = RmsNormAtomScratch {
            nf_packed,
            gamma_all,
            tmp0: ptx.regs.alloc_b32(),
            nf_f16: ptx.regs.alloc_b32(),
            tg2: ptx.regs.alloc_b32(),
        };
        // Precompute tg * 2
        ptx.shl_b32(scratch.tg2, tg, 1);
        Self {
            norm_factors_smem,
            gamma_smem,
            group_id,
            tg,
            k_counter,
            reg_m,
            k_warp_iters,
            warp_m_offset,
            scratch,
        }
    }

    /// Load norm factors from smem and pack as f16x2 into registers.
    /// Called once before the K-loop — norm factors stay in registers for the entire loop.
    /// Uses tmp0 as multi-purpose scratch for address computation.
    fn load_norm_factors(&self, ptx: &mut PtxBuilder, rm: u32) {
        let s = &self.scratch;
        let [nf_packed, nf_packed8] = s.nf_packed[rm as usize];

        // Row within block = warp_m_offset + rm*16 + group_id
        // norm_factor address = norm_factors_smem + row * 4
        ptx.add_s32(s.tmp0, self.warp_m_offset, self.group_id);
        if rm > 0 {
            ptx.add_s32_imm(s.tmp0, s.tmp0, (rm * 16) as i32);
        }
        ptx.shl_b32(s.tmp0, s.tmp0, 2); // row * 4 bytes
        ptx.add_s32(s.tmp0, self.norm_factors_smem, s.tmp0);
        ptx.ld_shared_b32(nf_packed, s.tmp0, 0); // load f32 directly into dest
        ptx.pack_f16x2_from_f32(nf_packed, nf_packed, s.nf_f16);

        // norm_factor for group_id + 8 row
        ptx.ld_shared_b32(nf_packed8, s.tmp0, 8 * 4); // use immediate offset
        ptx.pack_f16x2_from_f32(nf_packed8, nf_packed8, s.nf_f16);
    }
}

impl TransformAtom for RmsNormAtom {
    fn emit_prologue(&self, ptx: &mut PtxBuilder) {
        ptx.comment(&format!(
            "RmsNormAtom prologue: load norm factors into REGISTERS (REG_M={})",
            self.reg_m
        ));
        for rm in 0..self.reg_m {
            self.load_norm_factors(ptx, rm);
        }
    }

    fn emit_k_setup(&self, ptx: &mut PtxBuilder, ki: u32) {
        let s = &self.scratch;
        // Batch gamma load optimization: load ALL gamma values at once when ki=0.
        // For ki>0, the values are already in registers. This matches the hand-written
        // PTX pattern and eliminates redundant address computation per ki.
        if ki == 0 {
            ptx.comment(&format!(
                "RmsNormAtom k_setup: batch load gamma for all {} ki values",
                self.k_warp_iters
            ));
            // Compute base gamma address: gamma_smem + (k_counter + tg*2) * 2
            ptx.add_s32(s.tmp0, self.k_counter, s.tg2);
            ptx.shl_b32(s.tmp0, s.tmp0, 1); // byte offset
            ptx.add_s32(s.tmp0, self.gamma_smem, s.tmp0); // smem address
            // Load gamma for all ki values using immediate offsets from the base address.
            // Each ki is 16 elements apart = 32 bytes.
            // Within each ki: gamma_packed at +0, gamma_packed2 at +16.
            for k in 0..self.k_warp_iters {
                let base_off = (k * 16 * 2) as i32; // ki * 16 elements * 2 bytes
                ptx.ld_shared_b32(s.gamma_all[k as usize][0], s.tmp0, base_off);
                ptx.ld_shared_b32(s.gamma_all[k as usize][1], s.tmp0, base_off + 8 * 2);
            }
        }
        // For ki > 0: gamma values already loaded, no-op
    }

    fn emit_transform(&self, ptx: &mut PtxBuilder, frag: &mut [Reg; 4], ki: u32, rm: u32) {
        let s = &self.scratch;
        let [nf_packed, nf_packed8] = s.nf_packed[rm as usize];
        let [gamma_packed, gamma_packed2] = s.gamma_all[ki as usize];
        // 2 packed f16x2 multiplies per register — norm_factor × gamma
        ptx.mul_rn_f16x2(frag[0], frag[0], nf_packed);
        ptx.mul_rn_f16x2(frag[0], frag[0], gamma_packed);
        ptx.mul_rn_f16x2(frag[1], frag[1], nf_packed);
        ptx.mul_rn_f16x2(frag[1], frag[1], gamma_packed2);
        ptx.mul_rn_f16x2(frag[2], frag[2], nf_packed8);
        ptx.mul_rn_f16x2(frag[2], frag[2], gamma_packed);
        ptx.mul_rn_f16x2(frag[3], frag[3], nf_packed8);
        ptx.mul_rn_f16x2(frag[3], frag[3], gamma_packed2);
    }
}
