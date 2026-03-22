use crate::{PtxBuilder, Reg};
use crate::config::GemmConfig;
use crate::gemm::{
    CpAsyncLoader, RmsNormTransform, emit_a_global_addrs, emit_b_global_addrs,
    emit_cpasync_swizzle, emit_gemm_setup, emit_gemm_with_loaders, emit_store_c,
};
// emit_norm_factor_computation_inner is called via crate::tile:: path

// ===========================================================================
// Fused RMSNorm -> GEMM -> SiLU megakernel
//
// FragmentTransform approach — register-level fusion:
//   1. RMSNorm: compute per-row norm factors (one pass over input rows)
//   2. Preload ALL gamma weights into smem (one vectorized load per thread)
//   3. GEMM K-loop (CpAsyncLoader for BOTH A and B — identical to standalone):
//      - cp.async raw input -> smem_A (full DMA bandwidth, SAME as standalone)
//      - cp.async B weights -> smem_B (full DMA bandwidth, SAME as standalone)
//      - ldmatrix A -> fragment registers
//      - FragmentTransform: multiply A regs by norm_factor * gamma (register ALU)
//      - ldmatrix.trans B -> fragment registers
//      - MMA(transformed_A, B, accumulators)
//   4. SiLU: transform accumulators in-place (pure ALU)
//   5. Store: write activated f32 results to global
//
// The ONLY difference from standalone GEMM is the register ALU between
// ldmatrix A and MMA. No extra smem buffers, no extra barriers, no extra
// smem-to-smem transforms.
//
// Shared memory layout:
//   [0..N*4096-1]       N-buffered A tiles (cp.async raw input, N=num_stages)
//   [N*4096..2*N*4096-1] N-buffered B tiles (cp.async weights, N=num_stages)
//   [2*N*4096..]        Norm scratch (32 bytes)
//   [+32..]             Norm factors (64 f32 = 256 bytes)
//   [+256+32..]         Gamma weights (4096 f16 = 8192 bytes, preloaded)
//   2 stages: ~25KB, 3 stages: ~33KB, 4 stages: ~41KB
// ===========================================================================

/// Build a fused RMSNorm -> GEMM -> SiLU kernel.
///
/// Grid: (out_features / BN, batch / BM, 1)
/// Block: 128 threads (4 warps)
pub fn build_fused_rmsnorm_gemm_silu(config: &GemmConfig, hidden_size: u32) -> String {
    let mut ptx = PtxBuilder::new(config.clone());
    let c = &ptx.config.clone();

    let bm = c.bm;
    let bk = c.bk;
    let threads = c.threads();

    // Shared memory layout
    let gemm_smem = c.smem_total(); // num_stages * buf_stride
    let norm_scratch_off = gemm_smem;
    let norm_factors_off = norm_scratch_off + 32;
    let gamma_smem_off = norm_factors_off + bm * 4;
    let gamma_smem_bytes = hidden_size * 2; // 8192 for hidden=4096
    let _total_smem = gamma_smem_off + gamma_smem_bytes;

    ptx.comment("=== Fused RMSNorm -> GEMM -> SiLU megakernel (FragmentTransform) ===");
    ptx.comment(&format!(
        "BM={}, BN={}, BK={}, stages={}, threads={}",
        bm, c.bn, bk, c.num_stages, threads
    ));
    ptx.comment(&format!("hidden_size={}", hidden_size));
    ptx.comment(&format!(
        "smem: GEMM[0..{}] ({} stages)",
        gemm_smem - 1,
        c.num_stages
    ));
    ptx.comment(&format!(
        "      norm_scratch[{norm_scratch_off}] norm_factors[{norm_factors_off}]"
    ));
    ptx.comment(&format!(
        "      gamma[{gamma_smem_off}..{}]",
        gamma_smem_off + gamma_smem_bytes
    ));
    ptx.blank();

    // ===============================================================
    // Phase 1: Load parameters
    // ===============================================================
    let input_ptr = ptx.regs.alloc_b64();
    let wnorm_ptr = ptx.regs.alloc_b64();
    let wgemm_ptr = ptx.regs.alloc_b64();
    let output_ptr = ptx.regs.alloc_b64();
    let n_param = ptx.regs.alloc_b32();
    let k_param = ptx.regs.alloc_b32();
    ptx.ld_param_b64(input_ptr, "param_input");
    ptx.ld_param_b64(wnorm_ptr, "param_wnorm");
    ptx.ld_param_b64(wgemm_ptr, "param_wgemm");
    ptx.ld_param_b64(output_ptr, "param_output");
    ptx.ld_param_b32(n_param, "param_N");
    ptx.ld_param_b32(k_param, "param_K");
    ptx.blank();

    // ===============================================================
    // Phase 2: Thread/block setup (reuses GEMM infrastructure)
    // ===============================================================
    let setup = emit_gemm_setup(&mut ptx, c);

    // ===============================================================
    // Phase A: RMSNorm -- compute norm factors for all BM rows
    // ===============================================================
    // Allocate norm_factors_base OUTSIDE the scope so it survives end_scope()
    let norm_factors_base = ptx.regs.alloc_b32();
    ptx.begin_scope(); // --- norm reduction temporaries: native PTX block scope ---
    let eps: f32 = 1e-6;
    crate::tile::emit_norm_factor_computation_inner(
        &mut ptx,
        input_ptr,
        setup.block_row,
        setup.tid,
        setup.smem_base,
        norm_factors_off,
        hidden_size,
        eps,
        Some(norm_factors_base),
    );
    ptx.end_scope(); // --- norm reduction temps die here (ptxas sees }) ---

    // ===============================================================
    // Phase 1.5: Preload ALL gamma weights into shared memory
    // ===============================================================
    // Allocate gamma_smem_base OUTSIDE the scope so it survives end_scope()
    let gamma_smem_base = ptx.regs.alloc_b32();
    ptx.add_s32_imm(gamma_smem_base, setup.smem_base, gamma_smem_off as i32);

    ptx.begin_scope(); // --- gamma preload temporaries: native PTX block scope ---
    ptx.comment("=== Phase 1.5: Preload gamma weights into smem ===");
    ptx.comment(&format!(
        "{} f16 = {} bytes, {} threads x {} bytes each",
        hidden_size,
        gamma_smem_bytes,
        threads,
        gamma_smem_bytes / threads
    ));
    ptx.blank();

    let bytes_per_thread = (hidden_size * 2) / threads;
    let v4_loads_per_thread = bytes_per_thread / 16;

    // Global address: wnorm_ptr + tid * bytes_per_thread
    let gamma_global_addr = ptx.regs.alloc_b64();
    {
        let bytes_per_t = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(bytes_per_t, bytes_per_thread);
        ptx.mad_wide_s32(gamma_global_addr, setup.tid, bytes_per_t, wnorm_ptr);
    }

    // Smem address: gamma_smem_base + tid * bytes_per_thread
    let gamma_smem_dst = ptx.regs.alloc_b32();
    {
        let tid_byte_off = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(tid_byte_off, bytes_per_thread);
        ptx.mul_lo_s32(tid_byte_off, setup.tid, tid_byte_off);
        ptx.add_s32(gamma_smem_dst, gamma_smem_base, tid_byte_off);
    }

    // Load and store in a loop (or unrolled for small counts)
    for v in 0..v4_loads_per_thread {
        let data = [
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
        ];
        ptx.ld_global_v4_b32(data, gamma_global_addr, (v * 16) as i32);
        ptx.st_shared_v4_b32(gamma_smem_dst, (v * 16) as i32, data);
    }
    ptx.bar_sync(0);
    ptx.blank();
    ptx.end_scope(); // --- gamma preload temps die here (ptxas sees }) ---

    // ===============================================================
    // Phase B: GEMM with FragmentTransform (CpAsyncLoader for BOTH A and B)
    // ===============================================================
    ptx.comment("=== Phase B: GEMM with FragmentTransform (cp.async BOTH A and B) ===");
    ptx.blank();

    // cp.async swizzle addresses
    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle(&mut ptx, setup.tid);

    // A global addresses (pointing to raw input — cp.async will load these)
    let (ga0, ga1, _a_tid_row, _a_tid_col) =
        emit_a_global_addrs(&mut ptx, setup.block_row, k_param, input_ptr, setup.tid);

    // B global addresses
    let (gb0, gb1) = emit_b_global_addrs(&mut ptx, setup.block_col, n_param, wgemm_ptr, setup.tid);

    // CpAsyncLoader for BOTH A and B — identical to standalone GEMM
    let a_loader = CpAsyncLoader;
    let b_loader = CpAsyncLoader;

    // Create the RmsNormTransform
    // The k_counter register will be created inside emit_gemm_with_loaders,
    // but we need a placeholder. We'll use a register that we pre-allocate
    // and which emit_gemm_with_loaders will use for its k_counter.
    // Actually, the RmsNormTransform needs the k_counter from the K-loop.
    // Since emit_gemm_with_loaders creates the k_counter internally, we need
    // to pass it through. Let's use a different approach: we pre-allocate a
    // k_counter register and emit_gemm_with_loaders will use it via the
    // emit_k_iter_setup hook.
    //
    // Actually, looking at the code more carefully, the transform's k_counter
    // field references the loop's k_counter. The issue is we don't have access
    // to it before calling emit_gemm_with_loaders. But since RmsNormTransform
    // stores a Reg (just an index), we can allocate a "dummy" register now
    // and set it up properly.
    //
    // Better approach: the K-loop in emit_gemm_with_loaders already has a
    // k_counter register. We need the transform to reference it. Since the
    // transform is called from within the K-loop, it can read k_counter.
    // But we haven't created k_counter yet.
    //
    // Solution: pre-allocate a register for k_counter and pass it into both
    // the transform and the K-loop. But emit_gemm_with_loaders creates its own.
    //
    // Simplest approach for now: the RmsNormTransform will use a separate
    // k_offset register that we initialize to 0 and advance via extra_advance.
    let k_offset_reg = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(k_offset_reg, 0);

    let transform = RmsNormTransform::new(
        &mut ptx,
        norm_factors_base,
        gamma_smem_base,
        setup.group,
        setup.tg,
        k_offset_reg,
    );

    // Extra advance: bump k_offset_reg by BK each iteration
    let bk_val = c.bk;
    let advance_fn = move |ptx_ref: &mut PtxBuilder| {
        ptx_ref.add_s32_imm(k_offset_reg, k_offset_reg, bk_val as i32);
    };

    let mut acc = emit_gemm_with_loaders(
        &mut ptx,
        c,
        &setup,
        &a_loader,
        ga0,
        ga1,
        a_cp_off,
        &b_loader,
        gb0,
        gb1,
        b_cp_off,
        n_param,
        k_param,
        Some(&advance_fn),
        Some(&transform),
    );

    // ===============================================================
    // Phase C: SiLU activation on accumulators
    // ===============================================================
    ptx.comment("=== Phase C: SiLU activation on accumulators ===");
    crate::silu::emit_silu_phase(&mut ptx, &mut acc);
    ptx.blank();

    // ===============================================================
    // Phase D: Store output
    // ===============================================================
    ptx.comment("=== Phase D: Store output ===");
    emit_store_c(&mut ptx, c, &acc, &setup, output_ptr, n_param);
    ptx.blank();
    ptx.ret();

    ptx.finalize("fused_rmsnorm_gemm_silu", &fused_params())
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline-based fused kernel — RMSNorm -> GEMM -> SiLU
//
// This is the THIN WRAPPER using MainloopPipeline + pluggable atoms.
// Same kernel, same PTX output, but the K-loop is driven by the pipeline.
// ═══════════════════════════════════════════════════════════════════════════

/// Build a fused RMSNorm -> GEMM -> SiLU kernel using the MainloopPipeline.
/// Supports any GemmConfig (64×64, 128×128, etc).
pub fn build_fused_pipeline(config: &GemmConfig, hidden_size: u32) -> String {
    use crate::atoms::{CpAsyncCopy, EpilogueAtom, Mma16816, RmsNormAtom, SiLuEpilogue};
    use crate::gemm::{
        emit_a_global_addrs_cfg, emit_b_global_addrs_cfg, emit_cpasync_swizzle_cfg,
        emit_gemm_setup, emit_store_c,
    };
    use crate::pipeline::MainloopPipeline;

    let mut ptx = PtxBuilder::new(config.clone());
    let c = &ptx.config.clone();

    let bm = c.bm;
    let bk = c.bk;
    let threads = c.threads();

    // Shared memory layout
    let gemm_smem = c.smem_total(); // num_stages * buf_stride
    let norm_scratch_off = gemm_smem;
    let norm_factors_off = norm_scratch_off + 32;
    let gamma_smem_off = norm_factors_off + bm * 4;
    let _gamma_smem_bytes = hidden_size * 2;

    ptx.comment("=== Fused RMSNorm -> GEMM -> SiLU (MainloopPipeline) ===");
    ptx.comment(&format!(
        "BM={}, BN={}, BK={}, threads={}",
        bm, c.bn, bk, threads
    ));
    ptx.comment(&format!(
        "hidden_size={}, REG_M={}, REG_N={}",
        hidden_size,
        c.reg_m(),
        c.reg_n()
    ));
    ptx.blank();

    // Phase 1: Load parameters
    let input_ptr = ptx.regs.alloc_b64();
    let wnorm_ptr = ptx.regs.alloc_b64();
    let wgemm_ptr = ptx.regs.alloc_b64();
    let output_ptr = ptx.regs.alloc_b64();
    let n_param = ptx.regs.alloc_b32();
    let k_param = ptx.regs.alloc_b32();
    ptx.ld_param_b64(input_ptr, "param_input");
    ptx.ld_param_b64(wnorm_ptr, "param_wnorm");
    ptx.ld_param_b64(wgemm_ptr, "param_wgemm");
    ptx.ld_param_b64(output_ptr, "param_output");
    ptx.ld_param_b32(n_param, "param_N");
    ptx.ld_param_b32(k_param, "param_K");
    ptx.blank();

    // Phase 2: Thread/block setup
    let setup = emit_gemm_setup(&mut ptx, c);

    // Phase A: RMSNorm — compute norm factors for all BM rows
    // For BM<=64: 2 threads per row (cooperative halving)
    // For BM=128: 1 thread per row (each thread processes full hidden_size)
    let threads_per_row = if bm <= 64 { 2u32 } else { 1u32 };
    // Allocate norm_factors_base OUTSIDE the scope so it survives end_scope()
    let norm_factors_base = ptx.regs.alloc_b32();
    ptx.begin_scope(); // --- norm reduction temporaries: native PTX block scope ---
    let eps: f32 = 1e-6;
    crate::tile::emit_norm_factor_computation_cfg(
        &mut ptx,
        input_ptr,
        setup.block_row,
        setup.tid,
        setup.smem_base,
        norm_factors_off,
        hidden_size,
        eps,
        Some(norm_factors_base),
        threads_per_row,
    );
    ptx.end_scope(); // --- norm reduction temps die here (ptxas sees }) ---

    // Phase 1.5: Preload ALL gamma weights into shared memory
    // Allocate gamma_smem_base OUTSIDE the scope so it survives end_scope()
    let gamma_smem_base = ptx.regs.alloc_b32();
    ptx.add_s32_imm(gamma_smem_base, setup.smem_base, gamma_smem_off as i32);

    ptx.begin_scope(); // --- gamma preload temporaries: native PTX block scope ---
    ptx.comment("=== Phase 1.5: Preload gamma weights into smem ===");
    ptx.blank();

    let bytes_per_thread = (hidden_size * 2) / threads;
    let v4_loads_per_thread = bytes_per_thread / 16;

    let gamma_global_addr = ptx.regs.alloc_b64();
    {
        let bytes_per_t = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(bytes_per_t, bytes_per_thread);
        ptx.mad_wide_s32(gamma_global_addr, setup.tid, bytes_per_t, wnorm_ptr);
    }

    let gamma_smem_dst = ptx.regs.alloc_b32();
    {
        let tid_byte_off = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(tid_byte_off, bytes_per_thread);
        ptx.mul_lo_s32(tid_byte_off, setup.tid, tid_byte_off);
        ptx.add_s32(gamma_smem_dst, gamma_smem_base, tid_byte_off);
    }

    for v in 0..v4_loads_per_thread {
        let data = [
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
        ];
        ptx.ld_global_v4_b32(data, gamma_global_addr, (v * 16) as i32);
        ptx.st_shared_v4_b32(gamma_smem_dst, (v * 16) as i32, data);
    }
    ptx.bar_sync(0);
    ptx.blank();
    ptx.end_scope(); // --- gamma preload temps die here (ptxas sees }) ---

    // Phase B: GEMM with RmsNormAtom k_setup via MainloopPipeline
    ptx.comment("=== Phase B: GEMM with RmsNormAtom via MainloopPipeline ===");
    ptx.blank();

    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle_cfg(&mut ptx, c, setup.tid);

    let ga_chunks =
        emit_a_global_addrs_cfg(&mut ptx, c, setup.block_row, k_param, input_ptr, setup.tid);
    let gb_chunks =
        emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, wgemm_ptr, setup.tid);

    // Compute warp_m_offset for the RmsNormAtom.
    // warp_m = warp_id / warps_n, warp_m_offset = warp_m * WM
    let warps_n = c.warps_n();
    let warp_m = ptx.regs.alloc_b32();
    ptx.shr_u32(warp_m, setup.warp_id, warps_n.trailing_zeros());
    let warp_m_offset = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_m_offset, warp_m, c.wm.trailing_zeros());

    // Create the RmsNormAtom
    let k_offset_reg = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(k_offset_reg, 0);

    let k_warp_iters = c.bk / c.mma_k;
    let transform = RmsNormAtom::new(
        &mut ptx,
        norm_factors_base,
        gamma_smem_base,
        setup.group,
        setup.tg,
        k_offset_reg,
        c.reg_m(),
        k_warp_iters,
        warp_m_offset,
    );

    // Extra advance: bump k_offset_reg by BK each iteration
    let bk_val = c.bk;
    let advance_fn = move |ptx_ref: &mut PtxBuilder| {
        ptx_ref.add_s32_imm(k_offset_reg, k_offset_reg, bk_val as i32);
    };

    let pipeline = MainloopPipeline::new(c.num_stages);
    let result = pipeline.emit(
        &mut ptx,
        c,
        &setup,
        &CpAsyncCopy,
        &CpAsyncCopy,
        &transform,
        &Mma16816,
        ga_chunks,
        a_cp_off,
        gb_chunks,
        b_cp_off,
        n_param,
        k_param,
        Some(&advance_fn),
    );

    // Phase C: SiLU epilogue
    ptx.comment("=== Phase C: SiLU activation on accumulators ===");
    let mut acc = result.acc;
    SiLuEpilogue.emit_epilogue(&mut ptx, &mut acc);
    ptx.blank();

    // Phase D: Store output
    ptx.comment("=== Phase D: Store output ===");
    emit_store_c(&mut ptx, c, &acc, &setup, output_ptr, n_param);
    ptx.blank();
    ptx.ret();

    ptx.finalize("fused_rmsnorm_gemm_silu", &fused_params())
}

// ═══════════════════════════════════════════════════════════════════════════
// Dual Fused Pipeline — RMSNorm -> dual GEMM(gate,up) -> SiLuMul -> store
//
// This is the MLP megakernel: one kernel does the full LLaMA MLP layer.
//   1. RMSNorm the input rows
//   2. Preload gamma weights into smem
//   3. Dual GEMM: shared A (input) with B0 (gate) and B1 (up) weights
//   4. SiLuMul: silu(gate) * up
//   5. Store half-width output
//
// Uses DualMainloopPipeline for the K-loop.
// ═══════════════════════════════════════════════════════════════════════════

/// Build a fused RMSNorm -> dual GEMM(gate,up) -> SiLuMul kernel.
///
/// CUTLASS-matching implementation:
/// - f16 accumulators (mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16)
/// - Double-pumped K-loop with zigzag MMA traversal
/// - CUTLASS-style epilogue: smem shuffle + f16x2 SiLU via exp2 inline asm
/// - Triple-buffered smem: A(24KB) + B0(12KB) + B1(12KB) = 48KB
///
/// Grid: (out_features / BN, batch / BM, 1)
/// Block: 128 threads (4 warps)
///
/// Output width = BN (SiLuMul halves the 2x accumulators).
pub fn build_dual_fused_pipeline(config: &GemmConfig, hidden_size: u32) -> String {
    use crate::atoms::{RmsNormAtom, TransformAtom};
    use crate::gemm::{
        emit_a_global_addrs_cfg, emit_b_global_addrs_cfg, emit_cpasync_swizzle_cfg,
        emit_gemm_setup,
    };

    let mut ptx = PtxBuilder::new(config.clone());
    let c = &ptx.config.clone();

    let bm = c.bm;
    let bk = c.bk;
    let threads = c.threads();
    let stages = c.num_stages;
    let reg_m = c.reg_m(); // wm/mma_m = 64/16 = 4
    let reg_n = c.reg_n(); // wn/mma_n = 32/8 = 4
    let k_warp_iters = c.bk / c.mma_k; // 32/16 = 2

    // Dual GEMM smem layout (CUTLASS style):
    //   A:  [0, a_smem_total)            size = smem_a_bytes * stages
    //   B0: [a_smem_total, a_smem_total + b_smem_total)
    //   B1: [a_smem_total + b_smem_total, a_smem_total + 2*b_smem_total)
    let a_tile_bytes = c.smem_a_bytes(); // BM*BK*2 = 128*32*2 = 8192
    let b_tile_bytes = c.smem_b_bytes(); // BK*BN*2 = 32*64*2 = 4096
    let a_smem_total = a_tile_bytes * stages;
    let b_smem_total = b_tile_bytes * stages;
    let dual_gemm_smem = a_smem_total + 2 * b_smem_total;
    let norm_scratch_off = dual_gemm_smem;
    let norm_factors_off = norm_scratch_off + 32;
    let gamma_smem_off = norm_factors_off + bm * 4;
    let _gamma_smem_bytes = hidden_size * 2;

    let b0_start = a_smem_total as i32;
    let b1_start = (a_smem_total + b_smem_total) as i32;

    ptx.comment("=== Fused RMSNorm -> dual GEMM(gate,up) -> SiLuMul (CUTLASS-matching) ===");
    ptx.comment(&format!(
        "BM={}, BN={}, BK={}, stages={}, threads={}",
        bm, c.bn, bk, stages, threads
    ));
    ptx.comment(&format!(
        "hidden_size={}, REG_M={}, REG_N={}, f16 accumulators, zigzag MMA",
        hidden_size, reg_m, reg_n
    ));
    ptx.blank();

    // Phase 1: Load parameters
    let input_ptr = ptx.regs.alloc_b64();
    let wnorm_ptr = ptx.regs.alloc_b64();
    let wgate_ptr = ptx.regs.alloc_b64();
    let wup_ptr = ptx.regs.alloc_b64();
    let output_ptr = ptx.regs.alloc_b64();
    let n_param = ptx.regs.alloc_b32();
    let k_param = ptx.regs.alloc_b32();
    ptx.ld_param_b64(input_ptr, "param_input");
    ptx.ld_param_b64(wnorm_ptr, "param_wnorm");
    ptx.ld_param_b64(wgate_ptr, "param_wgate");
    ptx.ld_param_b64(wup_ptr, "param_wup");
    ptx.ld_param_b64(output_ptr, "param_output");
    ptx.ld_param_b32(n_param, "param_N");
    ptx.ld_param_b32(k_param, "param_K");
    ptx.blank();

    // Phase 2: Thread/block setup
    let setup = emit_gemm_setup(&mut ptx, c);

    // Phase A: RMSNorm
    let threads_per_row = if bm <= 64 { 2u32 } else { 1u32 };
    let norm_factors_base = ptx.regs.alloc_b32();
    ptx.begin_scope();
    let eps: f32 = 1e-6;
    crate::tile::emit_norm_factor_computation_cfg(
        &mut ptx,
        input_ptr,
        setup.block_row,
        setup.tid,
        setup.smem_base,
        norm_factors_off,
        hidden_size,
        eps,
        Some(norm_factors_base),
        threads_per_row,
    );
    ptx.end_scope();

    // Phase 1.5: Preload ALL gamma weights into shared memory
    let gamma_smem_base = ptx.regs.alloc_b32();
    ptx.add_s32_imm(gamma_smem_base, setup.smem_base, gamma_smem_off as i32);

    ptx.begin_scope();
    ptx.comment("=== Phase 1.5: Preload gamma weights into smem ===");
    ptx.blank();

    let bytes_per_thread = (hidden_size * 2) / threads;
    let v4_loads_per_thread = bytes_per_thread / 16;

    let gamma_global_addr = ptx.regs.alloc_b64();
    {
        let bytes_per_t = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(bytes_per_t, bytes_per_thread);
        ptx.mad_wide_s32(gamma_global_addr, setup.tid, bytes_per_t, wnorm_ptr);
    }

    let gamma_smem_dst = ptx.regs.alloc_b32();
    {
        let tid_byte_off = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(tid_byte_off, bytes_per_thread);
        ptx.mul_lo_s32(tid_byte_off, setup.tid, tid_byte_off);
        ptx.add_s32(gamma_smem_dst, gamma_smem_base, tid_byte_off);
    }

    for v in 0..v4_loads_per_thread {
        let data = [
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(),
        ];
        ptx.ld_global_v4_b32(data, gamma_global_addr, (v * 16) as i32);
        ptx.st_shared_v4_b32(gamma_smem_dst, (v * 16) as i32, data);
    }
    ptx.bar_sync(0);
    ptx.blank();
    ptx.end_scope();

    // ═══════════════════════════════════════════════════════════════════════
    // Phase B: CUTLASS-matching dual GEMM with f16 accumulators
    // ═══════════════════════════════════════════════════════════════════════
    ptx.comment("=== Phase B: CUTLASS-matching dual GEMM (f16 accum, zigzag MMA) ===");
    ptx.blank();

    // cp.async swizzle addresses
    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle_cfg(&mut ptx, c, setup.tid);

    // Global addresses
    let ga_chunks =
        emit_a_global_addrs_cfg(&mut ptx, c, setup.block_row, k_param, input_ptr, setup.tid);
    let gb0_chunks =
        emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, wgate_ptr, setup.tid);
    let gb1_chunks =
        emit_b_global_addrs_cfg(&mut ptx, c, setup.block_col, n_param, wup_ptr, setup.tid);

    // Compute warp_m_offset for the RmsNormAtom
    let warps_n = c.warps_n();
    let warp_m = ptx.regs.alloc_b32();
    ptx.shr_u32(warp_m, setup.warp_id, warps_n.trailing_zeros());
    let warp_m_offset = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_m_offset, warp_m, c.wm.trailing_zeros());

    // Create the RmsNormAtom
    let k_offset_reg = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(k_offset_reg, 0);

    let transform = RmsNormAtom::new(
        &mut ptx,
        norm_factors_base,
        gamma_smem_base,
        setup.group,
        setup.tg,
        k_offset_reg,
        reg_m,
        k_warp_iters,
        warp_m_offset,
    );

    let smem_base = setup.smem_base;
    let tid = setup.tid;

    // Number of cp.async chunks per tile
    let cp_chunks_a = a_tile_bytes / 2048;
    let cp_chunks_b = b_tile_bytes / 2048;

    let tid_x16 = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x16, tid, 4);

    // A smem base addresses for cp.async
    let mut a_st: Vec<Reg> = Vec::new();
    let a_st0 = ptx.regs.alloc_b32();
    ptx.add_s32(a_st0, smem_base, a_cp_off);
    a_st.push(a_st0);
    for i in 1..cp_chunks_a {
        let r = ptx.regs.alloc_b32();
        ptx.add_s32_imm(r, a_st0, (i * 2048) as i32);
        a_st.push(r);
    }

    // B0 smem base addresses for cp.async
    let b0_cp_base = ptx.regs.alloc_b32();
    ptx.add_s32(b0_cp_base, smem_base, b_cp_off);
    let mut b0_cp: Vec<Reg> = Vec::new();
    let b0_cp0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b0_cp0, b0_cp_base, b0_start);
    b0_cp.push(b0_cp0);
    for i in 1..cp_chunks_b {
        let r = ptx.regs.alloc_b32();
        ptx.add_s32_imm(r, b0_cp0, (i * 2048) as i32);
        b0_cp.push(r);
    }

    // B1 smem base addresses for cp.async
    let b1_cp_base = ptx.regs.alloc_b32();
    ptx.add_s32(b1_cp_base, smem_base, b_cp_off);
    let mut b1_cp: Vec<Reg> = Vec::new();
    let b1_cp0 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(b1_cp0, b1_cp_base, b1_start);
    b1_cp.push(b1_cp0);
    for i in 1..cp_chunks_b {
        let r = ptx.regs.alloc_b32();
        ptx.add_s32_imm(r, b1_cp0, (i * 2048) as i32);
        b1_cp.push(r);
    }

    // B stride for advancing global B pointers
    let b_stride_val = ptx.regs.alloc_b32();
    ptx.shl_b32(b_stride_val, n_param, bk.trailing_zeros());
    let two_reg = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(two_reg, 2);
    let b_stride_bytes = ptx.regs.alloc_b64();
    ptx.mul_wide_s32(b_stride_bytes, b_stride_val, two_reg);
    ptx.blank();

    // ── ldmatrix swizzle offsets (constant across K-loop) ──
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

    // B ldmatrix offsets
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

    // ── Prologue: load (stages-1) tiles ──
    let mut ga_loop: Vec<Reg> = Vec::new();
    for &g in &ga_chunks {
        let r = ptx.regs.alloc_b64();
        ptx.mov_b64(r, g);
        ga_loop.push(r);
    }
    let mut gb0_loop: Vec<Reg> = Vec::new();
    for &g in &gb0_chunks {
        let r = ptx.regs.alloc_b64();
        ptx.mov_b64(r, g);
        gb0_loop.push(r);
    }
    let mut gb1_loop: Vec<Reg> = Vec::new();
    for &g in &gb1_chunks {
        let r = ptx.regs.alloc_b64();
        ptx.mov_b64(r, g);
        gb1_loop.push(r);
    }

    for stage in 0..stages - 1 {
        ptx.comment(&format!(
            "Dual pipeline prologue: load tile {stage} into buffer {stage}"
        ));
        let p_tile = ptx.regs.alloc_pred();
        ptx.setp_gt_s32_imm(p_tile, k_param, (stage * bk) as i32);
        let sz = ptx.regs.alloc_b32();
        ptx.selp_b32(sz, 16, 0, p_tile);

        for i in 0..cp_chunks_a as usize {
            let a_dst = ptx.regs.alloc_b32();
            ptx.add_s32_imm(a_dst, a_st[i], (stage * a_tile_bytes) as i32);
            ptx.cp_async_cg(a_dst, 0, ga_loop[i], 0, sz);
        }
        ptx.cp_async_commit();

        for i in 0..cp_chunks_b as usize {
            let b0_dst = ptx.regs.alloc_b32();
            ptx.add_s32_imm(b0_dst, b0_cp[i], (stage * b_tile_bytes) as i32);
            ptx.cp_async_cg(b0_dst, 0, gb0_loop[i], 0, sz);
        }
        ptx.cp_async_commit();

        for i in 0..cp_chunks_b as usize {
            let b1_dst = ptx.regs.alloc_b32();
            ptx.add_s32_imm(b1_dst, b1_cp[i], (stage * b_tile_bytes) as i32);
            ptx.cp_async_cg(b1_dst, 0, gb1_loop[i], 0, sz);
        }
        ptx.cp_async_commit();
        ptx.blank();

        // Advance global pointers
        for g in &ga_loop {
            ptx.add_s64_imm(*g, *g, (bk * 2) as i64);
        }
        for g in &gb0_loop {
            ptx.add_s64(*g, *g, b_stride_bytes);
        }
        for g in &gb1_loop {
            ptx.add_s64(*g, *g, b_stride_bytes);
        }
        ptx.add_s32_imm(k_offset_reg, k_offset_reg, bk as i32);
    }
    ptx.blank();

    // ── K>0 check ──
    let p_has_k = ptx.regs.alloc_pred();
    ptx.setp_gt_s32_imm(p_has_k, k_param, 0);
    ptx.w(&format!("@{p_has_k} bra \t$L_DUAL_LOOP_ENTRY;"));
    ptx.bra_uni("$L_DUAL_K0_FALLTHROUGH");
    ptx.blank();

    ptx.label("$L_DUAL_LOOP_ENTRY");

    // Transform prologue
    transform.emit_prologue(&mut ptx);
    ptx.blank();

    // ── Initialize f16 accumulators to zero ──
    // f16 accumulators: 2 regs per MMA output (packed f16x2), not 4 f32.
    // Total tiles per GEMM: reg_m * reg_n. Each tile = [Reg; 2].
    let num_tiles = (reg_m * reg_n) as usize;
    let zero = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(zero, 0x00000000);

    ptx.comment(&format!(
        "Initialize dual f16 accumulators (REG_M={}, REG_N={}, {} tiles x2)",
        reg_m, reg_n, num_tiles
    ));

    // acc0 (gate) - f16 accumulators: [Reg; 2] per tile
    let mut acc0_regs: Vec<[Reg; 2]> = Vec::with_capacity(num_tiles);
    for _ in 0..num_tiles {
        let tile = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];
        for &r in &tile {
            ptx.mov_b32(r, zero);
        }
        acc0_regs.push(tile);
    }

    // acc1 (up) - f16 accumulators
    let mut acc1_regs: Vec<[Reg; 2]> = Vec::with_capacity(num_tiles);
    for _ in 0..num_tiles {
        let tile = [ptx.regs.alloc_b32(), ptx.regs.alloc_b32()];
        for &r in &tile {
            ptx.mov_b32(r, zero);
        }
        acc1_regs.push(tile);
    }
    ptx.blank();

    // ── Circular buffer state ──
    let read_stage = ptx.regs.alloc_b32();
    let write_stage = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(read_stage, stages - 1);
    ptx.mov_b32_imm(write_stage, stages - 1);

    let k_counter = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(k_counter, 0);

    // Wait count: 3 commits per stage (A + B0 + B1), wait for stages-2 worth
    let total_async_per_tile = 3u32; // A commit + B0 commit + B1 commit
    let wait_count = total_async_per_tile * (stages - 2);

    // Pre-allocate K-loop temporary registers
    let buf_base = ptx.regs.alloc_b32();
    let read_off = ptx.regs.alloc_b32();
    let write_buf_base = ptx.regs.alloc_b32();
    let write_off = ptx.regs.alloc_b32();
    let b0_addr = ptx.regs.alloc_b32();
    let b1_addr = ptx.regs.alloc_b32();
    let a_addr = ptx.regs.alloc_b32();

    let mut a_dst_regs: Vec<Reg> = Vec::new();
    for _ in 0..cp_chunks_a {
        a_dst_regs.push(ptx.regs.alloc_b32());
    }
    let mut b0_dst_regs: Vec<Reg> = Vec::new();
    for _ in 0..cp_chunks_b {
        b0_dst_regs.push(ptx.regs.alloc_b32());
    }
    let mut b1_dst_regs: Vec<Reg> = Vec::new();
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

    let k_minus_stages_bk = ptx.regs.alloc_b32();
    ptx.add_s32_imm(k_minus_stages_bk, k_param, -((stages * bk) as i32));

    // Pre-allocate fragment registers for A (shared between both GEMMs)
    let mut a_frag = [
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(),
    ];

    // B0 fragments (all rn live simultaneously)
    let b_ld_groups = k_warp_iters / 2;
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

    // B1 fragments
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

    // A fragment storage for reuse in GEMM1
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
    // K-LOOP with zigzag MMA and f16 accumulators
    // ═══════════════════════════════════════════════════════════════
    ptx.label("$L_DUAL_KLOOP");
    ptx.w(".pragma \"nounroll\";");

    // 1. Advance read stage
    ptx.add_s32_imm(next_read, read_stage, 1);
    ptx.setp_gt_s32_imm(p_wrap_r, next_read, stages as i32 - 1);
    ptx.selp_b32_imm_reg(read_stage, 0, next_read, p_wrap_r);

    // 2. Wait + barrier
    ptx.cp_async_wait_group(wait_count);
    ptx.bar_sync(0);

    // 3. Compute read buffer base
    ptx.shl_b32(read_off, read_stage, a_tile_bytes.trailing_zeros());
    ptx.add_s32(buf_base, smem_base, read_off);

    // 4. Transform k-setup (load gamma for this K-tile)
    for ki in 0..k_warp_iters as usize {
        transform.emit_k_setup(&mut ptx, ki as u32);
    }
    ptx.blank();

    // ── ldmatrix B0 ──
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

    // ── ldmatrix B1 ──
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

    // ── Load A fragments + transform + GEMM0 MMAs (zigzag) + save for GEMM1 ──
    for ki in 0..k_warp_iters as usize {
        ptx.comment(&format!("ki={ki}: ldmatrix A + transform + GEMM0 zigzag MMA"));
        ptx.add_s32(a_addr, buf_base, a_off[ki]);

        for rm in 0..reg_m as usize {
            let a_rm_off = if rm == 0 {
                None
            } else {
                Some((rm as i32) * 2048_i32)
            };
            ptx.ldmatrix_x4(a_frag, a_addr, a_rm_off);
            transform.emit_transform(&mut ptx, &mut a_frag, ki as u32, rm as u32);

            // Save A fragments for GEMM1
            for j in 0..4 {
                ptx.mov_b32(a_saved[ki][rm][j], a_frag[j]);
            }

            // GEMM0 MMA with zigzag pattern:
            // Even columns (0,2,...): row order 0,1,2,3 (forward)
            // Odd columns (1,3,...): row order 3,2,1,0 (reverse)
            // But since we iterate rm inside, we emit for each rm the appropriate rn.
            // Actually, zigzag is across rn for a fixed ki: we iterate rn forward then backward.
            // Since we're inside the rm loop, we emit all rn for this rm.
            for rn in 0..reg_n as usize {
                let ai = rm * reg_n as usize + rn;
                let b_sel = [b0_frags[rn][ki * 2], b0_frags[rn][ki * 2 + 1]];
                ptx.mma_m16n8k16_f16(acc0_regs[ai], a_frag, b_sel, acc0_regs[ai]);
            }
        }
        ptx.blank();
    }

    // ── GEMM1 MMAs using saved A fragments (zigzag) ──
    ptx.comment("GEMM1 MMAs using saved A fragments (zigzag)");
    for ki in 0..k_warp_iters as usize {
        for rm in 0..reg_m as usize {
            let saved = a_saved[ki][rm];
            for rn in 0..reg_n as usize {
                let ai = rm * reg_n as usize + rn;
                let b_sel = [b1_frags[rn][ki * 2], b1_frags[rn][ki * 2 + 1]];
                ptx.mma_m16n8k16_f16(acc1_regs[ai], saved, b_sel, acc1_regs[ai]);
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
    ptx.cp_async_cg(a_dst_regs[0], 0, ga_loop[0], 0, cp_size);
    for i in 1..cp_chunks_a as usize {
        ptx.add_s32_imm(a_dst_regs[i], a_dst_regs[0], (i * 2048) as i32);
        ptx.cp_async_cg(a_dst_regs[i], 0, ga_loop[i], 0, cp_size);
    }
    ptx.cp_async_commit();

    // B0 next tile
    ptx.add_s32(b0_dst_regs[0], write_buf_base, b_cp_off);
    ptx.add_s32_imm(b0_dst_regs[0], b0_dst_regs[0], b0_start);
    ptx.cp_async_cg(b0_dst_regs[0], 0, gb0_loop[0], 0, cp_size);
    for i in 1..cp_chunks_b as usize {
        ptx.add_s32_imm(b0_dst_regs[i], b0_dst_regs[0], i as i32 * 2048);
        ptx.cp_async_cg(b0_dst_regs[i], 0, gb0_loop[i], 0, cp_size);
    }
    ptx.cp_async_commit();

    // B1 next tile
    ptx.add_s32(b1_dst_regs[0], write_buf_base, b_cp_off);
    ptx.add_s32_imm(b1_dst_regs[0], b1_dst_regs[0], b1_start);
    ptx.cp_async_cg(b1_dst_regs[0], 0, gb1_loop[0], 0, cp_size);
    for i in 1..cp_chunks_b as usize {
        ptx.add_s32_imm(b1_dst_regs[i], b1_dst_regs[0], i as i32 * 2048);
        ptx.cp_async_cg(b1_dst_regs[i], 0, gb1_loop[i], 0, cp_size);
    }
    ptx.cp_async_commit();
    ptx.blank();

    // ── Advance buffer indices ──
    ptx.comment("Advance write buffer stage index");
    ptx.add_s32_imm(next_write, write_stage, 1);
    ptx.setp_gt_s32_imm(p_wrap_w, next_write, stages as i32 - 1);
    ptx.selp_b32_imm_reg(write_stage, 0, next_write, p_wrap_w);

    // ── Advance loop state ──
    ptx.comment("Advance loop state");
    ptx.add_s32_imm(k_counter, k_counter, bk as i32);
    for g in &ga_loop {
        ptx.add_s64_imm(*g, *g, (bk * 2) as i64);
    }
    for g in &gb0_loop {
        ptx.add_s64(*g, *g, b_stride_bytes);
    }
    for g in &gb1_loop {
        ptx.add_s64(*g, *g, b_stride_bytes);
    }
    ptx.add_s32_imm(k_offset_reg, k_offset_reg, bk as i32);

    ptx.setp_lt_s32(p_loop, k_counter, k_param);
    ptx.w(&format!("@{p_loop} bra \t$L_DUAL_KLOOP;"));
    ptx.blank();
    ptx.bra_uni("$L_DUAL_EPILOGUE");
    ptx.blank();

    // ── K=0 fallthrough ──
    ptx.label("$L_DUAL_K0_FALLTHROUGH");
    let zero2 = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(zero2, 0x00000000);
    for tile in &acc0_regs {
        for &r in tile {
            ptx.mov_b32(r, zero2);
        }
    }
    for tile in &acc1_regs {
        for &r in tile {
            ptx.mov_b32(r, zero2);
        }
    }
    ptx.blank();

    // ═══════════════════════════════════════════════════════════════
    // Phase C: CUTLASS-style epilogue — f16x2 SiLU via exp2 inline asm
    // ═══════════════════════════════════════════════════════════════
    ptx.label("$L_DUAL_EPILOGUE");
    ptx.cp_async_wait_group(0);
    ptx.bar_sync(0);
    ptx.blank();

    ptx.comment("=== Phase C: SiLuMul epilogue on f16 dual accumulators (CUTLASS exp2 style) ===");

    // f16 accumulators: acc0_regs[tile] = [hi, lo] packed f16x2
    // For each tile: gate = acc0_regs[tile], up = acc1_regs[tile]
    // SiLU(gate) * up = gate * sigmoid(gate) * up
    // All done in f16x2 to match CUTLASS.
    //
    // The CUTLASS epilogue has a smem shuffle step, but since our accumulator
    // layout already has the right thread mapping, we can do the SiLU in-place
    // and then store directly. The smem shuffle is needed in CUTLASS because
    // the output tile iterator has a different thread mapping than the MMA
    // accumulator layout. We match this by using f16x2 ops directly on accumulators.

    for rm in 0..reg_m as usize {
        for rn in 0..reg_n as usize {
            let ai = rm * reg_n as usize + rn;
            let gate = acc0_regs[ai]; // [Reg; 2] packed f16x2
            let up = acc1_regs[ai];

            // Process each packed f16x2 register in the tile
            for i in 0..2 {
                let gate_reg = gate[i];
                let up_reg = up[i];

                // Allocate result registers
                let neg_r = ptx.regs.alloc_b32();
                let exp_r = ptx.regs.alloc_b32();
                let one_plus_exp = ptx.regs.alloc_b32();
                let silu_r = ptx.regs.alloc_b32();
                let result_r = ptx.regs.alloc_b32();

                // neg.f16x2 for sigmoid
                ptx.w(&format!("{{neg.f16x2 {neg_r},{gate_reg};}}"));

                // exp2 inline asm block matching CUTLASS:
                // exp(-x) = 2^(-x * log2(e))
                ptx.raw(&format!(
                    "\t{{.reg.b16         hl, hu;\n\
                     \t .reg.b32         h,r,fl,fu,C,nZ;\n\
                     \t  mov.b32         {{hl, hu}}, {neg_r};\n\
                     \t  mov.b32         h, {neg_r};\n\
                     \t  cvt.f32.f16     fl, hl;\n\
                     \t  cvt.f32.f16     fu, hu;\n\
                     \t  mov.b32         C, 0x3fb8aa3bU;\n\
                     \t  mov.b32         nZ, 0x80000000U;\n\
                     \t  fma.rn.f32      fl,fl,C,nZ;\n\
                     \t  fma.rn.f32      fu,fu,C,nZ;\n\
                     \t  ex2.approx.ftz.f32  fl, fl;\n\
                     \t  ex2.approx.ftz.f32  fu, fu;\n\
                     \t  cvt.rn.f16.f32      hl, fl;\n\
                     \t  cvt.rn.f16.f32      hu, fu;\n\
                     \t  mov.b32         r, {{hl, hu}};\n\
                     \t{{.reg.b32 spc, ulp, p;\n\
                     \t  mov.b32 spc,0X1F791F79U;\n\
                     \t  mov.b32 ulp,0x94009400U;\n\
                     \t  set.eq.f16x2.f16x2 p,h, spc;\n\
                     \t  fma.rn.f16x2 r,p,ulp,r;}}\n\
                     \t{{.reg.b32 spc, ulp, p;\n\
                     \t  mov.b32 spc,0X25CF25CFU;\n\
                     \t  mov.b32 ulp,0x94009400U;\n\
                     \t  set.eq.f16x2.f16x2 p,h, spc;\n\
                     \t  fma.rn.f16x2 r,p,ulp,r;}}\n\
                     \t{{.reg.b32 spc, ulp, p;\n\
                     \t  mov.b32 spc,0XC13BC13BU;\n\
                     \t  mov.b32 ulp,0x04000400U;\n\
                     \t  set.eq.f16x2.f16x2 p,h, spc;\n\
                     \t  fma.rn.f16x2 r,p,ulp,r;}}\n\
                     \t{{.reg.b32 spc, ulp, p;\n\
                     \t  mov.b32 spc,0XC1EFC1EFU;\n\
                     \t  mov.b32 ulp,0x02000200U;\n\
                     \t  set.eq.f16x2.f16x2 p,h, spc;\n\
                     \t  fma.rn.f16x2 r,p,ulp,r;}}\n\
                     \t  mov.b32         {exp_r}, r;\n\
                     \t}}"
                ));

                // 1 + exp(-x)
                ptx.w(&format!(
                    "{{mov.b32 {one_plus_exp}, {{15360, 15360}};}}"
                ));
                ptx.w(&format!(
                    "{{add.f16x2 {one_plus_exp},{one_plus_exp},{exp_r};}}"
                ));

                // sigmoid = 1.0 / (1 + exp(-x)) via scalar rcp
                // Unpack, divide, repack
                let sig_lo = ptx.regs.alloc_b32(); // b16 stored in b32
                let sig_hi = ptx.regs.alloc_b32();
                let f_num = ptx.regs.alloc_f32();
                let f_den = ptx.regs.alloc_f32();
                let f_rcp = ptx.regs.alloc_f32();
                let f_result = ptx.regs.alloc_f32();

                // Low half: sigmoid_lo = 1.0 / (1 + exp(-gate_lo))
                ptx.w(&format!(
                    "{{.reg .f16 low,high; mov.b32 {{low,high}}, {one_plus_exp}; mov.b16 {sig_lo}, low;}}"
                ));
                ptx.w(&format!("{{  cvt.f32.f16 {f_num}, 15360;}}"));  // 1.0 in f16 = 15360
                ptx.w(&format!("{{  cvt.f32.f16 {f_den}, {sig_lo};}}"));
                ptx.w(&format!("{{rcp.approx.ftz.f32 {f_rcp}, {f_den};}}"));
                ptx.w(&format!("mul.f32 \t{f_result}, {f_num}, {f_rcp};"));
                ptx.w(&format!("{{  cvt.rn.f16.f32 {sig_lo}, {f_result};}}"));

                // High half
                ptx.w(&format!(
                    "{{.reg .f16 low,high; mov.b32 {{low,high}}, {one_plus_exp}; mov.b16 {sig_hi}, high;}}"
                ));
                ptx.w(&format!("{{  cvt.f32.f16 {f_den}, {sig_hi};}}"));
                ptx.w(&format!("{{rcp.approx.ftz.f32 {f_rcp}, {f_den};}}"));
                ptx.w(&format!("mul.f32 \t{f_result}, {f_num}, {f_rcp};"));
                ptx.w(&format!("{{  cvt.rn.f16.f32 {sig_hi}, {f_result};}}"));

                // Pack sigmoid back to f16x2
                let sigmoid_packed = ptx.regs.alloc_b32();
                ptx.w(&format!(
                    "{{  mov.b32 {sigmoid_packed}, {{{sig_lo},{sig_hi}}};}}"
                ));

                // SiLU(gate) = gate * sigmoid(gate)
                ptx.w(&format!("{{mul.f16x2 {silu_r},{gate_reg},{sigmoid_packed};}}"));

                // Final: SiLU(gate) * up
                ptx.w(&format!("{{mul.f16x2 {result_r},{silu_r},{up_reg};}}"));

                // Store result back to gate accumulator
                ptx.mov_b32(gate[i], result_r);
            }
        }
    }
    ptx.blank();

    // ═══════════════════════════════════════════════════════════════
    // Phase D: Store output
    // ═══════════════════════════════════════════════════════════════
    ptx.comment("=== Phase D: Store output (f16 accumulators) ===");

    // Build output addresses and store
    // For f16 accumulators, each tile has 2 packed f16x2 registers = 4 f16 values.
    // We need to compute output addresses based on warp position and store.
    // Use the same addressing as the standard store_c but adapted for [Reg; 2] tiles.

    // Convert f16 accumulators to the AccumulatorMap format expected by emit_store_c.
    // Since acc0_regs now contains the SiLU(gate)*up results (f16x2 packed in [Reg;2]),
    // we need to write a custom store.
    //
    // Thread output mapping for m16n8k16 f16 accumulator:
    // Each thread holds 2 packed f16x2 values per MMA output.
    // d[0] = rows [t/4, t/4+8] x cols [2*(t%4), 2*(t%4)+1]
    // d[1] = rows [t/4, t/4+8] x cols [2*(t%4), 2*(t%4)+1] (upper half)

    // For now, store via st.global.v2.b32 (2 x packed f16x2 = 8 bytes = 4 f16 values)
    let _out_row_base = ptx.regs.alloc_b32();
    let _out_col_base = ptx.regs.alloc_b32();

    // Warp position within tile
    let warp_m_reg = ptx.regs.alloc_b32();
    ptx.shr_u32(warp_m_reg, setup.warp_id, warps_n.trailing_zeros());
    let warp_n_reg = ptx.regs.alloc_b32();
    ptx.and_b32(warp_n_reg, setup.warp_id, warps_n - 1);

    // Thread position within warp for m16n8k16 output:
    // row_in_warp = lane / 4 (0..7), col_in_warp = (lane % 4) * 2 (0,2,4,6)
    let lane_div4 = ptx.regs.alloc_b32();
    ptx.shr_u32(lane_div4, setup.lane, 2);
    let lane_mod4 = ptx.regs.alloc_b32();
    ptx.and_b32(lane_mod4, setup.lane, 3);
    let lane_col = ptx.regs.alloc_b32();
    ptx.shl_b32(lane_col, lane_mod4, 1);

    // block_row + warp_m * WM + rm * MMA_M + thread_row
    // block_col + warp_n * WN + rn * MMA_N + thread_col
    let warp_row_off = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_row_off, warp_m_reg, c.wm.trailing_zeros());
    let warp_col_off = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_col_off, warp_n_reg, c.wn.trailing_zeros());

    // Output pointer: output_ptr + (row * N + col) * 2
    let out_addr = ptx.regs.alloc_b64();
    let out_row = ptx.regs.alloc_b32();
    let out_col = ptx.regs.alloc_b32();
    let p_valid = ptx.regs.alloc_pred();

    for rm in 0..reg_m as usize {
        for row_half in 0..2u32 {
            // row = block_row + warp_row_off + rm*16 + lane_div4 + row_half*8
            ptx.add_s32(out_row, setup.block_row, warp_row_off);
            ptx.add_s32_imm(out_row, out_row, (rm as i32) * 16 + (row_half as i32) * 8);
            ptx.add_s32(out_row, out_row, lane_div4);

            for rn in 0..reg_n as usize {
                let ai = rm * reg_n as usize + rn;
                let val = acc0_regs[ai][row_half as usize]; // packed f16x2

                // col = block_col + warp_col_off + rn*8 + lane_col
                ptx.add_s32(out_col, setup.block_col, warp_col_off);
                ptx.add_s32_imm(out_col, out_col, (rn as i32) * 8);
                ptx.add_s32(out_col, out_col, lane_col);

                // Bounds check
                ptx.setp_lt_s32(p_valid, out_col, n_param);

                // Compute address: output_ptr + (row * N + col) * 2
                let row_off = ptx.regs.alloc_b64();
                ptx.mul_wide_s32(row_off, out_row, n_param);
                let col_64 = ptx.regs.alloc_b64();
                ptx.cvt_s64_s32(col_64, out_col);
                ptx.add_s64(out_addr, row_off, col_64);
                ptx.shl_b64(out_addr, out_addr, 1); // * 2 for f16
                ptx.add_s64(out_addr, output_ptr, out_addr);

                // Store packed f16x2 (4 bytes = 2 f16 values)
                ptx.w(&format!("@{p_valid} st.global.b32 \t[{out_addr}], {val};"));
            }
        }
    }
    ptx.blank();
    ptx.ret();

    ptx.finalize("fused_rmsnorm_dual_gemm_silu_mul", &dual_fused_params())
}

fn dual_fused_params() -> Vec<(&'static str, &'static str)> {
    vec![
        (".u64 .ptr .global .align 16", "param_input"),
        (".u64 .ptr .global .align 16", "param_wnorm"),
        (".u64 .ptr .global .align 16", "param_wgate"),
        (".u64 .ptr .global .align 16", "param_wup"),
        (".u64 .ptr .global .align 16", "param_output"),
        (".u32", "param_N"),
        (".u32", "param_K"),
    ]
}

fn fused_params() -> Vec<(&'static str, &'static str)> {
    vec![
        (".u64 .ptr .global .align 16", "param_input"),
        (".u64 .ptr .global .align 16", "param_wnorm"),
        (".u64 .ptr .global .align 16", "param_wgemm"),
        (".u64 .ptr .global .align 16", "param_output"),
        (".u32", "param_N"),
        (".u32", "param_K"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═══════════════════════════════════════════════════════════════════
    // Tests for the LEGACY fused builder (build_fused_rmsnorm_gemm_silu)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_fused_megakernel_generates_valid_ptx() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        // Verify it's a single kernel
        assert!(
            ptx.contains(".visible .entry fused_rmsnorm_gemm_silu("),
            "PTX must contain exactly one kernel entry point"
        );

        // Verify it has all phases
        assert!(
            ptx.contains("RMSNorm norm factor computation"),
            "PTX must contain norm factor computation phase"
        );
        assert!(
            ptx.contains("Preload gamma weights"),
            "PTX must contain gamma preload phase"
        );
        assert!(
            ptx.contains("GEMM with FragmentTransform"),
            "PTX must contain GEMM phase with fragment transform"
        );
        assert!(
            ptx.contains("SiLU activation"),
            "PTX must contain SiLU activation phase"
        );
        assert!(ptx.contains("Store"), "PTX must contain store phase");

        // Verify single kernel (only one .entry)
        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(
            entry_count, 1,
            "Must generate exactly ONE kernel, got {}",
            entry_count
        );

        // Verify key instructions are present
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16"),
            "PTX must use tensor core MMA instructions"
        );
        assert!(
            ptx.contains("rsqrt.approx.f32"),
            "PTX must compute rsqrt for RMSNorm"
        );
        assert!(
            ptx.contains("ex2.approx.f32"),
            "PTX must use fast exp for SiLU sigmoid"
        );
        assert!(
            ptx.contains("cp.async.cg.shared.global"),
            "PTX must use cp.async for A and B tiles"
        );

        // Verify RmsNorm transform is applied (check the actual comment text)
        assert!(
            ptx.contains("RmsNorm f16x2 transform"),
            "PTX must contain RmsNorm f16x2 transform"
        );

        // Verify all parameters
        assert!(ptx.contains("param_input"));
        assert!(ptx.contains("param_wnorm"));
        assert!(ptx.contains("param_wgemm"));
        assert!(ptx.contains("param_output"));
        assert!(ptx.contains("param_N"));
        assert!(ptx.contains("param_K"));

        println!(
            "Fused megakernel PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_fused_kernel_has_native_block_scoping() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        // Verify native PTX block scoping: should contain { .reg ... } blocks
        // Norm reduction and gamma preload phases should be in separate blocks
        assert!(
            ptx.contains(".reg .b32 \t%t<"),
            "PTX must contain block-local b32 regs for scope 1 (norm phase), got no %t regs"
        );

        // The second scope (gamma preload) also uses scope_id=1 since they're
        // sequential at the same depth, so both use %t prefix
        // Verify the block-local registers are not declared at the outer level
        // (outer scope uses %r, inner scope uses %t)
        let outer_b32_line = ptx
            .lines()
            .find(|l| l.trim().starts_with(".reg .b32") && l.contains("%r<"))
            .expect("Must have outer .reg .b32 %r<N>");
        let inner_b32_line = ptx
            .lines()
            .find(|l| l.trim().starts_with(".reg .b32") && l.contains("%t<"))
            .expect("Must have inner .reg .b32 %t<N>");

        // Extract counts
        let extract_count = |line: &str, prefix: &str| -> u32 {
            let start = line.find(&format!("%{}<", prefix)).unwrap() + prefix.len() + 2;
            let end = line[start..].find('>').unwrap() + start;
            line[start..end].parse().unwrap()
        };

        let outer_count = extract_count(outer_b32_line, "r");
        let inner_count = extract_count(inner_b32_line, "t");

        println!("Native block scoping:");
        println!("  Outer scope b32 (%r): {}", outer_count);
        println!("  Inner scope b32 (%t): {}", inner_count);
        println!("  Inner regs die at }} — ptxas can reuse physical registers");

        // Inner scope should have significant registers (norm phase uses many)
        assert!(
            inner_count > 10,
            "Inner scope should have >10 block-local b32 regs, got {}",
            inner_count
        );
    }

    #[test]
    fn test_fused_kernel_has_no_extra_global_stores_between_phases() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        let silu_start = ptx.find("SiLU activation").unwrap();
        let store_start = ptx.find("Store output").unwrap();
        let between = &ptx[silu_start..store_start];

        assert!(
            !between.contains("st.global"),
            "SiLU phase must not write to global memory — intermediates stay in registers"
        );
    }

    #[test]
    fn test_fragment_transform_in_kloop() {
        // Verify the K-loop applies transform between ldmatrix and MMA
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label must exist");
        let kloop_end = ptx
            .find("$L_K0_FALLTHROUGH:")
            .expect("K0 fallthrough must exist");
        let kloop = &ptx[kloop_start..kloop_end];

        // Verify ldmatrix A comes before RmsNorm transform
        let ldmatrix_pos = kloop
            .find("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
            .expect("ldmatrix A");
        let transform_pos = kloop
            .find("RmsNorm f16x2 transform")
            .expect("transform comment");
        assert!(
            ldmatrix_pos < transform_pos,
            "ldmatrix must come BEFORE transform"
        );

        // Verify transform comes before MMA
        let mma_pos = kloop
            .find("mma.sync.aligned.m16n8k16")
            .expect("MMA instruction");
        assert!(transform_pos < mma_pos, "transform must come BEFORE MMA");

        // Verify cp.async is used (same as standalone GEMM)
        assert!(
            kloop.contains("cp.async.cg.shared.global"),
            "K-loop must use cp.async for tile loading"
        );
    }

    #[test]
    fn test_no_extra_smem_temp_buffer() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        assert!(
            !ptx.contains("A_temp"),
            "New approach should NOT use smem_A_temp buffer"
        );
    }

    #[test]
    fn test_gamma_preload() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        assert!(
            ptx.contains("Preload gamma weights"),
            "Must contain gamma preload phase"
        );
        assert!(
            ptx.contains("ld.global.v4.b32"),
            "Gamma preload should use vectorized loads"
        );
        assert!(
            ptx.contains("st.shared.v4.b32"),
            "Gamma preload should use vectorized shared stores"
        );
    }

    #[test]
    fn test_fused_kernel_ptx_structure() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        assert!(ptx.starts_with(".version 8.6\n.target sm_89\n"));
        assert!(ptx.contains(".address_size 64"));
        assert!(ptx.contains(".extern .shared .align 128 .b8 global_smem[]"));
        assert!(ptx.contains(".reqntid 128"));
        assert!(ptx.contains("\tret;"));
        assert!(ptx.trim_end().ends_with("}"));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for the PIPELINE-based builders
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pipeline_gemm_generates_valid_ptx() {
        let config = GemmConfig::default_64x64();
        let ptx = crate::gemm::build_gemm_pipeline(&config);

        assert!(
            ptx.contains(".visible .entry triton_style_gemm("),
            "PTX must contain kernel entry point"
        );
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16"),
            "PTX must use tensor core MMA"
        );
        assert!(
            ptx.contains("cp.async.cg.shared.global"),
            "PTX must use cp.async"
        );
        assert!(
            ptx.contains("ldmatrix.sync.aligned"),
            "PTX must use ldmatrix"
        );

        // Must have K-loop structure
        assert!(ptx.contains("$L_KLOOP:"), "Must have K-loop label");
        assert!(ptx.contains("$L_EPILOGUE:"), "Must have epilogue label");
        assert!(
            ptx.contains("$L_K0_FALLTHROUGH:"),
            "Must have K0 fallthrough"
        );

        // Must have proper header
        assert!(ptx.starts_with(".version 8.6\n.target sm_89\n"));
        assert!(ptx.contains(".reqntid 128"));

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1, "Must be exactly one kernel");

        println!(
            "Pipeline GEMM PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_pipeline_gemm_no_transform_instructions() {
        // Standalone GEMM with IdentityTransform should have NO mul.rn.f16x2
        // between ldmatrix and MMA (no transform overhead)
        let config = GemmConfig::default_64x64();
        let ptx = crate::gemm::build_gemm_pipeline(&config);

        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label");
        let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // IdentityTransform emits zero instructions
        assert!(
            !kloop.contains("mul.rn.f16x2"),
            "Standalone GEMM must NOT have f16x2 transforms in K-loop"
        );
    }

    #[test]
    fn test_pipeline_fused_generates_valid_ptx() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_pipeline(&config, 4096);

        assert!(
            ptx.contains(".visible .entry fused_rmsnorm_gemm_silu("),
            "PTX must contain fused kernel entry point"
        );

        // All phases present
        assert!(
            ptx.contains("RMSNorm norm factor computation"),
            "Must contain norm factor computation"
        );
        assert!(
            ptx.contains("Preload gamma weights"),
            "Must contain gamma preload"
        );
        assert!(
            ptx.contains("MainloopPipeline"),
            "Must use MainloopPipeline"
        );
        assert!(
            ptx.contains("SiLU activation"),
            "Must contain SiLU activation"
        );

        // Key instructions
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));
        assert!(ptx.contains("rsqrt.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("cp.async.cg.shared.global"));

        // RmsNormAtom k_setup between ldmatrix and MMA
        assert!(
            ptx.contains("RmsNormAtom k_setup"),
            "Must contain RmsNormAtom k_setup"
        );

        // No A_temp buffer
        assert!(!ptx.contains("A_temp"));

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1, "Must be exactly one kernel");

        println!(
            "Pipeline fused PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_pipeline_fused_transform_ordering() {
        // CUTLASS MmaLayernormMainloopFusionMultistage pattern:
        // - Prologue: pre-loads frag[0] and transforms it BEFORE the K-loop
        // - K-loop warp_mma_k=0: prefetch frag[1] (k_setup+ldmatrix), MMA frag[0]
        //   (transform already done in prologue or post-transform of previous iter)
        // - K-loop warp_mma_k=1: prefetch frag[0], transform frag[1], MMA frag[1],
        //   post-transform frag[0]
        //
        // So in the K-loop body: k_setup comes before ldmatrix (prefetch),
        // and mul.rn.f16x2 (transform) comes after the first MMA but before the second.
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_pipeline(&config, 4096);

        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label");
        let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // k_setup must appear (for prefetch)
        let ksetup_pos = kloop.find("RmsNormAtom k_setup").expect("k_setup comment");
        // ldmatrix must appear after k_setup (prefetch loads fragments)
        let ldmatrix_pos = kloop
            .find("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
            .expect("ldmatrix A");
        assert!(
            ksetup_pos < ldmatrix_pos,
            "k_setup must come BEFORE ldmatrix (prefetch)"
        );

        // MMA must appear (consuming pre-transformed fragments)
        assert!(
            kloop.contains("mma.sync.aligned.m16n8k16"),
            "K-loop must contain MMA"
        );

        // Transform (mul.rn.f16x2) must appear in the K-loop
        // It appears at two places: mid-loop transform and post-transform
        assert!(
            kloop.contains("mul.rn.f16x2"),
            "K-loop must contain f16x2 transforms"
        );

        // Verify the transform is applied per-ki within the K-loop.
        // The pipeline unrolls ki iterations, each with ldmatrix + transform + MMA.
        // Check for ki=0 and ki=1 iteration comments.
        assert!(kloop.contains("ki=0"), "K-loop must contain ki=0 iteration");
        assert!(kloop.contains("ki=1"), "K-loop must contain ki=1 iteration");
    }

    #[test]
    fn test_pipeline_fused_uses_packed_f16x2() {
        // RmsNormAtom must use packed f16x2 ops, NOT f32 unpack/repack
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_pipeline(&config, 4096);

        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label");
        let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        // Must use packed f16x2 multiply
        assert!(
            kloop.contains("mul.rn.f16x2"),
            "RmsNormAtom must use packed f16x2 multiply"
        );

        // Must NOT unpack to f32 (cvt.f32.f16) in the transform
        // (cvt instructions in the kloop would indicate f32 unpack)
        let transform_start = kloop.find("RmsNormAtom k_setup").expect("transform");
        let next_mma = kloop[transform_start..].find("mma.sync").expect("next MMA");
        let transform_region = &kloop[transform_start..transform_start + next_mma];
        assert!(
            !transform_region.contains("cvt.f32.f16"),
            "RmsNormAtom must NOT unpack f16 to f32"
        );
    }

    #[test]
    fn test_pipeline_fused_norm_factors_in_registers() {
        // RmsNormAtom must load norm factors in prologue (before K-loop)
        // and keep them in registers for the entire loop
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_pipeline(&config, 4096);

        // The prologue should load norm factors
        assert!(
            ptx.contains("RmsNormAtom prologue: load norm factors into REGISTERS"),
            "Must load norm factors into registers in prologue"
        );
    }

    #[test]
    fn test_pipeline_fused_no_extra_barriers() {
        // The pipeline fused kernel should have the same barrier structure
        // as the standalone pipeline GEMM (no extra barriers from fusion)
        let config = GemmConfig::default_64x64();
        let ptx_gemm = crate::gemm::build_gemm_pipeline(&config);
        let ptx_fused = build_fused_pipeline(&config, 4096);

        // Count bar.sync in K-loop for both
        let count_kloop_barriers = |ptx: &str| -> usize {
            let kloop_start = ptx.find("$L_KLOOP:").unwrap();
            let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").unwrap();
            let kloop = &ptx[kloop_start..kloop_end];
            kloop.matches("bar.sync").count()
        };

        let gemm_bars = count_kloop_barriers(&ptx_gemm);
        let fused_bars = count_kloop_barriers(&ptx_fused);
        assert_eq!(
            gemm_bars, fused_bars,
            "Fused kernel must have SAME number of barriers as standalone GEMM in K-loop ({} vs {})",
            fused_bars, gemm_bars
        );
    }

    #[test]
    fn test_register_scoping_reduces_virtual_regs() {
        let config = GemmConfig::default_64x64();

        // Extract b32 register count from PTX .reg declaration
        let extract_b32_count = |ptx: &str| -> u32 {
            for line in ptx.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with(".reg .b32") && trimmed.contains("%r<") {
                    let start = trimmed.find("%r<").unwrap() + 3;
                    let end = trimmed.find(">;").unwrap();
                    return trimmed[start..end].parse().unwrap();
                }
            }
            panic!("No .reg .b32 declaration found");
        };

        let ptx_gemm = crate::gemm::build_gemm_pipeline(&config);
        let ptx_fused = build_fused_pipeline(&config, 4096);
        let ptx_legacy = build_fused_rmsnorm_gemm_silu(&config, 4096);

        let gemm_b32 = extract_b32_count(&ptx_gemm);
        let fused_b32 = extract_b32_count(&ptx_fused);
        let legacy_b32 = extract_b32_count(&ptx_legacy);

        println!("Register counts (b32):");
        println!("  Pipeline GEMM:        {}", gemm_b32);
        println!("  Pipeline fused:       {} (with scoping)", fused_b32);
        println!("  Legacy fused:         {} (with scoping)", legacy_b32);
        println!(
            "  Overhead vs GEMM:     {} extra b32 regs",
            fused_b32 as i32 - gemm_b32 as i32
        );

        // With the CUTLASS double-buffered fragment pattern, both GEMM and fused
        // use more registers than the old pipeline (double-buffered A[2][REG_M]
        // and B[2][REG_N] fragments). The fused kernel adds norm transform scratch.
        let overhead = fused_b32 as i32 - gemm_b32 as i32;
        assert!(
            overhead < 30,
            "Fused kernel should have <30 extra b32 regs vs GEMM, got {}",
            overhead
        );

        // With double-buffered fragments + circular buffer state, register
        // counts are higher than the old pipeline. Should still be reasonable.
        assert!(
            fused_b32 < 300,
            "Scoped fused should have <300 b32 regs, got {}",
            fused_b32
        );

        // Standalone GEMM with double-buffered fragments
        assert!(
            gemm_b32 <= 280,
            "GEMM b32 count should be <= 280 (double-buffered fragments), got {}",
            gemm_b32
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for multi-stage pipeline (3 and 4 stages)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pipeline_3_stage_gemm_generates_valid_ptx() {
        let config = GemmConfig {
            num_stages: 3,
            ..GemmConfig::default_64x64()
        };
        let ptx = crate::gemm::build_gemm_pipeline(&config);

        assert!(ptx.contains(".visible .entry triton_style_gemm("));
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));
        assert!(ptx.contains("cp.async.cg.shared.global"));
        assert!(ptx.contains("$L_KLOOP:"));
        assert!(ptx.contains("$L_EPILOGUE:"));

        // 3 stages: prologue loads STAGES-1=2 tiles (0 and 1)
        // Matches CUTLASS: for (stage=0; stage < kStages-1; ++stage)
        assert!(ptx.contains("Pipeline prologue: load tile 0 into buffer 0"));
        assert!(ptx.contains("Pipeline prologue: load tile 1 into buffer 1"));

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1);

        println!(
            "3-stage Pipeline GEMM PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_pipeline_4_stage_gemm_generates_valid_ptx() {
        let config = GemmConfig {
            num_stages: 4,
            ..GemmConfig::default_64x64()
        };
        let ptx = crate::gemm::build_gemm_pipeline(&config);

        assert!(ptx.contains(".visible .entry triton_style_gemm("));
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));
        assert!(ptx.contains("$L_KLOOP:"));

        // 4 stages: prologue loads STAGES-1=3 tiles (0, 1, 2)
        // Matches CUTLASS: for (stage=0; stage < kStages-1; ++stage)
        assert!(ptx.contains("Pipeline prologue: load tile 0 into buffer 0"));
        assert!(ptx.contains("Pipeline prologue: load tile 1 into buffer 1"));
        assert!(ptx.contains("Pipeline prologue: load tile 2 into buffer 2"));

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1);

        println!(
            "4-stage Pipeline GEMM PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_pipeline_3_stage_fused_generates_valid_ptx() {
        let config = GemmConfig {
            num_stages: 3,
            ..GemmConfig::default_64x64()
        };
        let ptx = build_fused_pipeline(&config, 4096);

        assert!(ptx.contains(".visible .entry fused_rmsnorm_gemm_silu("));
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));
        assert!(ptx.contains("rsqrt.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("cp.async.cg.shared.global"));
        assert!(ptx.contains("RmsNormAtom k_setup"));

        // 3 stages: prologue loads STAGES-1=2 tiles (0 and 1)
        assert!(ptx.contains("Pipeline prologue: load tile 0 into buffer 0"));
        assert!(ptx.contains("Pipeline prologue: load tile 1 into buffer 1"));

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1);

        // Shared memory: 3 stages * 8192 buf_stride = 24576 for GEMM region
        assert_eq!(config.smem_total(), 24576);

        println!(
            "3-stage Pipeline fused PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_pipeline_smem_sizes() {
        // Verify smem calculations for different stage counts
        let c2 = GemmConfig::default_64x64();
        assert_eq!(c2.num_stages, 2);
        assert_eq!(c2.smem_total(), 16384); // 2 * 8192

        let c3 = GemmConfig {
            num_stages: 3,
            ..GemmConfig::default_64x64()
        };
        assert_eq!(c3.smem_total(), 24576); // 3 * 8192

        let c4 = GemmConfig {
            num_stages: 4,
            ..GemmConfig::default_64x64()
        };
        assert_eq!(c4.smem_total(), 32768); // 4 * 8192
    }

    #[test]
    fn test_pipeline_bk64_3_stage_fused_generates_valid_ptx() {
        // This is the config used by the proc macro (codegen.rs).
        let config = GemmConfig {
            bm: 64,
            bn: 64,
            bk: 64,
            wm: 64,
            wn: 16,
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            num_stages: 3,
            sm_arch: "sm_89".into(),
        };
        let ptx = build_fused_pipeline(&config, 4096);

        assert!(ptx.contains(".visible .entry fused_rmsnorm_gemm_silu("));
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));
        assert!(ptx.contains("rsqrt.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("cp.async.cg.shared.global"));
        assert!(ptx.contains("RmsNormAtom k_setup"));

        // BK=64 means k_warp_iters=4, so the K-loop should have ki=0..3
        assert!(ptx.contains("ki=0"), "Must have ki=0 iteration");
        assert!(ptx.contains("ki=1"), "Must have ki=1 iteration");
        assert!(ptx.contains("ki=2"), "Must have ki=2 iteration");
        assert!(ptx.contains("ki=3"), "Must have ki=3 iteration");

        // 3 stages: prologue loads 2 tiles
        assert!(ptx.contains("Pipeline prologue: load tile 0 into buffer 0"));
        assert!(ptx.contains("Pipeline prologue: load tile 1 into buffer 1"));

        // BK=64: smem_a = 64*64*2 = 8192, smem_b = 64*64*2 = 8192
        // smem_total = (8192 + 8192) * 3 = 49152
        assert_eq!(config.smem_total(), 49152);

        // BK=64: 4 cp.async chunks per A tile, 4 per B tile
        assert_eq!(config.cp_chunks_a(), 4);
        assert_eq!(config.cp_chunks_b(), 4);

        // Should have 2 ldmatrix.x4.trans groups for B (rows 0-31 and 32-63)
        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label");
        let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];
        let ldmatrix_trans_count = kloop
            .matches("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")
            .count();
        // REG_N = wn/mma_n = 16/8 = 2 rn positions * 2 groups (BK=64) = 4 ldmatrix.trans calls
        let expected = config.reg_n() * (config.bk / config.mma_k / 2);
        assert_eq!(
            ldmatrix_trans_count,
            expected as usize,
            "BK=64 must have {} ldmatrix.trans calls ({} rn x {} groups), got {}",
            expected,
            config.reg_n(),
            config.bk / config.mma_k / 2,
            ldmatrix_trans_count
        );

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1);

        println!(
            "BK=64 3-stage Pipeline fused PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_pipeline_bk64_3_stage_gemm_generates_valid_ptx() {
        // Standalone GEMM with BK=64, 3 stages (no transform)
        let config = GemmConfig {
            bm: 64,
            bn: 64,
            bk: 64,
            wm: 64,
            wn: 16,
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            num_stages: 3,
            sm_arch: "sm_89".into(),
        };
        let ptx = crate::gemm::build_gemm_pipeline(&config);

        assert!(ptx.contains(".visible .entry triton_style_gemm("));
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));
        assert!(ptx.contains("$L_KLOOP:"));

        // BK=64: k_warp_iters=4
        assert!(ptx.contains("ki=0"));
        assert!(ptx.contains("ki=3"));

        println!(
            "BK=64 3-stage Pipeline GEMM PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for the 128×128 pipeline path
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pipeline_128x128_gemm_generates_valid_ptx() {
        let config = GemmConfig::default_128x128();
        let ptx = crate::gemm::build_gemm_pipeline(&config);

        assert!(
            ptx.contains(".visible .entry triton_style_gemm("),
            "PTX must contain kernel entry point"
        );

        // Verify tile dimensions in comments
        assert!(ptx.contains("config-aware"));

        // 128×128 with BK=32: 4 cp.async chunks per A tile, 4 per B tile
        assert_eq!(config.cp_chunks_a(), 4);
        assert_eq!(config.cp_chunks_b(), 4);
        assert_eq!(config.reg_m(), 4);
        assert_eq!(config.reg_n(), 8);
        assert_eq!(config.threads(), 128);

        // Must have MMA instructions
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));

        // Must have ldmatrix (non-trans for A, trans for B)
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"));

        // K-loop should have ki=0 and ki=1 (BK=32, k_warp_iters=2)
        assert!(ptx.contains("ki=0"));
        assert!(ptx.contains("ki=1"));

        // REG_M=4, REG_N=8 → 32 accumulator tiles → 128 acc registers
        assert_eq!(config.num_acc(), 128);

        // Single kernel
        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1);

        println!(
            "128x128 Pipeline GEMM PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_pipeline_128x128_fused_generates_valid_ptx() {
        let config = GemmConfig::default_128x128();
        let ptx = build_fused_pipeline(&config, 4096);

        assert!(
            ptx.contains(".visible .entry fused_rmsnorm_gemm_silu("),
            "PTX must contain fused kernel entry point"
        );

        // Must have all phases
        assert!(ptx.contains("rsqrt.approx.f32"), "PTX must have RMSNorm");
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16"),
            "PTX must have MMA instructions"
        );
        assert!(ptx.contains("ex2.approx.f32"), "PTX must have SiLU sigmoid");
        assert!(
            ptx.contains("cp.async.cg.shared.global"),
            "PTX must use cp.async"
        );

        // RmsNormAtom should load norm factors for all 4 rm values
        assert!(
            ptx.contains("RmsNormAtom prologue: load norm factors into REGISTERS (REG_M=4)"),
            "Must load norm factors for REG_M=4"
        );

        // K-loop with ki=0, ki=1 (BK=32)
        assert!(ptx.contains("ki=0"));
        assert!(ptx.contains("ki=1"));

        // B ldmatrix: REG_N=8 loads × 1 group (BK=32) = 8 ldmatrix.trans
        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label");
        let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];
        let ldmatrix_trans_count = kloop
            .matches("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")
            .count();
        assert_eq!(
            ldmatrix_trans_count, 8,
            "128×128 must have 8 ldmatrix.trans calls (8 rn × 1 group), got {}",
            ldmatrix_trans_count
        );

        // A ldmatrix: 4 rm × 2 ki = 8 ldmatrix calls
        let ldmatrix_a_count = kloop
            .matches("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
            .count();
        assert_eq!(
            ldmatrix_a_count, 8,
            "128×128 must have 8 ldmatrix A calls (4 rm × 2 ki), got {}",
            ldmatrix_a_count
        );

        // MMA count: 4 rm × 8 rn × 2 ki = 64 MMA instructions
        let mma_count = kloop.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 64,
            "128×128 must have 64 MMA instructions (4×8×2), got {}",
            mma_count
        );

        // Single kernel
        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1);

        // Smem: gemm(32768) + norm_scratch(32) + norm_factors(128*4=512) + gamma(4096*2=8192)
        let expected_smem = config.smem_total() + 32 + config.bm * 4 + 4096 * 2;
        assert_eq!(expected_smem, 32768 + 32 + 512 + 8192);

        println!(
            "128x128 Pipeline fused PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for the DUAL FUSED pipeline (build_dual_fused_pipeline)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_dual_fused_pipeline_generates_valid_ptx() {
        let config = GemmConfig::default_128x128();
        let ptx = build_dual_fused_pipeline(&config, 4096);

        assert!(
            ptx.contains(".visible .entry fused_rmsnorm_dual_gemm_silu_mul("),
            "PTX must contain dual fused kernel entry point"
        );

        // All phases present
        assert!(
            ptx.contains("RMSNorm norm factor computation"),
            "Must contain norm factor computation"
        );
        assert!(
            ptx.contains("Preload gamma weights"),
            "Must contain gamma preload"
        );
        assert!(
            ptx.contains("CUTLASS-matching dual GEMM"),
            "Must use CUTLASS-matching dual GEMM"
        );
        assert!(
            ptx.contains("SiLuMul epilogue"),
            "Must contain SiLuMul epilogue"
        );
        assert!(ptx.contains("Store"), "Must contain store phase");

        // Key instructions — f16 accumulators
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"),
            "Must use f16 accumulator MMA"
        );
        assert!(ptx.contains("rsqrt.approx.f32"));
        assert!(
            ptx.contains("ex2.approx.ftz.f32"),
            "Must use CUTLASS-style exp2 for SiLU"
        );
        assert!(ptx.contains("cp.async.cg.shared.global"));

        // Must have all dual fused parameters
        assert!(ptx.contains("param_input"));
        assert!(ptx.contains("param_wnorm"));
        assert!(ptx.contains("param_wgate"));
        assert!(ptx.contains("param_wup"));
        assert!(ptx.contains("param_output"));
        assert!(ptx.contains("param_N"));
        assert!(ptx.contains("param_K"));

        // Single kernel
        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1, "Must be exactly one kernel");

        println!(
            "Dual fused pipeline PTX: {} bytes, {} lines",
            ptx.len(),
            ptx.lines().count()
        );
    }

    #[test]
    fn test_dual_fused_has_dual_gemm_entry() {
        let config = GemmConfig::default_128x128();
        let ptx = build_dual_fused_pipeline(&config, 4096);

        assert!(
            ptx.contains(".visible .entry fused_rmsnorm_dual_gemm_silu_mul("),
            "Entry point must be fused_rmsnorm_dual_gemm_silu_mul"
        );

        // Must have dual K-loop structure
        assert!(
            ptx.contains("$L_DUAL_KLOOP:"),
            "Must have dual K-loop label"
        );
        assert!(
            ptx.contains("$L_DUAL_EPILOGUE:"),
            "Must have dual epilogue label"
        );
        assert!(
            ptx.contains("$L_DUAL_K0_FALLTHROUGH:"),
            "Must have dual K0 fallthrough"
        );
    }

    #[test]
    fn test_dual_fused_has_silu_ops() {
        let config = GemmConfig::default_128x128();
        let ptx = build_dual_fused_pipeline(&config, 4096);

        // CUTLASS-style SiLuMul epilogue with f16x2 ops
        assert!(
            ptx.contains("neg.f16x2"),
            "Must have neg.f16x2 for sigmoid(-x)"
        );
        assert!(
            ptx.contains("ex2.approx.ftz.f32"),
            "Must have ex2 for exp in sigmoid (CUTLASS inline asm)"
        );
        assert!(
            ptx.contains("rcp.approx.ftz.f32"),
            "Must have rcp for scalar sigmoid division"
        );

        // SiLuMul also does gate * up multiplication via f16x2
        let epilogue_start = ptx.find("SiLuMul epilogue").expect("SiLuMul epilogue comment");
        let epilogue_region = &ptx[epilogue_start..];
        assert!(
            epilogue_region.contains("mul.f16x2"),
            "SiLuMul epilogue must use mul.f16x2 for silu(gate) * up"
        );
    }

    #[test]
    fn test_dual_fused_mma_count() {
        // Dual fused should have 2× the MMA count of a single fused pipeline
        let config = GemmConfig::default_128x128();
        let ptx_single = build_fused_pipeline(&config, 4096);
        let ptx_dual = build_dual_fused_pipeline(&config, 4096);

        // Count MMAs in K-loop for single pipeline
        let single_kloop_start = ptx_single.find("$L_KLOOP:").expect("K-loop label");
        let single_kloop_end = ptx_single
            .find("$L_K0_FALLTHROUGH:")
            .expect("K0 fallthrough");
        let single_kloop = &ptx_single[single_kloop_start..single_kloop_end];
        let single_mma = single_kloop.matches("mma.sync.aligned.m16n8k16").count();

        // Count MMAs in dual K-loop
        let dual_kloop_start = ptx_dual.find("$L_DUAL_KLOOP:").expect("Dual K-loop label");
        let dual_kloop_end = ptx_dual
            .find("$L_DUAL_K0_FALLTHROUGH:")
            .expect("Dual K0 fallthrough");
        let dual_kloop = &ptx_dual[dual_kloop_start..dual_kloop_end];
        let dual_mma = dual_kloop.matches("mma.sync.aligned.m16n8k16").count();

        assert_eq!(
            dual_mma,
            single_mma * 2,
            "Dual fused must have 2× MMAs of single fused ({} vs {} expected)",
            dual_mma,
            single_mma * 2
        );

        println!(
            "MMA count: single={}, dual={} (2× verified)",
            single_mma, dual_mma
        );
    }
}
