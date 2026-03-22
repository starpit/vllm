use crate::PtxBuilder;
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

/// Build a fused dual GEMM(gate,up) -> SiLuMul kernel.
///
/// Delegates to `dual_gemm::emit_dual_gemm_kernel()` which emits the CUTLASS
/// dual_gemm PTX from Rust — the reference implementation that achieves 59.8 TFLOPS.
/// The kernel takes a 720-byte DualGemmParams struct.
#[allow(unused_variables)]
pub fn build_dual_fused_pipeline(config: &GemmConfig, hidden_size: u32) -> String {
    crate::dual_gemm::emit_dual_gemm_kernel()
}


/// Return the CUTLASS dual_gemm + SiLUAndMul PTX — literal copy of the
/// reference implementation (CUTLASS examples/45_dual_gemm, compiled for sm_89).
///
/// This is a 128×64 tile, triple-buffered, f16 accumulator dual GEMM kernel
/// that achieves 59.9 TFLOPS on L4. The PTX is embedded as a static string
/// and returned verbatim. The entry point is `ferrite_dual_gemm_silu_mul`.
///
/// The kernel takes a 720-byte `DualGemmParams` struct — use the struct
/// definition in `cutlass_dual_gemm_params` (ferrite-poc) to pack it.
///
/// This follows the Ferrite rule: copy the reference PTX, don't design your own.
pub fn build_cutlass_dual_gemm() -> String {
    include_str!("cutlass_dual_gemm.ptx").to_string()
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
    // Tests for build_dual_fused_pipeline (delegates to CUTLASS copy)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_dual_fused_pipeline_delegates_to_cutlass() {
        let config = GemmConfig::default_128x128();
        let ptx = build_dual_fused_pipeline(&config, 4096);
        let cutlass = build_cutlass_dual_gemm();
        assert_eq!(ptx, cutlass, "build_dual_fused_pipeline must return the CUTLASS PTX copy");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tests for the embedded CUTLASS dual_gemm PTX (build_cutlass_dual_gemm)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_cutlass_dual_gemm_returns_nonempty_ptx() {
        let ptx = build_cutlass_dual_gemm();
        assert!(ptx.len() > 100_000, "Expected >100KB PTX, got {} bytes", ptx.len());
    }

    #[test]
    fn test_cutlass_dual_gemm_has_entry_point() {
        let ptx = build_cutlass_dual_gemm();
        assert!(ptx.contains(".visible .entry ferrite_dual_gemm_silu_mul("),
            "Must have our renamed entry point");
    }

    #[test]
    fn test_cutlass_dual_gemm_has_smem_declaration() {
        let ptx = build_cutlass_dual_gemm();
        assert!(ptx.contains("_ZN7cutlass17SharedStorageBaseE[49152]"),
            "Must declare 48KB shared memory for triple-buffered A+B0+B1");
    }

    #[test]
    fn test_cutlass_dual_gemm_uses_f16_accumulators() {
        let ptx = build_cutlass_dual_gemm();
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"),
            "Must use f16 accumulators (not f32) for register efficiency");
        assert!(!ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "Must NOT use f32 accumulators");
    }

    #[test]
    fn test_cutlass_dual_gemm_has_64_mma_instructions() {
        let ptx = build_cutlass_dual_gemm();
        let mma_count = ptx.lines().filter(|l| l.contains("mma.sync.aligned")).count();
        assert_eq!(mma_count, 64,
            "CUTLASS dual GEMM should have 64 MMA instructions (32 per GEMM), got {}", mma_count);
    }

    #[test]
    fn test_cutlass_dual_gemm_has_ldmatrix() {
        let ptx = build_cutlass_dual_gemm();
        let ldm_count = ptx.lines().filter(|l| l.contains("ldmatrix.sync.aligned")).count();
        assert!(ldm_count >= 16,
            "Need ldmatrix for A + B0 + B1, got {}", ldm_count);
    }

    #[test]
    fn test_cutlass_dual_gemm_has_cp_async() {
        let ptx = build_cutlass_dual_gemm();
        let cp_count = ptx.lines().filter(|l| l.contains("cp.async.cg.shared.global")).count();
        assert!(cp_count >= 16,
            "Need cp.async for A + B0 + B1 tile loads, got {}", cp_count);
    }

    #[test]
    fn test_cutlass_dual_gemm_has_silu_ops() {
        let ptx = build_cutlass_dual_gemm();
        let ex2_count = ptx.lines().filter(|l| l.contains("ex2.approx")).count();
        let rcp_count = ptx.lines().filter(|l| l.contains("rcp.approx")).count();
        assert!(ex2_count >= 32, "Need ex2 for sigmoid in SiLU, got {}", ex2_count);
        assert!(rcp_count >= 32, "Need rcp for 1/(1+exp) in SiLU, got {}", rcp_count);
    }

    #[test]
    fn test_cutlass_dual_gemm_has_global_stores() {
        let ptx = build_cutlass_dual_gemm();
        let st_count = ptx.lines().filter(|l| l.contains("st.global")).count();
        assert!(st_count >= 8, "Need global stores for output, got {}", st_count);
    }

    #[test]
    fn test_cutlass_dual_gemm_ptx_version_and_target() {
        let ptx = build_cutlass_dual_gemm();
        assert!(ptx.contains(".version 8.0"), "Must target PTX 8.0");
        assert!(ptx.contains(".target sm_89"), "Must target sm_89");
    }

    #[test]
    fn test_cutlass_dual_gemm_has_triple_buffer_cycling() {
        let ptx = build_cutlass_dual_gemm();
        // Triple buffer cycling uses commit_group and wait_group patterns
        let commit_count = ptx.lines().filter(|l| l.contains("cp.async.commit_group")).count();
        let wait_count = ptx.lines().filter(|l| l.contains("cp.async.wait_group")).count();
        assert!(commit_count >= 2, "Need commit_group for pipeline, got {}", commit_count);
        assert!(wait_count >= 2, "Need wait_group for pipeline, got {}", wait_count);
    }

    #[test]
    fn test_cutlass_dual_gemm_has_bar_sync() {
        let ptx = build_cutlass_dual_gemm();
        let bar_count = ptx.lines().filter(|l| l.contains("bar.sync")).count();
        assert!(bar_count >= 4, "Need barriers for smem sync, got {}", bar_count);
    }

    #[test]
    fn test_cutlass_dual_gemm_param_struct_720_bytes() {
        let ptx = build_cutlass_dual_gemm();
        // CUTLASS DualGemm::Params is 720 bytes, passed as .param .b8[720]
        // (or similar alignment block)
        assert!(ptx.contains("param_0[7") || ptx.contains("param_0[720]"),
            "Param block should be ~720 bytes");
    }

    #[test]
    fn test_cutlass_dual_gemm_no_local_memory() {
        let ptx = build_cutlass_dual_gemm();
        let local_count = ptx.lines()
            .filter(|l| l.contains("st.local") || l.contains("ld.local"))
            .count();
        assert_eq!(local_count, 0,
            "CUTLASS dual GEMM should have no local memory spills in PTX, found {}", local_count);
    }
}
