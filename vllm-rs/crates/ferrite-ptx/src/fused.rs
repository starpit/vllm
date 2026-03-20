use crate::PtxBuilder;
use crate::config::GemmConfig;
use crate::gemm::{
    CpAsyncLoader, RmsNormTransform,
    emit_gemm_setup, emit_cpasync_swizzle, emit_a_global_addrs,
    emit_b_global_addrs, emit_gemm_with_loaders, emit_store_c,
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
//   [0..8191]       Double-buffered A tiles (cp.async raw input — SAME as standalone)
//   [8192..16383]   Double-buffered B tiles (cp.async weights — SAME as standalone)
//   [16384..16415]  Norm scratch (32 bytes)
//   [16416..16671]  Norm factors (64 f32 = 256 bytes)
//   [16672..24863]  Gamma weights (4096 f16 = 8192 bytes, preloaded)
//   Total: ~25KB
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
    let gemm_smem = c.smem_total();               // 16384 (2 bufs * (A + B))
    let norm_scratch_off = gemm_smem;              // 16384
    let norm_factors_off = norm_scratch_off + 32;  // 16416
    let gamma_smem_off = norm_factors_off + bm * 4; // 16672
    let gamma_smem_bytes = hidden_size * 2;         // 8192 for hidden=4096
    let _total_smem = gamma_smem_off + gamma_smem_bytes; // ~24864

    ptx.comment("=== Fused RMSNorm -> GEMM -> SiLU megakernel (FragmentTransform) ===");
    ptx.comment(&format!("BM={}, BN={}, BK={}, threads={}", bm, c.bn, bk, threads));
    ptx.comment(&format!("hidden_size={}", hidden_size));
    ptx.comment(&format!("smem: A[0..8191] B[8192..16383] (SAME as standalone GEMM)"));
    ptx.comment(&format!("      norm_scratch[{norm_scratch_off}] norm_factors[{norm_factors_off}]"));
    ptx.comment(&format!("      gamma[{gamma_smem_off}..{}]", gamma_smem_off + gamma_smem_bytes));
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
    // Allocate norm_factors_base OUTSIDE the scope so it survives pop_scope()
    let norm_factors_base = ptx.regs.alloc_b32();
    ptx.push_scope();  // --- norm reduction temporaries scope ---
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
    ptx.pop_scope();   // --- norm reduction temps freed ---

    // ===============================================================
    // Phase 1.5: Preload ALL gamma weights into shared memory
    // ===============================================================
    // Allocate gamma_smem_base OUTSIDE the scope so it survives pop_scope()
    let gamma_smem_base = ptx.regs.alloc_b32();
    ptx.add_s32_imm(gamma_smem_base, setup.smem_base, gamma_smem_off as i32);

    ptx.push_scope();  // --- gamma preload temporaries scope ---
    ptx.comment("=== Phase 1.5: Preload gamma weights into smem ===");
    ptx.comment(&format!("{} f16 = {} bytes, {} threads x {} bytes each",
        hidden_size, gamma_smem_bytes, threads, gamma_smem_bytes / threads));
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
            ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
        ];
        ptx.ld_global_v4_b32(data, gamma_global_addr, (v * 16) as i32);
        ptx.st_shared_v4_b32(gamma_smem_dst, (v * 16) as i32, data);
    }
    ptx.bar_sync(0);
    ptx.blank();
    ptx.pop_scope();   // --- gamma preload temps freed ---

    // ===============================================================
    // Phase B: GEMM with FragmentTransform (CpAsyncLoader for BOTH A and B)
    // ===============================================================
    ptx.comment("=== Phase B: GEMM with FragmentTransform (cp.async BOTH A and B) ===");
    ptx.blank();

    // cp.async swizzle addresses
    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle(&mut ptx, setup.tid);

    // A global addresses (pointing to raw input — cp.async will load these)
    let (ga0, ga1, _a_tid_row, _a_tid_col) = emit_a_global_addrs(
        &mut ptx, setup.block_row, k_param, input_ptr, setup.tid,
    );

    // B global addresses
    let (gb0, gb1) = emit_b_global_addrs(
        &mut ptx, setup.block_col, n_param, wgemm_ptr, setup.tid,
    );

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
        &mut ptx, c, &setup,
        &a_loader, ga0, ga1, a_cp_off,
        &b_loader, gb0, gb1, b_cp_off,
        n_param, k_param,
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
pub fn build_fused_pipeline(config: &GemmConfig, hidden_size: u32) -> String {
    use crate::atoms::{CpAsyncCopy, RmsNormAtom, Mma16816, SiLuEpilogue, EpilogueAtom};
    use crate::pipeline::MainloopPipeline;
    use crate::gemm::{
        emit_gemm_setup, emit_cpasync_swizzle, emit_a_global_addrs,
        emit_b_global_addrs, emit_store_c,
    };

    let mut ptx = PtxBuilder::new(config.clone());
    let c = &ptx.config.clone();

    let bm = c.bm;
    let bk = c.bk;
    let threads = c.threads();

    // Shared memory layout
    let gemm_smem = c.smem_total();               // 16384
    let norm_scratch_off = gemm_smem;              // 16384
    let norm_factors_off = norm_scratch_off + 32;  // 16416
    let gamma_smem_off = norm_factors_off + bm * 4; // 16672
    let _gamma_smem_bytes = hidden_size * 2;

    ptx.comment("=== Fused RMSNorm -> GEMM -> SiLU (MainloopPipeline) ===");
    ptx.comment(&format!("BM={}, BN={}, BK={}, threads={}", bm, c.bn, bk, threads));
    ptx.comment(&format!("hidden_size={}", hidden_size));
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
    // Allocate norm_factors_base OUTSIDE the scope so it survives pop_scope()
    let norm_factors_base = ptx.regs.alloc_b32();
    ptx.push_scope();  // --- norm reduction temporaries scope ---
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
    ptx.pop_scope();   // --- norm reduction temps freed ---

    // Phase 1.5: Preload ALL gamma weights into shared memory
    // Allocate gamma_smem_base OUTSIDE the scope so it survives pop_scope()
    let gamma_smem_base = ptx.regs.alloc_b32();
    ptx.add_s32_imm(gamma_smem_base, setup.smem_base, gamma_smem_off as i32);

    ptx.push_scope();  // --- gamma preload temporaries scope ---
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
            ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
            ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
        ];
        ptx.ld_global_v4_b32(data, gamma_global_addr, (v * 16) as i32);
        ptx.st_shared_v4_b32(gamma_smem_dst, (v * 16) as i32, data);
    }
    ptx.bar_sync(0);
    ptx.blank();
    ptx.pop_scope();   // --- gamma preload temps freed ---

    // Phase B: GEMM with RmsNormAtom k_setup via MainloopPipeline
    ptx.comment("=== Phase B: GEMM with RmsNormAtom via MainloopPipeline ===");
    ptx.blank();

    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle(&mut ptx, setup.tid);

    let (ga0, ga1, _, _) = emit_a_global_addrs(
        &mut ptx, setup.block_row, k_param, input_ptr, setup.tid,
    );
    let (gb0, gb1) = emit_b_global_addrs(
        &mut ptx, setup.block_col, n_param, wgemm_ptr, setup.tid,
    );

    // Create the RmsNormAtom k_setup
    let k_offset_reg = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(k_offset_reg, 0);

    let transform = RmsNormAtom::new(
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

    let pipeline = MainloopPipeline::new(2);
    let result = pipeline.emit(
        &mut ptx, c, &setup,
        &CpAsyncCopy,
        &CpAsyncCopy,
        &transform,
        &Mma16816,
        ga0, ga1, a_cp_off,
        gb0, gb1, b_cp_off,
        n_param, k_param,
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
        assert!(ptx.contains(".visible .entry fused_rmsnorm_gemm_silu("),
            "PTX must contain exactly one kernel entry point");

        // Verify it has all phases
        assert!(ptx.contains("RMSNorm norm factor computation"),
            "PTX must contain norm factor computation phase");
        assert!(ptx.contains("Preload gamma weights"),
            "PTX must contain gamma preload phase");
        assert!(ptx.contains("GEMM with FragmentTransform"),
            "PTX must contain GEMM phase with fragment transform");
        assert!(ptx.contains("SiLU activation"),
            "PTX must contain SiLU activation phase");
        assert!(ptx.contains("Store"),
            "PTX must contain store phase");

        // Verify single kernel (only one .entry)
        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1,
            "Must generate exactly ONE kernel, got {}", entry_count);

        // Verify key instructions are present
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"),
            "PTX must use tensor core MMA instructions");
        assert!(ptx.contains("rsqrt.approx.f32"),
            "PTX must compute rsqrt for RMSNorm");
        assert!(ptx.contains("ex2.approx.f32"),
            "PTX must use fast exp for SiLU sigmoid");
        assert!(ptx.contains("cp.async.cg.shared.global"),
            "PTX must use cp.async for A and B tiles");

        // Verify RmsNorm transform is applied (check the actual comment text)
        assert!(ptx.contains("RmsNorm f16x2 transform"),
            "PTX must contain RmsNorm f16x2 transform");

        // Verify all parameters
        assert!(ptx.contains("param_input"));
        assert!(ptx.contains("param_wnorm"));
        assert!(ptx.contains("param_wgemm"));
        assert!(ptx.contains("param_output"));
        assert!(ptx.contains("param_N"));
        assert!(ptx.contains("param_K"));

        println!("Fused megakernel PTX: {} bytes, {} lines",
            ptx.len(), ptx.lines().count());
    }

    #[test]
    fn test_fused_kernel_has_no_extra_global_stores_between_phases() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        let silu_start = ptx.find("SiLU activation").unwrap();
        let store_start = ptx.find("Store output").unwrap();
        let between = &ptx[silu_start..store_start];

        assert!(!between.contains("st.global"),
            "SiLU phase must not write to global memory — intermediates stay in registers");
    }

    #[test]
    fn test_fragment_transform_in_kloop() {
        // Verify the K-loop applies transform between ldmatrix and MMA
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label must exist");
        let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").expect("K0 fallthrough must exist");
        let kloop = &ptx[kloop_start..kloop_end];

        // Verify ldmatrix A comes before RmsNorm transform
        let ldmatrix_pos = kloop.find("ldmatrix.sync.aligned.m8n8.x4.shared.b16").expect("ldmatrix A");
        let transform_pos = kloop.find("RmsNorm f16x2 transform").expect("transform comment");
        assert!(ldmatrix_pos < transform_pos, "ldmatrix must come BEFORE transform");

        // Verify transform comes before MMA
        let mma_pos = kloop.find("mma.sync.aligned.m16n8k16").expect("MMA instruction");
        assert!(transform_pos < mma_pos, "transform must come BEFORE MMA");

        // Verify cp.async is used (same as standalone GEMM)
        assert!(kloop.contains("cp.async.cg.shared.global"),
            "K-loop must use cp.async for tile loading");
    }

    #[test]
    fn test_no_extra_smem_temp_buffer() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        assert!(!ptx.contains("A_temp"),
            "New approach should NOT use smem_A_temp buffer");
    }

    #[test]
    fn test_gamma_preload() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_rmsnorm_gemm_silu(&config, 4096);

        assert!(ptx.contains("Preload gamma weights"),
            "Must contain gamma preload phase");
        assert!(ptx.contains("ld.global.v4.b32"),
            "Gamma preload should use vectorized loads");
        assert!(ptx.contains("st.shared.v4.b32"),
            "Gamma preload should use vectorized shared stores");
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

        assert!(ptx.contains(".visible .entry triton_style_gemm("),
            "PTX must contain kernel entry point");
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"),
            "PTX must use tensor core MMA");
        assert!(ptx.contains("cp.async.cg.shared.global"),
            "PTX must use cp.async");
        assert!(ptx.contains("ldmatrix.sync.aligned"),
            "PTX must use ldmatrix");

        // Must have K-loop structure
        assert!(ptx.contains("$L_KLOOP:"), "Must have K-loop label");
        assert!(ptx.contains("$L_EPILOGUE:"), "Must have epilogue label");
        assert!(ptx.contains("$L_K0_FALLTHROUGH:"), "Must have K0 fallthrough");

        // Must have proper header
        assert!(ptx.starts_with(".version 8.6\n.target sm_89\n"));
        assert!(ptx.contains(".reqntid 128"));

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1, "Must be exactly one kernel");

        println!("Pipeline GEMM PTX: {} bytes, {} lines",
            ptx.len(), ptx.lines().count());
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
        assert!(!kloop.contains("mul.rn.f16x2"),
            "Standalone GEMM must NOT have f16x2 transforms in K-loop");
    }

    #[test]
    fn test_pipeline_fused_generates_valid_ptx() {
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_pipeline(&config, 4096);

        assert!(ptx.contains(".visible .entry fused_rmsnorm_gemm_silu("),
            "PTX must contain fused kernel entry point");

        // All phases present
        assert!(ptx.contains("RMSNorm norm factor computation"),
            "Must contain norm factor computation");
        assert!(ptx.contains("Preload gamma weights"),
            "Must contain gamma preload");
        assert!(ptx.contains("MainloopPipeline"),
            "Must use MainloopPipeline");
        assert!(ptx.contains("SiLU activation"),
            "Must contain SiLU activation");

        // Key instructions
        assert!(ptx.contains("mma.sync.aligned.m16n8k16"));
        assert!(ptx.contains("rsqrt.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("cp.async.cg.shared.global"));

        // RmsNormAtom k_setup between ldmatrix and MMA
        assert!(ptx.contains("RmsNormAtom k_setup"),
            "Must contain RmsNormAtom k_setup");

        // No A_temp buffer
        assert!(!ptx.contains("A_temp"));

        let entry_count = ptx.matches(".visible .entry").count();
        assert_eq!(entry_count, 1, "Must be exactly one kernel");

        println!("Pipeline fused PTX: {} bytes, {} lines",
            ptx.len(), ptx.lines().count());
    }

    #[test]
    fn test_pipeline_fused_transform_ordering() {
        // Verify ldmatrix -> transform -> MMA ordering in pipeline fused kernel
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_pipeline(&config, 4096);

        let kloop_start = ptx.find("$L_KLOOP:").expect("K-loop label");
        let kloop_end = ptx.find("$L_K0_FALLTHROUGH:").expect("K0 fallthrough");
        let kloop = &ptx[kloop_start..kloop_end];

        let ksetup_pos = kloop.find("RmsNormAtom k_setup").expect("k_setup comment");
        let ldmatrix_pos = kloop.find("ldmatrix.sync.aligned.m8n8.x4.shared.b16").expect("ldmatrix A");
        let mul_pos = kloop.find("mul.rn.f16x2").expect("f16x2 transform");
        let mma_pos = kloop.find("mma.sync.aligned.m16n8k16").expect("MMA instruction");

        assert!(ksetup_pos < ldmatrix_pos, "k_setup (gamma load) must come BEFORE ldmatrix");
        assert!(ldmatrix_pos < mul_pos, "ldmatrix must come BEFORE transform mul");
        assert!(mul_pos < mma_pos, "transform must come BEFORE MMA");
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
        assert!(kloop.contains("mul.rn.f16x2"),
            "RmsNormAtom must use packed f16x2 multiply");

        // Must NOT unpack to f32 (cvt.f32.f16) in the transform
        // (cvt instructions in the kloop would indicate f32 unpack)
        let transform_start = kloop.find("RmsNormAtom k_setup").expect("transform");
        let next_mma = kloop[transform_start..].find("mma.sync").expect("next MMA");
        let transform_region = &kloop[transform_start..transform_start + next_mma];
        assert!(!transform_region.contains("cvt.f32.f16"),
            "RmsNormAtom must NOT unpack f16 to f32");
    }

    #[test]
    fn test_pipeline_fused_norm_factors_in_registers() {
        // RmsNormAtom must load norm factors in prologue (before K-loop)
        // and keep them in registers for the entire loop
        let config = GemmConfig::default_64x64();
        let ptx = build_fused_pipeline(&config, 4096);

        // The prologue should load norm factors
        assert!(ptx.contains("RmsNormAtom prologue: load norm factors into REGISTERS"),
            "Must load norm factors into registers in prologue");
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
        assert_eq!(gemm_bars, fused_bars,
            "Fused kernel must have SAME number of barriers as standalone GEMM in K-loop ({} vs {})",
            fused_bars, gemm_bars);
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
        println!("  Overhead vs GEMM:     {} extra b32 regs",
            fused_b32 as i32 - gemm_b32 as i32);

        // With scoping, fused should be close to GEMM baseline.
        // The extra registers are for: norm_factors_base, gamma_smem_base,
        // k_offset_reg, RmsNormAtom scratch (~15 regs). Should be <30 extra.
        let overhead = fused_b32 as i32 - gemm_b32 as i32;
        assert!(overhead < 30,
            "Fused kernel should have <30 extra b32 regs vs GEMM, got {}",
            overhead);

        // Fused should be well under the pre-scoping count of ~308
        assert!(fused_b32 < 260,
            "Scoped fused should have <260 b32 regs (was ~308 before scoping), got {}",
            fused_b32);

        // Standalone GEMM should be unchanged (no scopes in Identity path)
        assert!(gemm_b32 <= 227,
            "GEMM b32 count should be <= 227 (interleaved ki saves regs), got {}", gemm_b32);
    }
}
