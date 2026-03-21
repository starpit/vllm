use super::config::GemmConfig;
use super::gemm::{
    AccumulatorMap, CpAsyncLoader, NormalizedLoader, emit_a_global_addrs, emit_b_global_addrs,
    emit_cpasync_swizzle, emit_gemm_setup, emit_gemm_with_loaders, emit_store_c,
};
use super::tile::emit_norm_factor_computation;
use super::{PtxBuilder, Reg};

// ===========================================================================
// Fused RMSNorm -> GEMM -> SiLU megakernel
//
// One kernel launch, zero HBM round-trips between ops:
//   1. RMSNorm: compute per-row norm factors (one pass over input rows)
//   2. GEMM: K-loop with NormalizedLoader for A (loads input, normalizes,
//      writes f16 to smem) and CpAsyncLoader for B
//   3. SiLU: transform accumulators in-place (pure ALU)
//   4. Store: write activated f32 results to global
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
    let gemm_smem = c.smem_total(); // 16384
    let norm_scratch_off = gemm_smem;
    let norm_factors_off = norm_scratch_off + 32;
    let _total_smem = norm_factors_off + bm * 4;

    ptx.comment("=== Fused RMSNorm -> GEMM -> SiLU megakernel ===");
    ptx.comment(&format!(
        "BM={}, BN={}, BK={}, threads={}",
        bm, c.bn, bk, threads
    ));
    ptx.comment(&format!("hidden_size={}", hidden_size));
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
    let eps: f32 = 1e-6;
    let norm_factors_base = emit_norm_factor_computation(
        &mut ptx,
        input_ptr,
        setup.block_row,
        setup.tid,
        setup.smem_base,
        norm_factors_off,
        hidden_size,
        eps,
    );

    // ===============================================================
    // Phase B: GEMM with NormalizedLoader for A, CpAsyncLoader for B
    // ===============================================================
    ptx.comment("=== Phase B: GEMM with normalized A-tile loading ===");
    ptx.blank();

    // cp.async swizzle addresses
    let (a_cp_off, b_cp_off) = emit_cpasync_swizzle(&mut ptx, setup.tid);

    // A global addresses (pointing to input, not a pre-normalized buffer)
    let (ga0, ga1, a_tid_row, a_tid_col) =
        emit_a_global_addrs(&mut ptx, setup.block_row, k_param, input_ptr, setup.tid);

    // B global addresses
    let (gb0, gb1) = emit_b_global_addrs(&mut ptx, setup.block_col, n_param, wgemm_ptr, setup.tid);

    // RMSNorm weight pointer for this thread's K column
    let wnorm_thread = ptx.regs.alloc_b64();
    {
        let two_reg = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(two_reg, 2);
        ptx.mad_wide_s32(wnorm_thread, a_tid_col, two_reg, wnorm_ptr);
    }
    ptx.blank();

    // Row IDs for chunk 0 and chunk 1
    let grow_a0 = ptx.regs.alloc_b32();
    ptx.add_s32(grow_a0, setup.block_row, a_tid_row);
    let grow_a1 = ptx.regs.alloc_b32();
    ptx.add_s32_imm(grow_a1, grow_a0, 32);

    // Create NormalizedLoader
    let a_loader = NormalizedLoader {
        wnorm_ptr: wnorm_thread,
        norm_factors_base,
        row_id_chunk0: grow_a0,
        row_id_chunk1: grow_a1,
        block_row: setup.block_row,
    };
    let b_loader = CpAsyncLoader;

    // The wnorm pointer needs to advance by BK*2 each K-loop iteration.
    // We capture wnorm_thread by value (it's a Reg = Copy).
    let bk_val = c.bk;
    let wnorm_reg = wnorm_thread;
    let advance_wnorm = move |ptx: &mut PtxBuilder| {
        ptx.add_s64_imm(wnorm_reg, wnorm_reg, (bk_val * 2) as i64);
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
        Some(&advance_wnorm),
    );

    // ===============================================================
    // Phase C: SiLU activation on accumulators
    // ===============================================================
    ptx.comment("=== Phase C: SiLU activation on accumulators ===");
    super::silu::emit_silu_phase(&mut ptx, &mut acc);
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
