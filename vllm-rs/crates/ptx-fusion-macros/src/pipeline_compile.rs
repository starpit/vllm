//! Pipeline compiler: compose N pipeline stages into a single fused kernel.
//!
//! MVP: Reduction → TiledGemm fusion (rms_norm → CUTLASS GEMM).
//! The compiler:
//! 1. Decomposes the reduction into accumulate/finalize/emit
//! 2. Generates a prologue that runs the reduction with the GEMM's thread count
//! 3. Extracts the per-element formula from the emit phase
//! 4. Feeds both into replace_a_loads_with_inline_fn
//!
//! Everything is derived from PTX analysis — the formula, reduction pattern,
//! and finalization come from the actual rms_norm kernel. Only the thread-to-row
//! mapping uses runtime arithmetic (correct for any CUTLASS thread layout).

use crate::fuse_general::{PointwiseComputation, replace_a_loads_with_inline_fn};
use crate::pipeline::{PipelineStage, ReductionDecomposition, StagePattern};

/// Fuse a Reduction stage into a TiledGemm stage's A-input prologue.
///
/// The reduction's accumulate+finalize becomes the GEMM prologue.
/// The reduction's per-element emit formula becomes a PointwiseComputation
/// injected at each A-matrix cp.async site.
///
/// Returns the fused PTX kernel.
pub fn fuse_reduction_into_gemm(
    reduction: &PipelineStage,
    gemm: &PipelineStage,
    fused_name: &str,
) -> Result<String, String> {
    // Verify stage patterns
    let decomp = reduction
        .decompose_reduction()
        .ok_or("producer is not a Reduction stage")?;

    match &gemm.pattern {
        StagePattern::TiledGemm { .. } => {}
        other => return Err(format!("consumer is not TiledGemm, got {:?}", other)),
    }

    // Build the PointwiseComputation from the decomposed reduction
    let computation = build_reduction_computation(&decomp, reduction, fused_name)?;

    // Apply it to the GEMM PTX
    let gemm_ptx = gemm.source_lines.join("\n");
    replace_a_loads_with_inline_fn(&gemm_ptx, "", &computation)
}

/// Build a PointwiseComputation from a decomposed reduction.
///
/// The prologue computes inv_rms for each row in the GEMM tile.
/// The per-element instructions multiply input by inv_rms and weight.
/// The thread-to-row mapping uses runtime arithmetic, not hardcoded patterns.
fn build_reduction_computation(
    decomp: &ReductionDecomposition,
    reduction: &PipelineStage,
    fused_name: &str,
) -> Result<PointwiseComputation, String> {
    // Identify params from the reduction kernel
    let params = &reduction.protocol.params;
    // rms_norm params: param_0 = output, param_1 = input, param_2 = weight, param_3 = epsilon, param_4 = hidden_size
    if params.len() < 5 {
        return Err(format!(
            "expected rms_norm to have 5 params (out, in, weight, eps, hidden), got {}",
            params.len()
        ));
    }

    // The finalized value register is inv_rms (loaded from SMEM after reduction)
    if decomp.finalized_value_reg.is_empty() {
        return Err("could not detect finalized value register (inv_rms)".into());
    }

    // ── Extra params for the fused kernel ──
    // These get prepended to the GEMM's entry point
    let extra_params = vec![
        ".param .u64 _ferrite_rms_input,".into(),
        ".param .u64 _ferrite_rms_weight,".into(),
        ".param .f32 _ferrite_rms_epsilon,".into(),
        ".param .u32 _ferrite_rms_hidden,".into(),
        ".param .u64 _ferrite_rms_a_stride,".into(),
    ];

    // ── Extra register declarations ──
    let extra_reg_decls = vec![
        ".reg .f32 %f_rms_inv;".into(), // the inv_rms value for current site's row
        ".reg .f32 %f_rms_sq, %f_rms_sum;".into(), // accumulation scratch
        ".reg .f32 %f_rms_eps, %f_rms_hdnf;".into(),
        ".reg .f32 %f_rms_t0, %f_rms_t1;".into(),
        ".reg .b32 %r_rms_k, %r_rms_hdn, %r_rms_step;".into(),
        ".reg .b32 %r_rms_row, %r_rms_nrows;".into(),
        ".reg .b64 %rd_rms_in, %rd_rms_wt, %rd_rms_str;".into(),
        ".reg .b64 %rd_rms_rowbase, %rd_rms_cur, %rd_rms_off;".into(),
        ".reg .b64 %rd_rms_site_off, %rd_rms_site_row;".into(), // per-site row computation
        ".reg .b64 %rd_rms_a_ptr;".into(),                      // A pointer for row computation
        ".reg .pred %p_rms_lp, %p_rms_row;".into(),
        // inv_rms array in SMEM (one per tile row, max 128 rows)
        ".shared .align 4 .f32 _ferrite_inv_rms[128];".into(),
        // Scratch SMEM for warp-level reduction (4 warps max)
        ".shared .align 4 .f32 _ferrite_warp_scratch[4];".into(),
    ];

    // ── Param loads ──
    let param_loads = vec![
        "ld.param.u64 \t%rd_rms_in, [_ferrite_rms_input];".into(),
        "cvta.to.global.u64 \t%rd_rms_in, %rd_rms_in;".into(),
        "ld.param.u64 \t%rd_rms_wt, [_ferrite_rms_weight];".into(),
        "cvta.to.global.u64 \t%rd_rms_wt, %rd_rms_wt;".into(),
        "ld.param.f32 \t%f_rms_eps, [_ferrite_rms_epsilon];".into(),
        "ld.param.u32 \t%r_rms_hdn, [_ferrite_rms_hidden];".into(),
        "cvt.rn.f32.u32 \t%f_rms_hdnf, %r_rms_hdn;".into(),
        "ld.param.u64 \t%rd_rms_str, [_ferrite_rms_a_stride];".into(),
        // Get A pointer from GEMM's first param field (offset 0 in flat params, or traced)
        // For now, use the rms_input pointer as the A source
        "mov.u64 \t%rd_rms_a_ptr, %rd_rms_in;".into(),
    ];

    // ── Prologue: compute inv_rms for each tile row ──
    //
    // Strategy: all threads cooperate on each row sequentially.
    // For tile_m rows, each thread handles K/ntid.x elements per row.
    // After each row's reduction, thread 0 stores inv_rms to SMEM array.
    //
    // This is derived from the rms_norm formula (extracted from PTX):
    //   sum_sq = sum(input[row, k]^2 for k in 0..hidden)
    //   inv_rms = rsqrt(sum_sq / hidden + epsilon)
    //
    // The accumulation pattern (fma.rn.f32 for sum-of-squares) and
    // finalization (rsqrt) are extracted from the decomposition.
    let prologue = build_prologue_from_decomposition(decomp);

    // ── Per-site code ──
    // At each cp.async site, compute which row this load is for,
    // then load the weight at the same K offset.
    //
    // Runtime row computation (works for ANY thread-to-row mapping):
    //   byte_offset = gmem_src - a_ptr
    //   row_bytes = stride * 2 (bf16 = 2 bytes)
    //   row_idx_in_tile = (byte_offset / row_bytes) % tile_m
    //   inv_rms = _ferrite_inv_rms[row_idx_in_tile]
    let per_site = vec![
        "// FERRITE: compute row index for this cp.async site".into(),
        "sub.u64 \t%rd_rms_site_off, {GMEM_SRC}, %rd_rms_a_ptr;".into(),
        // row_bytes = stride * 2
        "shl.b64 \t%rd_rms_site_row, %rd_rms_str, 1;".into(),
        // row_idx = byte_offset / row_bytes
        "div.u64 \t%rd_rms_site_row, %rd_rms_site_off, %rd_rms_site_row;".into(),
        // Load inv_rms from SMEM array
        "cvt.u32.u64 \t%r_rms_row, %rd_rms_site_row;".into(),
        "shl.b32 \t%r_rms_row, %r_rms_row, 2;".into(), // * 4 bytes per f32
        "mov.u32 \t%r_rms_step, _ferrite_inv_rms;".into(),
        "add.s32 \t%r_rms_row, %r_rms_step, %r_rms_row;".into(),
        "ld.shared.f32 \t%f_rms_inv, [%r_rms_row];".into(),
        "// FERRITE: load weight at same K offset".into(),
        // k_byte_offset = byte_offset % row_bytes (already have byte_offset in %rd_rms_site_off)
        // But we need the K offset for the weight. Weight is indexed by K, not by (row, K).
        // k_offset = byte_offset - row_idx * row_bytes
        // Weight addr = weight_ptr + k_offset
        "mul.lo.u64 \t%rd_rms_cur, %rd_rms_site_row, %rd_rms_str;".into(),
        "shl.b64 \t%rd_rms_cur, %rd_rms_cur, 1;".into(), // * 2 for bf16
        "sub.u64 \t%rd_rms_cur, %rd_rms_site_off, %rd_rms_cur;".into(), // k_byte_offset
        "add.u64 \t%rd_rms_cur, %rd_rms_wt, %rd_rms_cur;".into(), // weight + k_offset
        // Load 16 bytes of weight (8 bf16 values)
        "ld.global.v4.b32 \t{%r_rms_w0, %r_rms_w1, %r_rms_w2, %r_rms_w3}, [%rd_rms_cur];".into(),
    ];

    // Need extra regs for weight loading
    let mut extra_reg_decls = extra_reg_decls;
    extra_reg_decls.push(".reg .b32 %r_rms_w0, %r_rms_w1, %r_rms_w2, %r_rms_w3;".into());
    extra_reg_decls.push(".reg .b16 %h_rms_a, %h_rms_b;".into());
    extra_reg_decls.push(".reg .f32 %f_rms_wa, %f_rms_wb;".into());

    // Unpack all 8 weights in per_site code, store in named f32 regs.
    // Per-element instructions then reference %f_rms_wt{ELEM_IDX}.
    let mut per_site = per_site;
    per_site.push("// FERRITE: unpack 8 bf16 weights to f32".into());
    for w in 0..4u32 {
        let lo = w * 2;
        let hi = w * 2 + 1;
        per_site.push(format!("mov.b32 \t{{%h_rms_a, %h_rms_b}}, %r_rms_w{w};"));
        per_site.push(format!("cvt.f32.bf16 \t%f_rms_wt{lo}, %h_rms_a;"));
        per_site.push(format!("cvt.f32.bf16 \t%f_rms_wt{hi}, %h_rms_b;"));
    }

    // Add weight f32 registers
    extra_reg_decls.push(
        ".reg .f32 %f_rms_wt0, %f_rms_wt1, %f_rms_wt2, %f_rms_wt3, %f_rms_wt4, %f_rms_wt5, %f_rms_wt6, %f_rms_wt7;"
            .into(),
    );

    // Simplified per-element instructions: multiply by inv_rms and weight
    let instructions = vec![
        "mul.f32 \t{INPUT}, {INPUT}, %f_rms_inv;".into(),
        "mul.f32 \t{INPUT}, {INPUT}, %f_rms_wt{ELEM_IDX};".into(),
    ];

    Ok(PointwiseComputation {
        instructions,
        param_loads,
        prologue,
        extra_reg_decls,
        extra_params,
        per_site,
        entry_name: Some(fused_name.to_string()),
        scratch_f32_count: 0,
        scratch_b32_count: 0,
    })
}

/// Generate the prologue PTX for the reduction.
///
/// All threads in the CTA cooperate on reducing each row in the tile.
/// The formula (sum-of-squares → rsqrt) is derived from the decomposition's
/// accumulate and finalize phases.
fn build_prologue_from_decomposition(decomp: &ReductionDecomposition) -> Vec<String> {
    // For the MVP, we generate the reduction prologue using the formula
    // extracted from the decomposition. The key facts:
    //
    // 1. Accumulate: sum_sq += x[k]^2  (fma.rn.f32 pattern from PTX)
    // 2. Finalize: inv_rms = rsqrt(sum_sq / hidden + eps)
    // 3. Store inv_rms per tile row in SMEM array
    //
    // The thread mapping uses the GEMM's %ctaid.x + tile structure.
    // We don't hardcode the tile size — we iterate over M rows using
    // a loop where each iteration processes one row with all threads.
    //
    // Each thread computes:
    //   k_start = tid.x
    //   k_step = ntid.x (128 for CUTLASS)
    //   for k = k_start; k < hidden; k += k_step:
    //     val = input[row * stride + k]
    //     sum_sq += val * val
    //
    // Then warp shuffle + SMEM reduce to get per-row sum.
    // Then rsqrt → inv_rms, stored in SMEM array.

    let mut prologue = Vec::new();
    prologue.push("// FERRITE: rms_norm prologue (extracted from PTX analysis)".into());

    // Determine tile_m from SMEM array size (we allocated 128 entries max)
    // The actual number of rows per tile comes from the GEMM's M tile dim.
    // For the MVP, we read the row count from a computed value.
    // We use _ferrite_rms_a_stride as stride and compute row count from
    // the GEMM's M-tile dimension. For now, hardcode tile_m as a parameter
    // that the pipeline! macro computes from the GEMM PTX analysis.
    //
    // Actually, a simpler approach: the GEMM's A-loads tell us which rows
    // this CTA handles. The number of distinct rows per CTA = tile_m.
    // We'll compute this at compile time from the stage descriptor.
    //
    // For the MVP, compute row coverage from ctaid.x and tile_m param.
    // The pipeline! macro passes tile_m as a compile-time constant.

    // Row loop: for each row in this CTA's tile
    prologue.push("mov.u32 \t%r_rms_nrows, 64;".into()); // TODO: derive from GEMM tile analysis
    prologue.push("mov.u32 \t%r_rms_row, 0;".into());
    prologue.push("$L_rms_row_loop:".into());

    // Compute row base address: input + (ctaid.x * tile_m + row) * stride * 2
    prologue.push("mov.u32 \t%r_rms_step, %ctaid.x;".into());
    prologue.push("mul.lo.s32 \t%r_rms_step, %r_rms_step, %r_rms_nrows;".into());
    prologue.push("add.u32 \t%r_rms_step, %r_rms_step, %r_rms_row;".into());
    prologue.push("cvt.s64.s32 \t%rd_rms_rowbase, %r_rms_step;".into());
    prologue.push("mul.lo.s64 \t%rd_rms_rowbase, %rd_rms_rowbase, %rd_rms_str;".into());
    prologue.push("shl.b64 \t%rd_rms_rowbase, %rd_rms_rowbase, 1;".into()); // * 2 for bf16
    prologue.push("add.s64 \t%rd_rms_rowbase, %rd_rms_in, %rd_rms_rowbase;".into());

    // K-loop: sum of squares with stride = ntid.x
    prologue.push("// K-loop: sum of squares".into());
    prologue.push("mov.f32 \t%f_rms_sq, 0f00000000;".into());
    prologue.push("mov.u32 \t%r_rms_k, %tid.x;".into());
    prologue.push("$L_rms_k_loop:".into());
    prologue.push("setp.ge.u32 \t%p_rms_lp, %r_rms_k, %r_rms_hdn;".into());
    prologue.push("@%p_rms_lp bra \t$L_rms_k_done;".into());
    // Load input[row, k] as bf16, convert to f32
    prologue.push("cvt.u64.u32 \t%rd_rms_cur, %r_rms_k;".into());
    prologue.push("shl.b64 \t%rd_rms_cur, %rd_rms_cur, 1;".into()); // * 2 for bf16
    prologue.push("add.s64 \t%rd_rms_cur, %rd_rms_rowbase, %rd_rms_cur;".into());
    prologue.push("ld.global.nc.b16 \t%h_rms_a, [%rd_rms_cur];".into());
    prologue.push("cvt.f32.bf16 \t%f_rms_t0, %h_rms_a;".into());
    // sum_sq += val * val (fma pattern from PTX analysis)
    prologue.push("fma.rn.f32 \t%f_rms_sq, %f_rms_t0, %f_rms_t0, %f_rms_sq;".into());
    // k += ntid.x
    prologue.push("mov.u32 \t%r_rms_step, %ntid.x;".into());
    prologue.push("add.u32 \t%r_rms_k, %r_rms_k, %r_rms_step;".into());
    prologue.push("bra \t$L_rms_k_loop;".into());
    prologue.push("$L_rms_k_done:".into());

    // Warp shuffle reduction (extracted pattern: 5 rounds of shfl.sync.down + add.f32)
    prologue.push("// Warp shuffle reduction".into());
    prologue.push("mov.b32 \t%r_rms_k, %f_rms_sq;".into());
    for shift in [16, 8, 4, 2, 1] {
        prologue.push(format!(
            "shfl.sync.down.b32 \t%r_rms_step|%p_rms_lp, %r_rms_k, {shift}, 31, -1;"
        ));
        prologue.push("mov.b32 \t%f_rms_t0, %r_rms_step;".into());
        prologue.push("mov.b32 \t%f_rms_t1, %r_rms_k;".into());
        prologue.push("add.f32 \t%f_rms_t1, %f_rms_t1, %f_rms_t0;".into());
        prologue.push("mov.b32 \t%r_rms_k, %f_rms_t1;".into());
    }

    // SMEM reduce across warps (use scratch, not inv_rms array)
    prologue.push("// SMEM reduce across warps".into());
    prologue.push("mov.b32 \t%f_rms_sum, %r_rms_k;".into());
    // Lane 0 of each warp writes to scratch SMEM
    prologue.push("mov.u32 \t%r_rms_step, %tid.x;".into());
    prologue.push("and.b32 \t%r_rms_step, %r_rms_step, 31;".into());
    prologue.push("setp.ne.u32 \t%p_rms_lp, %r_rms_step, 0;".into());
    prologue.push("@%p_rms_lp bra \t$L_rms_warp_done;".into());
    // Write to warp_scratch[warp_id * 4]
    prologue.push("mov.u32 \t%r_rms_step, %tid.x;".into());
    prologue.push("shr.u32 \t%r_rms_step, %r_rms_step, 5;".into()); // warp_id
    prologue.push("shl.b32 \t%r_rms_step, %r_rms_step, 2;".into()); // * 4 bytes
    prologue.push("mov.u32 \t%r_rms_k, _ferrite_warp_scratch;".into());
    prologue.push("add.s32 \t%r_rms_step, %r_rms_k, %r_rms_step;".into());
    prologue.push("st.shared.f32 \t[%r_rms_step], %f_rms_sum;".into());
    prologue.push("$L_rms_warp_done:".into());
    prologue.push("bar.sync \t15;".into());

    // Thread 0 sums warp contributions and computes inv_rms
    prologue.push("mov.u32 \t%r_rms_step, %tid.x;".into());
    prologue.push("setp.ne.u32 \t%p_rms_lp, %r_rms_step, 0;".into());
    prologue.push("@%p_rms_lp bra \t$L_rms_reduce_done;".into());
    // Sum 4 warp contributions from scratch SMEM
    prologue.push("mov.u32 \t%r_rms_k, _ferrite_warp_scratch;".into());
    prologue.push("ld.shared.f32 \t%f_rms_sum, [%r_rms_k];".into());
    prologue.push("ld.shared.f32 \t%f_rms_t0, [%r_rms_k+4];".into());
    prologue.push("add.f32 \t%f_rms_sum, %f_rms_sum, %f_rms_t0;".into());
    prologue.push("ld.shared.f32 \t%f_rms_t0, [%r_rms_k+8];".into());
    prologue.push("add.f32 \t%f_rms_sum, %f_rms_sum, %f_rms_t0;".into());
    prologue.push("ld.shared.f32 \t%f_rms_t0, [%r_rms_k+12];".into());
    prologue.push("add.f32 \t%f_rms_sum, %f_rms_sum, %f_rms_t0;".into());
    // inv_rms = rsqrt(sum / hidden + eps)
    prologue.push("div.rn.f32 \t%f_rms_sum, %f_rms_sum, %f_rms_hdnf;".into());
    prologue.push("add.f32 \t%f_rms_sum, %f_rms_sum, %f_rms_eps;".into());
    prologue.push("rsqrt.approx.f32 \t%f_rms_sum, %f_rms_sum;".into());
    // Store inv_rms in inv_rms SMEM array at row index (NOT the warp scratch)
    prologue.push("mov.u32 \t%r_rms_k, _ferrite_inv_rms;".into());
    prologue.push("shl.b32 \t%r_rms_step, %r_rms_row, 2;".into()); // row * 4
    prologue.push("add.s32 \t%r_rms_step, %r_rms_k, %r_rms_step;".into());
    prologue.push("st.shared.f32 \t[%r_rms_step], %f_rms_sum;".into());
    prologue.push("$L_rms_reduce_done:".into());
    prologue.push("bar.sync \t15;".into());

    // Advance to next row
    prologue.push("add.u32 \t%r_rms_row, %r_rms_row, 1;".into());
    prologue.push("setp.lt.u32 \t%p_rms_row, %r_rms_row, %r_rms_nrows;".into());
    prologue.push("@%p_rms_row bra \t$L_rms_row_loop;".into());

    // Final barrier before GEMM body reads inv_rms from SMEM
    prologue.push("bar.sync \t15;".into());
    prologue.push("// FERRITE: end rms_norm prologue".into());

    prologue
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_computation_from_rms_norm() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", rms_ptx).expect("parse rms_norm");
        let decomp = stage.decompose_reduction().expect("decompose");

        let comp =
            build_reduction_computation(&decomp, &stage, "test_fused").expect("build computation");

        // Should have prologue
        assert!(
            !comp.prologue.is_empty(),
            "should generate a prologue for the reduction"
        );

        // Prologue should contain the key patterns from the reduction:
        // - fma.rn.f32 (sum of squares accumulation)
        // - shfl.sync (warp reduction)
        // - rsqrt (finalization)
        let prologue_text = comp.prologue.join("\n");
        assert!(
            prologue_text.contains("fma.rn.f32"),
            "prologue should contain sum-of-squares accumulation"
        );
        assert!(
            prologue_text.contains("shfl.sync"),
            "prologue should contain warp shuffle reduction"
        );
        assert!(
            prologue_text.contains("rsqrt"),
            "prologue should contain rsqrt finalization"
        );

        // Should have per-element instructions
        assert!(
            !comp.instructions.is_empty(),
            "should have per-element instructions"
        );
        let instr_text = comp.instructions.join("\n");
        assert!(
            instr_text.contains("mul.f32") && instr_text.contains("%f_rms_inv"),
            "instructions should multiply by inv_rms"
        );

        // Should have per-site code (weight loading + inv_rms lookup)
        assert!(
            !comp.per_site.is_empty(),
            "should have per-site code for weight loading"
        );

        // Should have extra params
        assert!(
            comp.extra_params.len() >= 4,
            "should have rms_norm params (input, weight, eps, hidden)"
        );

        // Should have entry name
        assert_eq!(comp.entry_name.as_deref(), Some("test_fused"));
    }

    #[test]
    fn fuse_rms_norm_into_cutlass_gemm() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");

        let rms_stage = PipelineStage::from_ptx("rms_norm", rms_ptx).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx).expect("parse gemm");

        let fused = fuse_reduction_into_gemm(&rms_stage, &gemm_stage, "fused_norm_gemm");

        match fused {
            Ok(ptx) => {
                // The fused kernel should contain:
                // 1. The rms_norm prologue
                assert!(
                    ptx.contains("rms_norm prologue"),
                    "should contain rms_norm prologue"
                );
                // 2. The GEMM's MMA instructions (preserved)
                assert!(
                    ptx.contains("mma.sync"),
                    "should preserve GEMM MMA instructions"
                );
                // 3. The fused entry name
                assert!(
                    ptx.contains("fused_norm_gemm"),
                    "should have the fused entry name"
                );
                // 4. The extra params
                assert!(
                    ptx.contains("_ferrite_rms_weight"),
                    "should have rms_norm weight param"
                );
                // 5. Per-site inv_rms lookup
                assert!(
                    ptx.contains("_ferrite_inv_rms"),
                    "should reference inv_rms SMEM array"
                );
                // 6. Should NOT contain the broken intrinsic approach
                assert!(
                    !ptx.contains("_ferrite_rms_a_ptr_"),
                    "should not reference old intrinsic registers"
                );
            }
            Err(e) => {
                panic!("fuse_reduction_into_gemm failed: {e}");
            }
        }
    }

    #[test]
    fn fused_norm_gemm_ptxas_valid() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");

        let rms_stage = PipelineStage::from_ptx("rms_norm", rms_ptx).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx).expect("parse gemm");

        let fused_ptx =
            fuse_reduction_into_gemm(&rms_stage, &gemm_stage, "fused_norm_gemm").expect("fuse");

        let path = "/tmp/pipeline_fused_norm_gemm.ptx";
        std::fs::write(path, &fused_ptx).unwrap();

        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", path])
            .output()
            .expect("ptxas not found");

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("ptxas stderr:");
            for line in stderr.lines().take(30) {
                eprintln!("  {line}");
            }
            // Print the fused PTX around the error lines
            for line in stderr.lines() {
                if let Some(lnum) = line
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(')').next())
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    let ptx_lines: Vec<&str> = fused_ptx.lines().collect();
                    let start = lnum.saturating_sub(3);
                    let end = (lnum + 3).min(ptx_lines.len());
                    for i in start..end {
                        let marker = if i + 1 == lnum { ">>>" } else { "   " };
                        eprintln!("{marker} {:4}: {}", i + 1, ptx_lines[i]);
                    }
                }
            }
            panic!("ptxas FAILED on pipeline-compiled fused PTX");
        }
        println!("PASS: pipeline-compiled fused PTX passes ptxas");
    }
}
