use crate::PtxBuilder;
use crate::gemm::AccumulatorMap;

// ═══════════════════════════════════════════════════════════════════════════
// SiLuMul epilogue — silu(gate) * up on a wide GEMM accumulator
//
// For LLaMA's MLP: the wide GEMM computes [gate, up] with concatenated
// weights [w_gate; w_up]. The accumulator has 2× the normal N-tiles.
// This epilogue splits at the midpoint, applies SiLU to the gate half,
// and multiplies gate × up elementwise.
//
// AccumulatorMap layout for wide GEMM (e.g., reg_m=4, reg_n=8):
//   regs[rm * reg_n + rn] where:
//     rn = 0..reg_n/2  → gate tiles (N-tiles for w_gate output)
//     rn = reg_n/2..reg_n  → up tiles (N-tiles for w_up output)
//
// After epilogue, the gate tiles contain silu(gate) * up.
// The up tiles are no longer needed (they can be overwritten).
// The store phase should only write reg_n/2 tiles per row.
// ═══════════════════════════════════════════════════════════════════════════

/// Apply SiLU(gate) * up to a wide GEMM accumulator in-place.
///
/// Precondition: `acc.reg_n` must be even (gate and up halves).
/// After this call, the first half of each row contains silu(gate) * up.
/// The second half is consumed and should not be stored.
pub fn emit_silu_mul_phase(ptx: &mut PtxBuilder, acc: &mut AccumulatorMap) {
    assert!(
        acc.reg_n % 2 == 0,
        "SiLuMul epilogue requires even reg_n (got {})",
        acc.reg_n
    );

    let half_n = acc.reg_n / 2;

    ptx.comment(&format!(
        "SiLuMul epilogue: silu(gate) * up (reg_m={}, reg_n={}, half_n={})",
        acc.reg_m, acc.reg_n, half_n
    ));

    // Scratch registers for SiLU computation
    let fx = ptx.regs.alloc_f32(); // holds gate value as .f32
    let ft = ptx.regs.alloc_f32(); // temp for sigmoid
    let fu = ptx.regs.alloc_f32(); // holds up value as .f32

    for rm in 0..acc.reg_m {
        for rn in 0..half_n {
            let gate_idx = (rm * acc.reg_n + rn) as usize;
            let up_idx = (rm * acc.reg_n + half_n + rn) as usize;

            let gate_tile = acc.regs[gate_idx];
            let up_tile = acc.regs[up_idx];

            // Process each of the 4 registers in the tile
            for i in 0..4 {
                let gate_reg = gate_tile[i];
                let up_reg = up_tile[i];

                // Move gate accumulator to .f32
                ptx.mov_b32_to_f32(fx, gate_reg);

                // SiLU(gate): fx = gate * sigmoid(gate)
                ptx.neg_f32(ft, fx);
                ptx.mul_f32_imm(ft, ft, std::f32::consts::LOG2_E);
                ptx.ex2_approx_f32(ft, ft);
                ptx.add_f32_imm(ft, ft, 1.0);
                ptx.rcp_approx_f32(ft, ft);
                ptx.mul_f32(fx, fx, ft); // fx = silu(gate)

                // Multiply by up: fx = silu(gate) * up
                ptx.mov_b32_to_f32(fu, up_reg);
                ptx.mul_f32(fx, fx, fu);

                // Store result back to gate tile register
                ptx.mov_f32_to_b32(gate_reg, fx);
            }
        }
    }

    // After epilogue: gate tiles (first half) contain silu(gate) * up.
    // Update reg_n to reflect that only the first half is valid output.
    acc.reg_n = half_n;
    // Truncate regs to only keep the gate (output) tiles
    let mut new_regs = Vec::new();
    for rm in 0..acc.reg_m {
        for rn in 0..half_n {
            new_regs.push(acc.regs[(rm * (half_n * 2) + rn) as usize]);
        }
    }
    acc.regs = new_regs;
}

/// SiLuMul epilogue atom for the pipeline.
pub struct SiLuMulEpilogue;

impl crate::atoms::EpilogueAtom for SiLuMulEpilogue {
    fn emit_epilogue(&self, ptx: &mut PtxBuilder, acc: &mut AccumulatorMap) {
        emit_silu_mul_phase(ptx, acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GemmConfig;
    use crate::gemm::AccumulatorMap;
    use crate::Reg;

    fn default_config() -> GemmConfig {
        GemmConfig::default_64x64()
    }

    fn make_wide_acc(ptx: &mut PtxBuilder, reg_m: u32, reg_n: u32) -> AccumulatorMap {
        let mut regs = Vec::new();
        for _ in 0..(reg_m * reg_n) {
            regs.push([
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ]);
        }
        AccumulatorMap {
            regs,
            reg_m,
            reg_n,
        }
    }

    #[test]
    fn test_silu_mul_epilogue_requires_even_reg_n() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 2, 8); // even reg_n = 8
        emit_silu_mul_phase(&mut ptx, &mut acc);
        // Should not panic
        assert_eq!(acc.reg_n, 4, "Output reg_n should be halved");
    }

    #[test]
    #[should_panic(expected = "even reg_n")]
    fn test_silu_mul_epilogue_panics_odd_reg_n() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 2, 7); // odd reg_n = 7
        emit_silu_mul_phase(&mut ptx, &mut acc);
    }

    #[test]
    fn test_silu_mul_epilogue_halves_reg_n() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 4, 8);
        assert_eq!(acc.reg_n, 8);
        assert_eq!(acc.regs.len(), 32); // 4 * 8

        emit_silu_mul_phase(&mut ptx, &mut acc);

        assert_eq!(acc.reg_n, 4, "reg_n must be halved after SiLuMul");
        assert_eq!(acc.regs.len(), 16, "regs must be truncated to 4 * 4 = 16");
    }

    #[test]
    fn test_silu_mul_epilogue_has_silu_instructions() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 1, 2); // minimal: 1 gate tile + 1 up tile
        emit_silu_mul_phase(&mut ptx, &mut acc);
        let body = &ptx.body;

        // SiLU ops per element: neg, mul(log2e), ex2, add(1), rcp, mul(x*sig)
        assert!(body.contains("neg.f32"), "Must have neg for sigmoid(-x)");
        assert!(body.contains("ex2.approx.f32"), "Must have ex2 for exp");
        assert!(body.contains("rcp.approx.f32"), "Must have rcp for 1/(1+exp)");
    }

    #[test]
    fn test_silu_mul_epilogue_has_multiply() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 1, 2);
        emit_silu_mul_phase(&mut ptx, &mut acc);
        let body = &ptx.body;

        // Must have mul for both silu(gate) and silu(gate)*up
        let mul_count = body.matches("mul.f32").count();
        // Per element: 2 mul for silu + 1 mul for *up = 3 mul
        // 4 elements per tile, 1 gate tile: 3 * 4 = 12
        assert_eq!(mul_count, 12, "Expected 12 mul.f32 (3 per element × 4 elements), got {}", mul_count);
    }

    #[test]
    fn test_silu_mul_epilogue_instruction_counts_2x4() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 2, 4); // 2 gate tiles + 2 up tiles per row, 2 rows
        emit_silu_mul_phase(&mut ptx, &mut acc);
        let body = &ptx.body;

        // reg_m=2, half_n=2, so 2*2=4 gate tiles, each with 4 elements = 16 elements
        let neg_count = body.matches("neg.f32").count();
        let ex2_count = body.matches("ex2.approx.f32").count();
        let rcp_count = body.matches("rcp.approx.f32").count();
        let mul_count = body.matches("mul.f32").count();

        assert_eq!(neg_count, 16, "16 neg.f32 (1 per element × 16 elements)");
        assert_eq!(ex2_count, 16, "16 ex2.approx.f32");
        assert_eq!(rcp_count, 16, "16 rcp.approx.f32");
        assert_eq!(mul_count, 48, "48 mul.f32 (3 per element × 16 elements)");
    }

    #[test]
    fn test_silu_mul_epilogue_no_f16x2_ops() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 1, 2);
        emit_silu_mul_phase(&mut ptx, &mut acc);

        assert!(
            !ptx.body.contains("f16x2"),
            "SiLuMul operates on f32 accumulators, must NOT use f16x2"
        );
    }

    #[test]
    fn test_silu_mul_epilogue_output_preserves_tile_count() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 2, 6); // 3 gate + 3 up per row
        emit_silu_mul_phase(&mut ptx, &mut acc);
        // Output should have reg_m * half_n = 2 * 3 = 6 tiles
        assert_eq!(acc.regs.len(), 6);
        assert_eq!(acc.reg_m, 2);
        assert_eq!(acc.reg_n, 3);
    }

    #[test]
    fn test_silu_mul_epilogue_comment() {
        let mut ptx = PtxBuilder::new(default_config());
        let mut acc = make_wide_acc(&mut ptx, 2, 8);
        emit_silu_mul_phase(&mut ptx, &mut acc);

        assert!(ptx.body.contains("SiLuMul epilogue"), "Must have descriptive comment");
    }
}
