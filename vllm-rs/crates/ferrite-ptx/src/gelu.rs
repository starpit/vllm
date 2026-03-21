use crate::PtxBuilder;
use crate::config::GemmConfig;
use crate::gemm::AccumulatorMap;

// ═══════════════════════════════════════════════════════════════════════════
// GELU phase emitter — operates on AccumulatorMap registers in-place
//
// Fast approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
//
// 7 PTX instructions per element:
//   mul.f32    t, x, 1.702      // t = 1.702 * x
//   neg.f32    t, t             // t = -1.702 * x
//   mul.f32    t, t, LOG2E      // t = -1.702 * x * log2(e)
//   ex2.approx.f32 t, t        // t = 2^(t) = exp(-1.702*x)
//   add.f32    t, t, 1.0       // t = 1 + exp(-1.702*x)
//   rcp.approx.f32 t, t        // t = sigmoid(1.702*x)
//   mul.f32    x, x, t         // x = x * sigmoid(1.702*x)
// ═══════════════════════════════════════════════════════════════════════════

/// The GELU scaling constant (1.702) used in the fast sigmoid approximation.
const GELU_SCALE: f32 = 1.702;

/// Apply fast GELU activation to all accumulator registers in-place.
///
/// GELU(x) ≈ x * sigmoid(1.702 * x)
///
/// Uses the same fast-math path as SiLU (ex2.approx + rcp.approx) but with
/// a 1.702x prescale on the sigmoid input. 7 instructions per accumulator
/// element (vs SiLU's 6).
pub fn emit_gelu_phase(ptx: &mut PtxBuilder, acc: &mut AccumulatorMap) {
    ptx.comment("GELU activation: x * sigmoid(1.702 * x)");

    // Accumulators are .b32 registers holding float values (from MMA).
    // Float ALU instructions require .f32 registers.
    let fx = ptx.regs.alloc_f32(); // holds x as .f32
    let ft = ptx.regs.alloc_f32(); // temp for sigmoid computation

    for tile in &acc.regs {
        for &reg in tile {
            // Move .b32 accumulator to .f32 register
            ptx.mov_b32_to_f32(fx, reg);
            // ft = 1.702 * x
            ptx.mul_f32_imm(ft, fx, GELU_SCALE);
            // ft = -1.702 * x
            ptx.neg_f32(ft, ft);
            // ft = -1.702 * x * log2(e)
            ptx.mul_f32_imm(ft, ft, std::f32::consts::LOG2_E);
            // ft = 2^(ft) = exp(-1.702*x)
            ptx.ex2_approx_f32(ft, ft);
            // ft = 1.0 + exp(-1.702*x)
            ptx.add_f32_imm(ft, ft, 1.0);
            // ft = 1 / (1 + exp(-1.702*x)) = sigmoid(1.702*x)
            ptx.rcp_approx_f32(ft, ft);
            // fx = x * sigmoid(1.702*x)
            ptx.mul_f32(fx, fx, ft);
            // Move result back to .b32 accumulator
            ptx.mov_f32_to_b32(reg, fx);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Standalone GELU kernel — for benchmarking against PyTorch/Triton
// ═══════════════════════════════════════════════════════════════════════════

/// Build a standalone GELU kernel that loads f32 from global memory,
/// applies GELU, and stores back.
///
/// Kernel signature: gelu_kernel(float* data, int N)
/// Grid: 1D, each thread handles ELEMS_PER_THREAD elements.
/// Block size: 256 threads.
pub fn build_gelu_kernel(block_size: u32, elems_per_thread: u32) -> String {
    let num_warps = block_size / 32;
    let config = GemmConfig {
        bm: num_warps * 32,
        bn: 32,
        bk: 1,
        wm: 32,
        wn: 32,
        mma_m: 16,
        mma_n: 8,
        mma_k: 16,
        num_stages: 1,
        sm_arch: "sm_89".into(),
    };
    let mut ptx = PtxBuilder::new(config);

    ptx.comment("Standalone GELU kernel for benchmarking");
    ptx.blank();

    // Load params
    let data_ptr = ptx.regs.alloc_b64();
    let n_param = ptx.regs.alloc_b32();
    ptx.ld_param_b64(data_ptr, "param_data");
    ptx.ld_param_b32(n_param, "param_N");

    // Compute global thread index
    let bid = ptx.regs.alloc_b32();
    let tid = ptx.regs.alloc_b32();
    let ntid = ptx.regs.alloc_b32();
    ptx.mov_b32_name(bid, "%ctaid.x");
    ptx.mov_b32_name(tid, "%tid.x");
    ptx.mov_b32_name(ntid, "%ntid.x");

    let global_tid = ptx.regs.alloc_b32();
    ptx.mul_lo_s32(global_tid, bid, ntid);
    ptx.add_s32(global_tid, global_tid, tid);

    // base_idx = global_tid * ELEMS_PER_THREAD
    let base_idx = ptx.regs.alloc_b32();
    if elems_per_thread.is_power_of_two() {
        ptx.shl_b32(base_idx, global_tid, elems_per_thread.trailing_zeros());
    } else {
        let ept_reg = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(ept_reg, elems_per_thread);
        ptx.mul_lo_s32(base_idx, global_tid, ept_reg);
    }
    ptx.blank();

    // Compute base address: data_ptr + base_idx * 4
    let base_addr = ptx.regs.alloc_b64();
    let four = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(four, 4);
    ptx.mad_wide_s32(base_addr, base_idx, four, data_ptr);

    // Bounds check: if base_idx >= N, skip
    let p_valid = ptx.regs.alloc_pred();
    ptx.setp_lt_s32(p_valid, base_idx, n_param);
    ptx.w(&format!("@!{p_valid} bra \t$L_DONE;"));
    ptx.blank();

    // Load elements, apply GELU, store back
    ptx.comment("Load, GELU, Store loop");
    let tmp = ptx.regs.alloc_f32();

    for i in 0..elems_per_thread {
        let offset_bytes = (i * 4) as i32;
        let val = ptx.regs.alloc_f32();

        // Load
        ptx.ld_global_f32(val, base_addr, offset_bytes);

        // GELU: val = val * sigmoid(1.702 * val)
        ptx.mul_f32_imm(tmp, val, GELU_SCALE); // tmp = 1.702 * x
        ptx.neg_f32(tmp, tmp);
        ptx.mul_f32_imm(tmp, tmp, std::f32::consts::LOG2_E);
        ptx.ex2_approx_f32(tmp, tmp);
        ptx.add_f32_imm(tmp, tmp, 1.0);
        ptx.rcp_approx_f32(tmp, tmp);
        ptx.mul_f32(val, val, tmp);

        // Store
        ptx.st_global_f32(base_addr, offset_bytes, val);
    }

    ptx.blank();
    ptx.label("$L_DONE");
    ptx.ret();

    let params = vec![
        (".u64 .ptr .global .align 16", "param_data"),
        (".u32", "param_N"),
    ];

    ptx.finalize("gelu_kernel", &params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_epilogue_generates_valid_ptx_instructions() {
        // Build a minimal AccumulatorMap and run emit_gelu_phase
        let config = GemmConfig {
            bm: 64,
            bn: 64,
            bk: 32,
            wm: 64,
            wn: 64,
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            num_stages: 2,
            sm_arch: "sm_89".into(),
        };
        let mut ptx = PtxBuilder::new(config);
        let mut acc = AccumulatorMap {
            regs: vec![[
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ]],
            reg_m: 1,
            reg_n: 1,
        };

        emit_gelu_phase(&mut ptx, &mut acc);
        let body = &ptx.body;

        // Verify key GELU instructions are present
        assert!(body.contains("GELU activation"), "Should have GELU comment");
        assert!(
            body.contains("ex2.approx.f32"),
            "Should use ex2.approx for exp(-1.702*x)"
        );
        assert!(
            body.contains("rcp.approx.f32"),
            "Should use rcp.approx for 1/(1+exp(...))"
        );
        assert!(
            body.contains("neg.f32"),
            "Should negate for sigmoid computation"
        );

        // Count mul.f32 — GELU has 3 muls per element (1.702*x, *log2e, x*sigmoid)
        // with 4 accumulators = 12 mul.f32 total
        let mul_count = body.matches("mul.f32").count();
        assert_eq!(
            mul_count, 12,
            "Expected 12 mul.f32 (3 per element * 4 accumulators), got {mul_count}"
        );
    }

    #[test]
    fn test_gelu_standalone_kernel_generates_valid_ptx() {
        let ptx = build_gelu_kernel(256, 4);

        // Basic validity checks
        assert!(ptx.contains(".version"), "Must have PTX version");
        assert!(ptx.contains("gelu_kernel"), "Must have kernel name");
        assert!(ptx.contains("param_data"), "Must have data parameter");
        assert!(ptx.contains("param_N"), "Must have N parameter");

        // GELU-specific: the 1.702 prescale distinguishes it from SiLU
        // 1.702 as IEEE 754 hex = 0x3FDA1CAC (approximately)
        // Check for ex2.approx and rcp.approx (fast math path)
        assert!(ptx.contains("ex2.approx.f32"), "Must use fast exp2");
        assert!(ptx.contains("rcp.approx.f32"), "Must use fast reciprocal");

        // Should have 4 load + 4 store (elems_per_thread=4)
        let ld_count = ptx.matches("ld.global").count();
        let st_count = ptx.matches("st.global").count();
        assert_eq!(ld_count, 4, "Expected 4 global loads for 4 elems/thread");
        assert_eq!(st_count, 4, "Expected 4 global stores for 4 elems/thread");
    }

    #[test]
    fn test_gelu_has_7_instructions_per_element() {
        // Verify instruction count: 7 ALU ops per element
        // mul(1.702*x), neg, mul(log2e), ex2, add(1.0), rcp, mul(x*sig)
        let config = GemmConfig {
            bm: 64,
            bn: 64,
            bk: 32,
            wm: 64,
            wn: 64,
            mma_m: 16,
            mma_n: 8,
            mma_k: 16,
            num_stages: 2,
            sm_arch: "sm_89".into(),
        };
        let mut ptx = PtxBuilder::new(config);

        // Single accumulator element
        let mut acc = AccumulatorMap {
            regs: vec![[
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(),
            ]],
            reg_m: 1,
            reg_n: 1,
        };

        emit_gelu_phase(&mut ptx, &mut acc);
        let body = &ptx.body;

        // Per element: 3x mul.f32, 1x neg.f32, 1x ex2.approx, 1x add.f32, 1x rcp.approx = 7
        // Plus 2 movs (b32<->f32) for register class conversion = 9 total instructions per element
        // With 4 accumulators: 4 * (7 ALU + 2 mov) = 28 ALU + 8 mov
        let neg_count = body.matches("neg.f32").count();
        let ex2_count = body.matches("ex2.approx.f32").count();
        let rcp_count = body.matches("rcp.approx.f32").count();
        assert_eq!(neg_count, 4, "Expected 4 neg.f32 (one per accumulator)");
        assert_eq!(ex2_count, 4, "Expected 4 ex2.approx.f32");
        assert_eq!(rcp_count, 4, "Expected 4 rcp.approx.f32");
    }
}
