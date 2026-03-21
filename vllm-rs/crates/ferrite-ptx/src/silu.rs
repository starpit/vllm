use crate::PtxBuilder;
use crate::config::GemmConfig;
use crate::gemm::AccumulatorMap;

// ═══════════════════════════════════════════════════════════════════════════
// SiLU phase emitter — operates on AccumulatorMap registers in-place
// ═══════════════════════════════════════════════════════════════════════════

/// Apply SiLU activation (x * sigmoid(x)) to all accumulator registers in-place.
///
/// SiLU(x) = x / (1 + exp(-x))
///
/// In PTX, using fast math approximations:
///   neg.f32    t, x          // t = -x
///   mul.f32    t, t, LOG2E   // t = -x * log2(e)
///   ex2.approx.f32 t, t      // t = 2^(-x*log2e) = exp(-x)
///   add.f32    t, t, 1.0     // t = 1 + exp(-x)
///   rcp.approx.f32 t, t      // t = 1 / (1 + exp(-x)) = sigmoid(x)
///   mul.f32    x, x, t       // x = x * sigmoid(x)
pub fn emit_silu_phase(ptx: &mut PtxBuilder, acc: &mut AccumulatorMap) {
    ptx.comment("SiLU activation: x * sigmoid(x)");

    // Accumulators are .b32 registers holding float values (from MMA).
    // Float ALU instructions require .f32 registers, so we move to .f32,
    // apply SiLU, and move back.
    let fx = ptx.regs.alloc_f32(); // holds x as .f32
    let ft = ptx.regs.alloc_f32(); // temp for sigmoid computation

    for tile in &acc.regs {
        for &reg in tile {
            // Move .b32 accumulator to .f32 register
            ptx.mov_b32_to_f32(fx, reg);
            // ft = -x
            ptx.neg_f32(ft, fx);
            // ft = -x * log2(e)   [log2(e) = 1.4426950408...]
            ptx.mul_f32_imm(ft, ft, std::f32::consts::LOG2_E);
            // ft = 2^(ft) = exp(-x)
            ptx.ex2_approx_f32(ft, ft);
            // ft = 1.0 + exp(-x)
            ptx.add_f32_imm(ft, ft, 1.0);
            // ft = 1 / (1 + exp(-x)) = sigmoid(x)
            ptx.rcp_approx_f32(ft, ft);
            // fx = x * sigmoid(x)
            ptx.mul_f32(fx, fx, ft);
            // Move result back to .b32 accumulator
            ptx.mov_f32_to_b32(reg, fx);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Standalone SiLU kernel — for benchmarking against Triton
// ═══════════════════════════════════════════════════════════════════════════

/// Build a standalone SiLU kernel that loads f32 from global memory,
/// applies SiLU, and stores back.
///
/// Kernel signature: silu_kernel(float* data, int N)
/// Grid: 1D, each thread handles ELEMS_PER_THREAD elements.
/// Block size: 256 threads.
pub fn build_silu_kernel(block_size: u32, elems_per_thread: u32) -> String {
    // We use GemmConfig only for sm_arch and finalize; the GEMM-specific
    // fields are irrelevant for this kernel.
    // Set config so threads() = block_size.
    // threads = (bm/wm) * (bn/wn) * 32, so we need (bm/wm)*(bn/wn) = block_size/32
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

    ptx.comment("Standalone SiLU kernel for benchmarking");
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

    // Load elements, apply SiLU, store back
    ptx.comment("Load, SiLU, Store loop");
    let tmp = ptx.regs.alloc_f32();

    for i in 0..elems_per_thread {
        let offset_bytes = (i * 4) as i32;
        let val = ptx.regs.alloc_f32();

        // Load
        ptx.ld_global_f32(val, base_addr, offset_bytes);

        // SiLU: val = val * sigmoid(val)
        ptx.neg_f32(tmp, val);
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

    // Custom finalize without .reqntid (we set threads() via config.bm)
    let params = vec![
        (".u64 .ptr .global .align 16", "param_data"),
        (".u32", "param_N"),
    ];

    ptx.finalize("silu_kernel", &params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_phase_emits_correct_instructions() {
        let config = GemmConfig {
            bm: 64, bn: 64, bk: 32, wm: 64, wn: 64,
            mma_m: 16, mma_n: 8, mma_k: 16, num_stages: 2,
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
        emit_silu_phase(&mut ptx, &mut acc);
        let body = &ptx.body;

        assert!(body.contains("neg.f32"), "SiLU must negate for -x");
        assert!(body.contains("ex2.approx.f32"), "SiLU must use ex2.approx for exp(-x)");
        assert!(body.contains("rcp.approx.f32"), "SiLU must use rcp.approx for 1/(1+exp(-x))");
        assert!(body.contains("add.f32"), "SiLU must add 1.0");
        assert!(body.contains("mul.f32"), "SiLU must multiply x * sigmoid(x)");
    }

    #[test]
    fn test_silu_phase_6_ops_per_element() {
        let config = GemmConfig {
            bm: 64, bn: 64, bk: 32, wm: 64, wn: 64,
            mma_m: 16, mma_n: 8, mma_k: 16, num_stages: 2,
            sm_arch: "sm_89".into(),
        };
        let mut ptx = PtxBuilder::new(config);
        // Single element (1 tile with 4 regs)
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
        emit_silu_phase(&mut ptx, &mut acc);
        let body = &ptx.body;

        // Per element: neg, mul(log2e), ex2, add(1.0), rcp, mul(x*sig) = 6 ALU ops
        // 4 elements total
        assert_eq!(body.matches("neg.f32").count(), 4);
        assert_eq!(body.matches("ex2.approx.f32").count(), 4);
        assert_eq!(body.matches("rcp.approx.f32").count(), 4);
        assert_eq!(body.matches("add.f32").count(), 4);
        // mul.f32 = 2 per element (log2e multiply + final multiply)
        assert_eq!(body.matches("mul.f32").count(), 8);
    }

    #[test]
    fn test_silu_standalone_kernel_valid_ptx() {
        let ptx = build_silu_kernel(256, 4);

        assert!(ptx.contains(".version"), "Must have PTX version");
        assert!(ptx.contains("silu_kernel"), "Must have entry point name");
        assert!(ptx.contains("param_data"), "Must have data parameter");
        assert!(ptx.contains("param_N"), "Must have N parameter");
        assert!(ptx.contains(".visible .entry silu_kernel("), "Must have visible entry");
    }

    #[test]
    fn test_silu_standalone_has_ex2_and_rcp() {
        let ptx = build_silu_kernel(256, 4);
        assert!(ptx.contains("ex2.approx.f32"), "Must use fast exp2");
        assert!(ptx.contains("rcp.approx.f32"), "Must use fast reciprocal");
    }

    #[test]
    fn test_silu_standalone_load_store_counts() {
        let ptx = build_silu_kernel(256, 8);
        let ld_count = ptx.matches("ld.global.f32").count();
        let st_count = ptx.matches("st.global.f32").count();
        assert_eq!(ld_count, 8, "Expected 8 global loads for 8 elems/thread");
        assert_eq!(st_count, 8, "Expected 8 global stores for 8 elems/thread");
    }

    #[test]
    fn test_silu_standalone_has_bounds_check() {
        let ptx = build_silu_kernel(256, 4);
        assert!(ptx.contains("setp.lt.s32"), "Must have bounds check predicate");
        assert!(ptx.contains("$L_DONE"), "Must have done label for bounds skip");
    }
}
