use super::{PtxBuilder, Reg};
use super::config::GemmConfig;

// ═══════════════════════════════════════════════════════════════════════════
// RMSNorm phase emitter — operates on registers, writes normalized values
// to shared memory as A-tile for a subsequent GEMM.
// ═══════════════════════════════════════════════════════════════════════════

/// Phase emitter context: describes where the input row lives and where
/// the normalized output goes.
pub struct RmsNormPhaseConfig {
    /// Number of f16 elements per row (hidden_size)
    pub hidden_size: u32,
    /// Threads per block (must match GEMM block size)
    pub threads: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

/// Emit the RMSNorm phase into a PtxBuilder.
///
/// This phase:
///   1. Loads a row of f16 from global memory (input_ptr + row * hidden_size)
///   2. Converts to f32, computes sum-of-squares via warp shuffle reduction
///   3. Computes rsqrt(mean(x^2) + eps)
///   4. Multiplies by f16 weight, converts back to f16
///   5. Stores normalized f16 values to shared memory (for use as GEMM A-tile)
///
/// `input_ptr` — .b64 register pointing to the start of the input row (global)
/// `weight_ptr` — .b64 register pointing to the weight vector (global)
/// `smem_out` — .b32 register with shared memory base for output tile
/// `eps` — epsilon value
pub fn emit_rmsnorm_phase(
    ptx: &mut PtxBuilder,
    input_ptr: Reg,
    weight_ptr: Reg,
    smem_out: Reg,
    tid: Reg,
    lane: Reg,
    warp_id: Reg,
    cfg: &RmsNormPhaseConfig,
) {
    let elems_per_thread = cfg.hidden_size / cfg.threads;
    let num_warps = cfg.threads / 32;

    ptx.comment("═══ RMSNorm phase: load, normalize, store to smem ═══");
    ptx.blank();

    // ── Step 1: Load f16 elements from global memory, convert to f32 ──
    ptx.comment("Load input row (f16) and convert to f32");
    let mut f32_vals = Vec::new();

    // Compute thread's base offset: tid * elems_per_thread * 2 (bytes)
    let base_byte_off = ptx.regs.alloc_b64();
    let tid_x_ept = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_x_ept, tid, elems_per_thread.trailing_zeros());
    // Byte offset = tid_x_ept * 2
    let two = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(two, 2);
    ptx.mul_wide_s32(base_byte_off, tid_x_ept, two);
    let input_addr = ptx.regs.alloc_b64();
    ptx.add_s64(input_addr, input_ptr, base_byte_off);

    for i in 0..elems_per_thread {
        let raw = ptx.regs.alloc_b32();
        ptx.ld_global_b32(raw, input_addr, (i * 2) as i32);
        // raw is a 16-bit value loaded into a 32-bit register (zero-extended by ld.b16
        // — but we need ld.global.b16). Actually ld.global.b32 loads 4 bytes.
        // For f16, we should load 2 bytes. Let's load pairs as b32 instead.
        // Each b32 holds two f16 values packed.
        f32_vals.push(raw); // placeholder
    }
    // Actually, let's redo this properly: load f16 values two at a time (packed in b32),
    // then extract and convert each half.
    f32_vals.clear();

    // Load pairs of f16 as b32
    let num_pairs = elems_per_thread / 2;
    let mut raw_pairs = Vec::new();
    for i in 0..num_pairs {
        let raw = ptx.regs.alloc_b32();
        ptx.ld_global_b32(raw, input_addr, (i * 4) as i32);
        raw_pairs.push(raw);
    }

    // Convert each pair: extract low/high f16, convert to f32
    for &pair in &raw_pairs {
        // Low half: cvt.f32.f16 extracts bits[15:0]
        let lo = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{lo}, {pair};"));
        f32_vals.push(lo);

        // High half: need to shift right by 16 first
        let hi_bits = ptx.regs.alloc_b32();
        ptx.shr_u32(hi_bits, pair, 16);
        let hi = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{hi}, {hi_bits};"));
        f32_vals.push(hi);
    }
    ptx.blank();

    // ── Step 2: Compute partial sum of squares ──
    ptx.comment("Compute partial sum-of-squares");
    let partial_sum = ptx.regs.alloc_f32();
    ptx.mov_f32_imm(partial_sum, 0.0);
    for &v in &f32_vals {
        // fma: partial_sum = v * v + partial_sum
        ptx.w(&format!("fma.rn.f32 \t{partial_sum}, {v}, {v}, {partial_sum};"));
    }
    ptx.blank();

    // ── Step 3: Warp-level reduction via butterfly shuffle ──
    ptx.comment("Warp reduction: butterfly shuffle for sum-of-squares");
    // shfl.sync.bfly reduces within a warp of 32 threads
    let sum_reg = ptx.regs.alloc_f32();
    ptx.mov_b32_to_f32(sum_reg, partial_sum);
    // Actually partial_sum is already f32. We need a b32 alias for shfl.
    let sum_b32 = ptx.regs.alloc_b32();
    ptx.mov_f32_to_b32(sum_b32, partial_sum);

    for offset in [16, 8, 4, 2, 1] {
        let shfl_result = ptx.regs.alloc_b32();
        ptx.w(&format!(
            "shfl.sync.bfly.b32 \t{shfl_result}, {sum_b32}, {offset}, 0x1F, 0xFFFFFFFF;"
        ));
        // Convert shfl result to f32 and add
        let shfl_f32 = ptx.regs.alloc_f32();
        ptx.mov_b32_to_f32(shfl_f32, shfl_result);
        let new_sum = ptx.regs.alloc_f32();
        ptx.add_f32(new_sum, partial_sum, shfl_f32);
        ptx.mov_b32_to_f32(partial_sum, new_sum); // update partial_sum
        // Update b32 alias
        ptx.mov_f32_to_b32(sum_b32, new_sum);
    }
    ptx.blank();

    // ── Step 4: Cross-warp reduction via shared memory ──
    ptx.comment("Cross-warp reduction via shared memory");
    // Use a scratch area in shared memory. We'll use the last 128 bytes of smem
    // (after the output tile area). Each warp lane 0 writes its sum.
    let smem_base = ptx.regs.alloc_b32();
    ptx.mov_b32_name(smem_base, "global_smem");

    // Scratch at smem_base + 0 (before the output tile; we'll use first num_warps*4 bytes)
    // Actually use a high offset to avoid collision: hidden_size*2 bytes for output
    let scratch_off = cfg.hidden_size * 2; // output tile size in bytes
    let scratch_addr = ptx.regs.alloc_b32();
    ptx.add_s32_imm(scratch_addr, smem_base, scratch_off as i32);

    // Lane 0 of each warp writes to scratch[warp_id]
    let p_lane0 = ptx.regs.alloc_pred();
    ptx.w(&format!("setp.eq.s32 \t{p_lane0}, {lane}, 0;"));

    let warp_scratch = ptx.regs.alloc_b32();
    let warp_off = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_off, warp_id, 2); // warp_id * 4
    ptx.add_s32(warp_scratch, scratch_addr, warp_off);

    // Predicated store
    ptx.w(&format!("@{p_lane0} st.shared.b32 \t[{warp_scratch}], {sum_b32};"));

    ptx.bar_sync(0);

    // Thread 0 reads all warp sums and reduces
    let p_tid0 = ptx.regs.alloc_pred();
    ptx.w(&format!("setp.eq.s32 \t{p_tid0}, {tid}, 0;"));

    let total_sum = ptx.regs.alloc_f32();
    ptx.mov_f32_imm(total_sum, 0.0);

    // Load and sum all warp contributions (unrolled)
    for w in 0..num_warps {
        let ws = ptx.regs.alloc_b32();
        ptx.w(&format!("@{p_tid0} ld.shared.b32 \t{ws}, [{scratch_addr}+{}];", w * 4));
        let ws_f32 = ptx.regs.alloc_f32();
        ptx.mov_b32_to_f32(ws_f32, ws);
        ptx.w(&format!("@{p_tid0} add.f32 \t{total_sum}, {total_sum}, {ws_f32};"));
    }

    // Compute mean = total_sum / hidden_size
    let mean_val = ptx.regs.alloc_f32();
    ptx.w(&format!(
        "@{p_tid0} mul.f32 \t{mean_val}, {total_sum}, 0F{:08X};",
        (1.0f32 / cfg.hidden_size as f32).to_bits()
    ));

    // Add epsilon
    ptx.w(&format!(
        "@{p_tid0} add.f32 \t{mean_val}, {mean_val}, 0F{:08X};",
        cfg.eps.to_bits()
    ));

    // rsqrt
    let scale = ptx.regs.alloc_f32();
    ptx.w(&format!("@{p_tid0} rsqrt.approx.f32 \t{scale}, {mean_val};"));

    // Broadcast scale to all threads via shared memory
    let scale_b32 = ptx.regs.alloc_b32();
    ptx.mov_f32_to_b32(scale_b32, scale);
    ptx.w(&format!("@{p_tid0} st.shared.b32 \t[{scratch_addr}], {scale_b32};"));
    ptx.bar_sync(0);

    // All threads read the scale
    let scale_shared = ptx.regs.alloc_b32();
    ptx.w(&format!("ld.shared.b32 \t{scale_shared}, [{scratch_addr}];"));
    let scale_all = ptx.regs.alloc_f32();
    ptx.mov_b32_to_f32(scale_all, scale_shared);
    ptx.blank();

    // ── Step 5: Load weight, multiply x * scale * weight, convert to f16, store to smem ──
    ptx.comment("Apply normalization: y = x * scale * weight, store f16 to smem");

    // Load weight (f16 pairs)
    let weight_addr = ptx.regs.alloc_b64();
    ptx.add_s64(weight_addr, weight_ptr, base_byte_off);

    let mut weight_f32 = Vec::new();
    for i in 0..num_pairs {
        let raw = ptx.regs.alloc_b32();
        ptx.ld_global_b32(raw, weight_addr, (i * 4) as i32);
        let lo = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{lo}, {raw};"));
        weight_f32.push(lo);
        let hi_bits = ptx.regs.alloc_b32();
        ptx.shr_u32(hi_bits, raw, 16);
        let hi = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{hi}, {hi_bits};"));
        weight_f32.push(hi);
    }

    // Compute smem output address
    let smem_out_addr = ptx.regs.alloc_b32();
    let tid_byte_off_smem = ptx.regs.alloc_b32();
    ptx.shl_b32(tid_byte_off_smem, tid_x_ept, 1); // tid * elems_per_thread * 2 bytes
    ptx.add_s32(smem_out_addr, smem_out, tid_byte_off_smem);

    // For each pair: y = x * scale * w, pack two f16 into b32, store
    for i in 0..num_pairs as usize {
        let idx0 = i * 2;
        let idx1 = i * 2 + 1;

        // y0 = x0 * scale * w0  (use fma: y0 = x0 * scale then y0 * w0 ...
        // actually simpler: t = x * scale; y = t * w)
        let t0 = ptx.regs.alloc_f32();
        ptx.mul_f32(t0, f32_vals[idx0], scale_all);
        let y0 = ptx.regs.alloc_f32();
        ptx.mul_f32(y0, t0, weight_f32[idx0]);

        let t1 = ptx.regs.alloc_f32();
        ptx.mul_f32(t1, f32_vals[idx1], scale_all);
        let y1 = ptx.regs.alloc_f32();
        ptx.mul_f32(y1, t1, weight_f32[idx1]);

        // Convert to f16 and pack
        let h0 = ptx.regs.alloc_b32();
        ptx.w(&format!("cvt.rn.f16.f32 \t{h0}, {y0};"));
        let h1 = ptx.regs.alloc_b32();
        ptx.w(&format!("cvt.rn.f16.f32 \t{h1}, {y1};"));

        // Pack: result = h0 | (h1 << 16)
        let h1_shifted = ptx.regs.alloc_b32();
        ptx.shl_b32(h1_shifted, h1, 16);
        let packed = ptx.regs.alloc_b32();
        ptx.or_b32(packed, h0, h1_shifted);

        // Store to shared memory
        ptx.w(&format!("st.shared.b32 \t[{smem_out_addr}+{}], {packed};", i * 4));
    }
    ptx.blank();
    ptx.bar_sync(0);
    ptx.comment("═══ RMSNorm phase complete ═══");
}

// ═══════════════════════════════════════════════════════════════════════════
// Standalone RMSNorm kernel — for benchmarking against Triton
// ═══════════════════════════════════════════════════════════════════════════

/// Build a standalone RMSNorm kernel.
///
/// Kernel signature: rmsnorm_kernel(half* output, half* input, half* weight,
///                                   float eps, int hidden_size, int num_rows)
///
/// Grid: 1D, one block per row.
/// Block: `block_size` threads (128 or 256).
/// Each thread handles hidden_size/block_size elements.
///
/// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
pub fn build_rmsnorm_kernel(block_size: u32, hidden_size: u32) -> String {
    let num_warps = block_size / 32;
    let elems_per_thread = hidden_size / block_size;
    assert!(hidden_size % block_size == 0, "hidden_size must be divisible by block_size");
    assert!(elems_per_thread % 2 == 0, "elems_per_thread must be even for f16 pair loading");

    let config = GemmConfig {
        bm: num_warps * 32, bn: 32, bk: 1,
        wm: 32, wn: 32,
        mma_m: 16, mma_n: 8, mma_k: 16,
        num_stages: 1,
        sm_arch: "sm_89".into(),
    };
    let mut ptx = PtxBuilder::new(config);

    ptx.comment("Standalone RMSNorm kernel for benchmarking");
    ptx.comment(&format!("block_size={}, hidden_size={}, elems_per_thread={}",
                          block_size, hidden_size, elems_per_thread));
    ptx.blank();

    // ── Load parameters ──
    let output_ptr = ptx.regs.alloc_b64();
    let input_ptr = ptx.regs.alloc_b64();
    let weight_ptr = ptx.regs.alloc_b64();
    let eps_param = ptx.regs.alloc_b32(); // f32 loaded as b32
    ptx.ld_param_b64(output_ptr, "param_output");
    ptx.ld_param_b64(input_ptr, "param_input");
    ptx.ld_param_b64(weight_ptr, "param_weight");
    ptx.ld_param_b32(eps_param, "param_eps");
    ptx.blank();

    // ── Thread/block indices ──
    let bid = ptx.regs.alloc_b32();
    let tid = ptx.regs.alloc_b32();
    ptx.mov_b32_name(bid, "%ctaid.x");
    ptx.mov_b32_name(tid, "%tid.x");

    let lane = ptx.regs.alloc_b32();
    ptx.and_b32(lane, tid, 31);
    let warp_id = ptx.regs.alloc_b32();
    ptx.shr_u32(warp_id, tid, 5);
    ptx.blank();

    // ── Compute row base address ──
    // row_offset = bid * hidden_size * 2 (bytes, f16)
    ptx.comment("Compute row base address");
    let row_byte_off = ptx.regs.alloc_b64();
    let hs_bytes = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(hs_bytes, hidden_size * 2);
    ptx.mul_wide_s32(row_byte_off, bid, hs_bytes);

    let row_input_ptr = ptx.regs.alloc_b64();
    ptx.add_s64(row_input_ptr, input_ptr, row_byte_off);
    let row_output_ptr = ptx.regs.alloc_b64();
    ptx.add_s64(row_output_ptr, output_ptr, row_byte_off);
    ptx.blank();

    // ── Compute thread base offset within the row ──
    // thread_offset = tid * elems_per_thread * 2 (bytes)
    let thread_byte_off = ptx.regs.alloc_b64();
    let ept_bytes = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(ept_bytes, elems_per_thread * 2);
    ptx.mul_wide_s32(thread_byte_off, tid, ept_bytes);

    let thread_input = ptx.regs.alloc_b64();
    ptx.add_s64(thread_input, row_input_ptr, thread_byte_off);
    let thread_output = ptx.regs.alloc_b64();
    ptx.add_s64(thread_output, row_output_ptr, thread_byte_off);
    let thread_weight = ptx.regs.alloc_b64();
    ptx.add_s64(thread_weight, weight_ptr, thread_byte_off);
    ptx.blank();

    // ── Step 1: Load f16 pairs, convert to f32 ──
    ptx.comment("Load input (f16 pairs), convert to f32");
    let num_pairs = elems_per_thread / 2;
    let mut raw_pairs = Vec::new();
    let mut f32_vals = Vec::new();

    for i in 0..num_pairs {
        let raw = ptx.regs.alloc_b32();
        ptx.ld_global_b32(raw, thread_input, (i * 4) as i32);
        raw_pairs.push(raw);

        // Low f16
        let lo = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{lo}, {raw};"));
        f32_vals.push(lo);

        // High f16
        let hi_bits = ptx.regs.alloc_b32();
        ptx.shr_u32(hi_bits, raw, 16);
        let hi = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{hi}, {hi_bits};"));
        f32_vals.push(hi);
    }
    ptx.blank();

    // ── Step 2: Partial sum of squares ──
    ptx.comment("Compute partial sum-of-squares");
    let partial_sum = ptx.regs.alloc_f32();
    ptx.mov_f32_imm(partial_sum, 0.0);
    for &v in &f32_vals {
        ptx.w(&format!("fma.rn.f32 \t{partial_sum}, {v}, {v}, {partial_sum};"));
    }
    ptx.blank();

    // ── Step 3: Warp-level butterfly shuffle reduction ──
    ptx.comment("Warp reduction via butterfly shuffle");
    let sum_b32 = ptx.regs.alloc_b32();
    ptx.mov_f32_to_b32(sum_b32, partial_sum);

    for offset in [16, 8, 4, 2, 1] {
        let shfl_result = ptx.regs.alloc_b32();
        ptx.w(&format!(
            "shfl.sync.bfly.b32 \t{shfl_result}, {sum_b32}, {offset}, 0x1F, 0xFFFFFFFF;"
        ));
        let shfl_f32 = ptx.regs.alloc_f32();
        ptx.mov_b32_to_f32(shfl_f32, shfl_result);
        ptx.add_f32(partial_sum, partial_sum, shfl_f32);
        ptx.mov_f32_to_b32(sum_b32, partial_sum);
    }
    ptx.blank();

    // ── Step 4: Cross-warp reduction via shared memory ──
    ptx.comment("Cross-warp reduction via shared memory");
    let smem_base = ptx.regs.alloc_b32();
    ptx.mov_b32_name(smem_base, "global_smem");

    // Lane 0 of each warp writes to smem[warp_id * 4]
    let p_lane0 = ptx.regs.alloc_pred();
    ptx.w(&format!("setp.eq.s32 \t{p_lane0}, {lane}, 0;"));

    let warp_scratch = ptx.regs.alloc_b32();
    let warp_off = ptx.regs.alloc_b32();
    ptx.shl_b32(warp_off, warp_id, 2);
    ptx.add_s32(warp_scratch, smem_base, warp_off);

    ptx.w(&format!("@{p_lane0} st.shared.b32 \t[{warp_scratch}], {sum_b32};"));
    ptx.bar_sync(0);

    // All threads in the first warp read and reduce
    // Actually simpler: thread 0 reduces, broadcasts via smem.
    // But for best perf with few warps, let warp 0 do the final reduction.
    let p_tid0 = ptx.regs.alloc_pred();
    ptx.w(&format!("setp.eq.s32 \t{p_tid0}, {tid}, 0;"));

    let total_sum = ptx.regs.alloc_f32();
    ptx.mov_f32_imm(total_sum, 0.0);

    for w in 0..num_warps {
        let ws = ptx.regs.alloc_b32();
        ptx.w(&format!("@{p_tid0} ld.shared.b32 \t{ws}, [{smem_base}+{}];", w * 4));
        let ws_f32 = ptx.regs.alloc_f32();
        ptx.mov_b32_to_f32(ws_f32, ws);
        ptx.w(&format!("@{p_tid0} add.f32 \t{total_sum}, {total_sum}, {ws_f32};"));
    }

    // mean = total_sum / hidden_size
    let mean_val = ptx.regs.alloc_f32();
    ptx.w(&format!(
        "@{p_tid0} mul.f32 \t{mean_val}, {total_sum}, 0F{:08X};",
        (1.0f32 / hidden_size as f32).to_bits()
    ));

    // mean + eps
    let eps_f32 = ptx.regs.alloc_f32();
    ptx.mov_b32_to_f32(eps_f32, eps_param);
    ptx.w(&format!("@{p_tid0} add.f32 \t{mean_val}, {mean_val}, {eps_f32};"));

    // rsqrt
    let scale = ptx.regs.alloc_f32();
    ptx.w(&format!("@{p_tid0} rsqrt.approx.f32 \t{scale}, {mean_val};"));

    // Broadcast scale via shared memory
    let scale_b32 = ptx.regs.alloc_b32();
    ptx.mov_f32_to_b32(scale_b32, scale);
    ptx.w(&format!("@{p_tid0} st.shared.b32 \t[{smem_base}], {scale_b32};"));
    ptx.bar_sync(0);

    let scale_shared = ptx.regs.alloc_b32();
    ptx.w(&format!("ld.shared.b32 \t{scale_shared}, [{smem_base}];"));
    let scale_all = ptx.regs.alloc_f32();
    ptx.mov_b32_to_f32(scale_all, scale_shared);
    ptx.blank();

    // ── Step 5: Load weight, normalize, convert to f16, store ──
    ptx.comment("Load weight, compute y = x * scale * w, store f16");

    // Load weight f16 pairs
    let mut weight_f32 = Vec::new();
    for i in 0..num_pairs {
        let raw = ptx.regs.alloc_b32();
        ptx.ld_global_b32(raw, thread_weight, (i * 4) as i32);

        let lo = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{lo}, {raw};"));
        weight_f32.push(lo);

        let hi_bits = ptx.regs.alloc_b32();
        ptx.shr_u32(hi_bits, raw, 16);
        let hi = ptx.regs.alloc_f32();
        ptx.w(&format!("cvt.f32.f16 \t{hi}, {hi_bits};"));
        weight_f32.push(hi);
    }

    // Normalize and store pairs
    for i in 0..num_pairs as usize {
        let idx0 = i * 2;
        let idx1 = i * 2 + 1;

        // t = x * scale
        let t0 = ptx.regs.alloc_f32();
        ptx.mul_f32(t0, f32_vals[idx0], scale_all);
        // y = t * weight
        let y0 = ptx.regs.alloc_f32();
        ptx.mul_f32(y0, t0, weight_f32[idx0]);

        let t1 = ptx.regs.alloc_f32();
        ptx.mul_f32(t1, f32_vals[idx1], scale_all);
        let y1 = ptx.regs.alloc_f32();
        ptx.mul_f32(y1, t1, weight_f32[idx1]);

        // Convert to f16 and pack
        let h0 = ptx.regs.alloc_b32();
        ptx.w(&format!("cvt.rn.f16.f32 \t{h0}, {y0};"));
        let h1 = ptx.regs.alloc_b32();
        ptx.w(&format!("cvt.rn.f16.f32 \t{h1}, {y1};"));

        let h1_shifted = ptx.regs.alloc_b32();
        ptx.shl_b32(h1_shifted, h1, 16);
        let packed = ptx.regs.alloc_b32();
        ptx.or_b32(packed, h0, h1_shifted);

        // Store to global output
        ptx.st_global_b32(thread_output, (i * 4) as i32, packed);
    }
    ptx.blank();

    ptx.ret();

    let params = vec![
        (".u64 .ptr .global .align 16", "param_output"),
        (".u64 .ptr .global .align 16", "param_input"),
        (".u64 .ptr .global .align 16", "param_weight"),
        (".f32", "param_eps"),
    ];

    // Need shared memory for warp reduction scratch (num_warps * 4 bytes)
    ptx.finalize("rmsnorm_kernel", &params)
}
