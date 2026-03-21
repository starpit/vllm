use crate::{PtxBuilder, Reg};

// ═══════════════════════════════════════════════════════════════════════════
// TileProducer trait — abstracts how tiles arrive in shared memory.
//
// The GEMM K-loop calls `emit_tile_load` instead of hardcoding cp.async.
// Different producers fill smem differently:
//   - CpAsyncProducer: hardware DMA from global memory (fast, for raw GEMM)
//   - NormalizedProducer: load from global, normalize, write to smem
//     (for RMSNorm→GEMM fusion)
// ═══════════════════════════════════════════════════════════════════════════

/// State produced during prologue setup that the GEMM K-loop needs.
pub struct TileProducerState {
    /// Global pointer registers for chunk 0 and chunk 1 (current loop iteration).
    pub g_ptrs: Vec<Reg>,
    /// B stride in bytes (for B pointer advancement).
    pub b_stride_bytes: Option<Reg>,
}

/// Context passed to tile load emission.
pub struct TileLoadCtx {
    /// smem destination address for chunk 0 (already includes buffer offset + cp_off).
    pub smem_dst0: Reg,
    /// smem destination address for chunk 1 (dst0 + 2048).
    pub smem_dst1: Reg,
    /// Predicated copy size: 16 or 0.
    pub cp_size: Reg,
    /// Global pointer for chunk 0.
    pub g_ptr0: Reg,
    /// Global pointer for chunk 1.
    pub g_ptr1: Reg,
    /// Predicate for whether to load (true = load, false = skip).
    pub p_load: Reg,
}

// ═══════════════════════════════════════════════════════════════════════════
// CpAsyncProducer — wraps hardware DMA (cp.async.cg.shared.global)
// ═══════════════════════════════════════════════════════════════════════════

/// Fills a tile via cp.async from global memory. This is the fast path
/// used by standalone GEMM.
pub struct CpAsyncProducer;

impl CpAsyncProducer {
    /// Emit cp.async for two chunks (chunk 0 and chunk 1) of a tile.
    pub fn emit_tile_load(ptx: &mut PtxBuilder, ctx: &TileLoadCtx) {
        ptx.cp_async_cg(ctx.smem_dst0, 0, ctx.g_ptr0, 0, ctx.cp_size);
        ptx.cp_async_cg(ctx.smem_dst1, 0, ctx.g_ptr1, 0, ctx.cp_size);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NormalizedProducer — loads input, applies x * norm_factor * weight,
// converts to f16, and writes to smem.
// ═══════════════════════════════════════════════════════════════════════════

/// Context for the NormalizedProducer — held across the K-loop.
pub struct NormProducerCtx {
    /// .b64 register pointing to RMSNorm weight vector (advances by BK*2 each iter).
    pub wnorm_ptr: Reg,
    /// .b32 register: base of per-row norm factors in smem.
    pub norm_factors_base: Reg,
    /// .b32 register: global row index for chunk 0 (block_row + a_tid_row).
    pub row_id_chunk0: Reg,
    /// .b32 register: global row index for chunk 1 (= row_id_chunk0 + 32).
    pub row_id_chunk1: Reg,
    /// .b32 register: block_row (starting row of this block).
    pub block_row: Reg,
}

/// Emit a normalized A-tile store for one chunk.
///
/// Loads 16 bytes (4 pairs of f16) from global input, loads corresponding
/// RMSNorm weight, multiplies by the per-row norm factor, converts back to f16,
/// and stores to the swizzled smem address.
///
/// Uses ld.global.v4.b32 for the input (16 bytes in one instruction) and
/// ld.global.v4.b32 for the weight to maximize throughput.
pub fn emit_normalized_chunk_store(
    ptx: &mut PtxBuilder,
    global_addr: Reg,      // .b64: global address for this chunk's 16 bytes
    wnorm_addr: Reg,       // .b64: weight address for same K-range
    smem_addr: Reg,         // .b32: swizzled smem destination
    row_id: Reg,            // .b32: global row index
    norm_factors_base: Reg, // .b32: base of norm factor array
    block_row: Reg,         // .b32: block starting row
    predicate: Option<Reg>, // optional predicate (None = always store)
) {
    // Load norm factor for this row from smem
    let local_row = ptx.regs.alloc_b32();
    ptx.sub_s32(local_row, row_id, block_row);
    let factor_off = ptx.regs.alloc_b32();
    ptx.shl_b32(factor_off, local_row, 2); // * 4 bytes
    let factor_addr = ptx.regs.alloc_b32();
    ptx.add_s32(factor_addr, norm_factors_base, factor_off);
    let norm_factor_b32 = ptx.regs.alloc_b32();
    ptx.ld_shared_b32(norm_factor_b32, factor_addr, 0);
    let norm_factor = ptx.regs.alloc_f32();
    ptx.mov_b32_to_f32(norm_factor, norm_factor_b32);

    // Load 16 bytes from input using ld.global.v4.b32
    // Each b32 holds a pair of f16 values (4 pairs = 8 f16s = 16 bytes)
    let raw_in = [
        ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
    ];
    // Load 16 bytes from weight
    let raw_wt = [
        ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
        ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
    ];

    if let Some(pred) = predicate {
        ptx.pred_ld_global_v4_b32(pred, raw_in, global_addr, 0);
        ptx.pred_ld_global_v4_b32(pred, raw_wt, wnorm_addr, 0);
    } else {
        ptx.ld_global_v4_b32(raw_in, global_addr, 0);
        ptx.ld_global_v4_b32(raw_wt, wnorm_addr, 0);
    }

    // Process each pair: extract lo/hi f16, normalize, pack back, store
    for pair_idx in 0..4u32 {
        let inp = raw_in[pair_idx as usize];
        let wgt = raw_wt[pair_idx as usize];

        // Low f16 -> f32
        let x_lo = ptx.regs.alloc_f32();
        ptx.cvt_f32_f16(x_lo, inp);
        let w_lo = ptx.regs.alloc_f32();
        ptx.cvt_f32_f16(w_lo, wgt);

        // y_lo = x_lo * norm_factor * w_lo
        // Use fma: t_lo = x_lo * norm_factor + 0, then y_lo = t_lo * w_lo
        // Actually mul is fine since we don't need the add:
        let t_lo = ptx.regs.alloc_f32();
        ptx.mul_f32(t_lo, x_lo, norm_factor);
        let y_lo = ptx.regs.alloc_f32();
        ptx.mul_f32(y_lo, t_lo, w_lo);

        // High f16 -> f32
        let hi_input = ptx.regs.alloc_b32();
        ptx.shr_u32(hi_input, inp, 16);
        let x_hi = ptx.regs.alloc_f32();
        ptx.cvt_f32_f16(x_hi, hi_input);
        let hi_weight = ptx.regs.alloc_b32();
        ptx.shr_u32(hi_weight, wgt, 16);
        let w_hi = ptx.regs.alloc_f32();
        ptx.cvt_f32_f16(w_hi, hi_weight);
        let t_hi = ptx.regs.alloc_f32();
        ptx.mul_f32(t_hi, x_hi, norm_factor);
        let y_hi = ptx.regs.alloc_f32();
        ptx.mul_f32(y_hi, t_hi, w_hi);

        // Convert back to f16 and pack
        let h_lo = ptx.regs.alloc_b32();
        ptx.cvt_rn_f16_f32(h_lo, y_lo);
        let h_hi = ptx.regs.alloc_b32();
        ptx.cvt_rn_f16_f32(h_hi, y_hi);

        let h_hi_shifted = ptx.regs.alloc_b32();
        ptx.shl_b32(h_hi_shifted, h_hi, 16);
        let packed = ptx.regs.alloc_b32();
        ptx.or_b32(packed, h_lo, h_hi_shifted);

        // Store to shared memory
        if let Some(pred) = predicate {
            ptx.pred_st_shared_b32(pred, smem_addr, (pair_idx * 4) as i32, packed);
        } else {
            ptx.st_shared_b32(smem_addr, (pair_idx * 4) as i32, packed);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RMSNorm reduction — computes per-row norm factors
// ═══════════════════════════════════════════════════════════════════════════

/// Emit the RMSNorm norm-factor computation phase.
///
/// With 128 threads and BM=64 rows, assigns 2 threads per row.
/// Each thread loads hidden_size/2 elements, computes partial sum-of-squares,
/// reduces with partner via shfl, computes rsqrt, stores to smem.
///
/// Returns the register holding the smem base of norm factors.
pub fn emit_norm_factor_computation(
    ptx: &mut PtxBuilder,
    input_ptr: Reg,        // .b64: base of input tensor
    block_row: Reg,        // .b32: starting row of this block
    tid: Reg,              // .b32: thread id
    smem_base: Reg,        // .b32: shared memory base
    norm_factors_off: u32, // byte offset in smem for norm factors array
    hidden_size: u32,
    eps: f32,
) -> Reg {
    emit_norm_factor_computation_inner(
        ptx, input_ptr, block_row, tid, smem_base, norm_factors_off,
        hidden_size, eps, None,
    )
}

/// Like `emit_norm_factor_computation`, but accepts an optional pre-allocated
/// `norm_factors_base` register. When `Some(reg)` is passed, that register is
/// used instead of allocating a new one — useful when the caller wants to
/// allocate it outside a register scope so it survives `pop_scope()`.
pub fn emit_norm_factor_computation_inner(
    ptx: &mut PtxBuilder,
    input_ptr: Reg,
    block_row: Reg,
    tid: Reg,
    smem_base: Reg,
    norm_factors_off: u32,
    hidden_size: u32,
    eps: f32,
    pre_allocated_base: Option<Reg>,
) -> Reg {
    // Determine parallelism based on caller's context.
    // The number of rows to process is inferred from the smem reservation:
    // norm_factors region holds (norm_factors_off bytes after gemm_smem) / 4 rows.
    // But we don't have BM here. Use a heuristic: if hidden_size >= 4096 and
    // we're called from a 128-row context, use 1 thread/row. Otherwise 2 threads/row.
    //
    // Actually, let's just check: with 128 threads, 2 threads/row gives 64 rows.
    // If we need more rows, we must use 1 thread/row.
    // The caller should use the _cfg variant for BM>64.
    emit_norm_factor_computation_cfg(ptx, input_ptr, block_row, tid, smem_base,
        norm_factors_off, hidden_size, eps, pre_allocated_base, 2)
}

/// Config-aware norm factor computation.
/// `threads_per_row`: 2 for BM<=64 (cooperative halving), 1 for BM=128 (single-thread).
pub fn emit_norm_factor_computation_cfg(
    ptx: &mut PtxBuilder,
    input_ptr: Reg,
    block_row: Reg,
    tid: Reg,
    smem_base: Reg,
    norm_factors_off: u32,
    hidden_size: u32,
    eps: f32,
    pre_allocated_base: Option<Reg>,
    threads_per_row: u32,
) -> Reg {
    ptx.comment(&format!("=== RMSNorm norm factor computation ({} threads/row) ===", threads_per_row));
    ptx.blank();

    let norm_factors_base = pre_allocated_base.unwrap_or_else(|| ptx.regs.alloc_b32());
    ptx.add_s32_imm(norm_factors_base, smem_base, norm_factors_off as i32);

    let elems_per_thread = hidden_size / threads_per_row;
    let tpr_shift = threads_per_row.trailing_zeros();

    // my_row = tid / threads_per_row, my_part = tid % threads_per_row
    let my_row = ptx.regs.alloc_b32();
    ptx.shr_u32(my_row, tid, tpr_shift);
    let my_part = ptx.regs.alloc_b32();
    ptx.and_b32(my_part, tid, threads_per_row - 1);

    // Global row address: input_ptr + (block_row + my_row) * hidden_size * 2
    let global_row = ptx.regs.alloc_b32();
    ptx.add_s32(global_row, block_row, my_row);
    let row_addr = ptx.regs.alloc_b64();
    {
        let hs_bytes = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(hs_bytes, hidden_size * 2);
        ptx.mad_wide_s32(row_addr, global_row, hs_bytes, input_ptr);
    }

    // Thread's element offset within the row: my_part * elems_per_thread * 2 bytes
    let part_byte_off = ptx.regs.alloc_b64();
    {
        let part_bytes = ptx.regs.alloc_b32();
        ptx.mov_b32_imm(part_bytes, elems_per_thread * 2);
        ptx.mul_wide_s32(part_byte_off, my_part, part_bytes);
    }
    let thread_addr = ptx.regs.alloc_b64();
    ptx.add_s64(thread_addr, row_addr, part_byte_off);

    // Compute partial sum-of-squares using a loop with unrolled v4 loads
    let partial_sum = ptx.regs.alloc_f32();
    ptx.mov_f32_imm(partial_sum, 0.0);

    // Each v4.b32 loads 4 b32s = 8 f16s.
    let num_v4_loads = elems_per_thread / 8;
    let unroll = 4u32;
    let loop_iters = num_v4_loads / unroll;

    let loop_ctr = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(loop_ctr, 0);
    let loop_lim = ptx.regs.alloc_b32();
    ptx.mov_b32_imm(loop_lim, loop_iters);

    let cur_addr = ptx.regs.alloc_b64();
    ptx.mov_b64(cur_addr, thread_addr);

    ptx.label("$L_NORM_SOS_LOOP");
    {
        for u in 0..unroll {
            let data = [
                ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
                ptx.regs.alloc_b32(), ptx.regs.alloc_b32(),
            ];
            ptx.ld_global_v4_b32(data, cur_addr, (u * 16) as i32);

            // Each b32 holds 2 f16s. Extract and accumulate.
            for &d in &data {
                let lo = ptx.regs.alloc_f32();
                ptx.cvt_f32_f16(lo, d);
                ptx.fma_f32(partial_sum, lo, lo, partial_sum);

                let hi_bits = ptx.regs.alloc_b32();
                ptx.shr_u32(hi_bits, d, 16);
                let hi = ptx.regs.alloc_f32();
                ptx.cvt_f32_f16(hi, hi_bits);
                ptx.fma_f32(partial_sum, hi, hi, partial_sum);
            }
        }
        ptx.add_s64_imm(cur_addr, cur_addr, (unroll * 16) as i64);
        ptx.add_s32_imm(loop_ctr, loop_ctr, 1);
        let p_loop = ptx.regs.alloc_pred();
        ptx.setp_lt_s32(p_loop, loop_ctr, loop_lim);
        ptx.pred_bra(p_loop, "$L_NORM_SOS_LOOP");
    }
    ptx.blank();

    if threads_per_row > 1 {
        // Reduce with partner thread(s) via shfl.bfly
        for delta in 0..tpr_shift {
            let sum_b32 = ptx.regs.alloc_b32();
            ptx.mov_f32_to_b32(sum_b32, partial_sum);
            let partner_sum = ptx.regs.alloc_b32();
            ptx.shfl_bfly(partner_sum, sum_b32, 1 << delta);
            let partner_f32 = ptx.regs.alloc_f32();
            ptx.mov_b32_to_f32(partner_f32, partner_sum);
            ptx.add_f32(partial_sum, partial_sum, partner_f32);
        }
    }

    // Compute norm factor: rsqrt(mean_sq + eps)
    let mean_val = ptx.regs.alloc_f32();
    ptx.mul_f32_imm(mean_val, partial_sum, 1.0f32 / hidden_size as f32);
    ptx.add_f32_imm(mean_val, mean_val, eps);
    let scale = ptx.regs.alloc_f32();
    ptx.rsqrt_approx_f32(scale, mean_val);

    if threads_per_row > 1 {
        // Only first thread in each group stores
        let p_first = ptx.regs.alloc_pred();
        ptx.setp_eq_s32(p_first, my_part, 0);

        let scale_b32 = ptx.regs.alloc_b32();
        ptx.mov_f32_to_b32(scale_b32, scale);
        let row_off = ptx.regs.alloc_b32();
        ptx.shl_b32(row_off, my_row, 2);
        let row_factor_addr = ptx.regs.alloc_b32();
        ptx.add_s32(row_factor_addr, norm_factors_base, row_off);
        ptx.w(&format!("@{p_first} st.shared.b32 \t[{row_factor_addr}], {scale_b32};"));
    } else {
        // All threads store (1 thread per row, no predicate needed)
        let scale_b32 = ptx.regs.alloc_b32();
        ptx.mov_f32_to_b32(scale_b32, scale);
        let row_off = ptx.regs.alloc_b32();
        ptx.shl_b32(row_off, my_row, 2);
        let row_factor_addr = ptx.regs.alloc_b32();
        ptx.add_s32(row_factor_addr, norm_factors_base, row_off);
        ptx.st_shared_b32(row_factor_addr, 0, scale_b32);
    }
    ptx.bar_sync(0);
    ptx.blank();

    norm_factors_base
}
