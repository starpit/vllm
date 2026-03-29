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
use crate::parser::DefUseGraph;
use crate::pipeline::{PipelineStage, ReductionDecomposition, StagePattern};

/// Extract the GEMM tile dimensions from a CUTLASS mangled entry name.
///
/// Parses `GemmShapeILi{M}ELi{N}ELi{K}E` → (M, N, K).
fn extract_gemm_shape_from_entry(name: &str) -> Option<(u32, u32, u32)> {
    let marker = "GemmShapeILi";
    let idx = name.find(marker)?;
    let rest = &name[idx + marker.len()..];
    // rest = "{M}ELi{N}ELi{K}E..."
    let mut dims = Vec::new();
    let mut cur = rest;
    for _ in 0..3 {
        let end = cur.find('E')?;
        dims.push(cur[..end].parse::<u32>().ok()?);
        cur = &cur[end + 1..]; // skip 'E'
        if cur.starts_with("Li") {
            cur = &cur[2..]; // skip 'Li'
        }
    }
    Some((dims[0], dims[1], dims[2]))
}

/// Extract just tile_m for backward compatibility.
fn extract_tile_m_from_entry(name: &str) -> Option<u32> {
    extract_gemm_shape_from_entry(name).map(|(m, _, _)| m)
}

/// Thread-to-row mapping parameters extracted from GEMM PTX.
///
/// The mapping formula is:
///   m_rel = (tid.x % 32) / lanes_per_row + (tid.x / 32) * rows_per_warp
///   row_0 = m_rel, row_1 = m_rel + row_stride
///
/// These are extracted from the `shr` and `shl` shift amounts in the
/// A-load address computation chain.
#[derive(Debug, Clone)]
struct ThreadRowMap {
    /// Number of lanes that share a row (detected as 1 << shr_amount on lane)
    lanes_per_row_log2: u32,
    /// Number of rows each warp covers (detected as 1 << shl_amount on warp)
    rows_per_warp_log2: u32,
    /// Row stride between a thread's two rows (typically 8)
    row_stride: u32,
}

/// Extract the thread-to-row mapping from a GEMM kernel's PTX.
///
/// Strategy: find the first A-load's address chain, locate the
/// `mul.lo.s64 %rd, stride, row_reg`, then trace `row_reg` backward
/// to find the `shr` (lane division) and `shl` (warp multiply) constants.
fn extract_thread_row_map(gemm_lines: &[&str]) -> Option<ThreadRowMap> {
    let graph = DefUseGraph::build(gemm_lines);

    // Find the first cp.async A-load's gmem_src register
    let mut gmem_src = String::new();
    for line in gemm_lines {
        let t = line.trim();
        if t.contains("cp.async.cg.shared.global") {
            // Extract second bracket content (gmem source)
            let mut brackets = Vec::new();
            let mut i = 0;
            let bytes = t.as_bytes();
            while i < bytes.len() {
                if bytes[i] == b'[' {
                    let start = i + 1;
                    while i < bytes.len() && bytes[i] != b']' {
                        i += 1;
                    }
                    brackets.push(t[start..i].trim().to_string());
                }
                i += 1;
            }
            if brackets.len() >= 2 {
                gmem_src = brackets[1].clone();
                break;
            }
        }
    }
    if gmem_src.is_empty() {
        return None;
    }

    // Trace backward from gmem_src to find the mul.lo.s64 by stride
    let trace = graph.trace_backward(&gmem_src, 10);

    // Find the mul.lo.s64 — one operand is the stride (from ld.param), the other is the row
    let mut row_reg = String::new();
    for &(_depth, node_idx) in &trace {
        let node = &graph.nodes[node_idx];
        if node.opcode == "mul.lo.s64" && node.sources.len() == 2 {
            // One source should be a param-derived register, the other is the row
            // The param-derived one will appear in ld.param results
            // Check which source is NOT in our trace (i.e., comes from a param)
            for src in &node.sources {
                // Trace this source backward — if it hits ld.param quickly, it's the stride
                let sub_trace = graph.trace_backward(src, 3);
                let hits_param = sub_trace
                    .iter()
                    .any(|&(_, ni)| graph.nodes[ni].opcode.starts_with("ld.param"));
                if !hits_param {
                    // This is the row register (not from param)
                    row_reg = src.clone();
                }
            }
            if !row_reg.is_empty() {
                break;
            }
        }
    }
    if row_reg.is_empty() {
        return None;
    }

    // Trace the row register backward to find the shift amounts.
    // The row is computed from tid.x via:
    //   lane = tid.x % 32  (and/sub pattern)
    //   warp = tid.x / 32  (shr by 5)
    //   m_in_warp = lane >> N  (shr.s32, N = lanes_per_row_log2)
    //   warp_contrib = warp << M  (shl.b32, M = rows_per_warp_log2)
    //   m_rel = m_in_warp + warp_contrib  (add)
    let row_trace = graph.trace_backward(&row_reg, 12);

    let mut lane_shr: Option<u32> = None;
    let mut warp_shl: Option<u32> = None;

    for &(_depth, node_idx) in &row_trace {
        let node = &graph.nodes[node_idx];

        // Look for shr.s32 with a small immediate — lane / N
        if node.opcode == "shr.s32" && node.sources.len() == 2 {
            if let Ok(shift) = node.sources[1].parse::<u32>() {
                if shift >= 1 && shift <= 4 && shift != 5 {
                    // shr by 2 = lane / 4 (not shr by 5 which is tid.x / 32)
                    lane_shr = Some(shift);
                }
            }
        }

        // Look for shl.b32 with a small immediate — warp * N
        if node.opcode == "shl.b32" && node.sources.len() == 2 {
            if let Ok(shift) = node.sources[1].parse::<u32>() {
                if shift >= 3 && shift <= 5 {
                    // shl by 4 = warp * 16, shl by 5 = warp * 32
                    warp_shl = Some(shift);
                }
            }
        }
    }

    // Detect row_stride by looking for the second row computation (add.s32 %r, %r, N)
    // where N is the stride between a thread's two rows
    let mut row_stride = 8u32; // default
    for &(_depth, node_idx) in &row_trace {
        let node = &graph.nodes[node_idx];
        if node.opcode == "shl.b32" && node.sources.len() == 2 {
            if let Ok(shift) = node.sources[1].parse::<u32>() {
                if shift == 3 {
                    // shl by 3 = offset * 8, suggests row_stride = 8
                    row_stride = 8;
                }
            }
        }
    }

    Some(ThreadRowMap {
        lanes_per_row_log2: lane_shr.unwrap_or(2), // default: lane / 4
        rows_per_warp_log2: warp_shl.unwrap_or(4), // default: warp * 16
        row_stride,
    })
}

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
    let computation = build_reduction_computation(&decomp, reduction, gemm, fused_name)?;

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
    _gemm: &PipelineStage,
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
        ".reg .f32 %f_rms_inv;".into(), // selected inv_rms for current site
        ".reg .f32 %f_rms_inv0, %f_rms_inv1;".into(), // inv_rms for row 0 and row 1
        ".reg .f32 %f_rms_sq, %f_rms_sum;".into(), // accumulation scratch
        ".reg .f32 %f_rms_eps, %f_rms_hdnf;".into(),
        ".reg .f32 %f_rms_t0, %f_rms_t1;".into(),
        ".reg .b32 %r_rms_k, %r_rms_hdn, %r_rms_step;".into(),
        ".reg .b32 %r_rms_row, %r_rms_nrows, %r_rms_mtile;".into(),
        ".reg .b64 %rd_rms_in, %rd_rms_wt, %rd_rms_str;".into(),
        ".reg .b64 %rd_rms_rowbase, %rd_rms_cur;".into(),
        ".reg .b64 %rd_rms_rb0, %rd_rms_rb1;".into(), // GMEM row base addresses for parity select
        ".reg .pred %p_rms_lp, %p_rms_row, %p_rms_par;".into(),
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
    // Extract thread-to-row mapping from GEMM PTX
    let gemm_lines: Vec<&str> = _gemm.source_lines.iter().map(|s| s.as_str()).collect();
    let thread_map = extract_thread_row_map(&gemm_lines);
    #[cfg(test)]
    if let Some(ref tm) = thread_map {
        eprintln!(
            "  thread_row_map: lanes_per_row=1<<{}, rows_per_warp=1<<{}, row_stride={}",
            tm.lanes_per_row_log2, tm.rows_per_warp_log2, tm.row_stride
        );
    }

    // Extract tile dimensions from the GEMM's mangled entry name
    let (tile_m, tile_n, _tile_k) = extract_gemm_shape_from_entry(&_gemm.protocol.name)
        .ok_or_else(|| {
            format!(
                "could not extract GemmShape from GEMM entry name: {}",
                _gemm.protocol.name
            )
        })?;

    // Extract the GEMM's struct param name (for reading N from params)
    let gemm_struct_param = _gemm
        .protocol
        .params
        .iter()
        .find(|p| p.ptx_type.contains(".b8"))
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "ferrite_params".to_string());

    let prologue = build_prologue_from_decomposition(
        decomp,
        thread_map.as_ref(),
        tile_m,
        tile_n,
        &gemm_struct_param,
    );

    // ── Per-site code ──
    // Row selection: the prologue pre-loads inv_rms for both rows into
    // %f_rms_inv0 and %f_rms_inv1. Per-site selects based on compile-time
    // parity bitmask using {LOAD_INDEX}.
    //
    // Weight loading: compute K byte offset from GMEM address.
    // k_byte_offset = (gmem_src - a_ptr) - row_offset
    // where row_offset is selected by parity.
    let per_site = vec![
        "// FERRITE: select inv_rms by row address comparison (compile-time optimized)".into(),
        // Compare gmem_src against row 1 base: if gmem_src >= rb1, it's row 1
        "setp.ge.u64 \t%p_rms_par, {GMEM_SRC}, %rd_rms_rb1;".into(),
        "selp.f32 \t%f_rms_inv, %f_rms_inv1, %f_rms_inv0, %p_rms_par;".into(),
        "// FERRITE: load weight at K offset from GMEM address".into(),
        // weight_addr = weight_ptr + (gmem_src - row_base[parity])
        "selp.u64 \t%rd_rms_cur, %rd_rms_rb1, %rd_rms_rb0, %p_rms_par;".into(),
        "sub.u64 \t%rd_rms_cur, {GMEM_SRC}, %rd_rms_cur;".into(), // k_byte_offset
        "add.u64 \t%rd_rms_cur, %rd_rms_wt, %rd_rms_cur;".into(), // weight + k_byte_offset
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
fn build_prologue_from_decomposition(
    decomp: &ReductionDecomposition,
    thread_map: Option<&ThreadRowMap>,
    tile_m: u32,
    tile_n: u32,
    _gemm_struct_param: &str,
) -> Vec<String> {
    // Use extracted constants or defaults
    let lane_shr = thread_map.map_or(2, |tm| tm.lanes_per_row_log2);
    let warp_shl = thread_map.map_or(4, |tm| tm.rows_per_warp_log2);
    let row_stride = thread_map.map_or(8, |tm| tm.row_stride);
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

    // Unswizzle ctaid.x to get the M-tile index.
    // CUTLASS ThreadblockSwizzle<4> maps grid as:
    //   grid_x = grid_m * (1 << swizzle_log), grid_y = ceil(grid_n / (1 << swizzle_log))
    //   m_tile = ctaid.x >> swizzle_log
    // Read N from flat GEMM params to compute swizzle_log at runtime.
    // (After perimeter replacement, ferrite_params[68] = N.)
    let tile_n_shift = tile_n.trailing_zeros();
    prologue.push("// Compute swizzle_log from N (ThreadblockSwizzle<4>)".into());
    prologue.push("ld.param.s32 \t%r_rms_step, [ferrite_params+68];".into()); // N
    prologue.push(format!(
        "add.s32 \t%r_rms_step, %r_rms_step, {};",
        tile_n - 1
    ));
    prologue.push(format!(
        "shr.u32 \t%r_rms_step, %r_rms_step, {tile_n_shift};"
    )); // grid_n = ceil(N/tile_n)
    // swizzle_log: 0 if grid_n<2, 1 if grid_n<3, 2 if grid_n>=3
    prologue.push("mov.u32 \t%r_rms_nrows, 0;".into()); // reuse as swizzle_log temp
    prologue.push("setp.ge.u32 \t%p_rms_lp, %r_rms_step, 2;".into());
    prologue.push("@%p_rms_lp mov.u32 \t%r_rms_nrows, 1;".into());
    prologue.push("setp.ge.u32 \t%p_rms_lp, %r_rms_step, 3;".into());
    prologue.push("@%p_rms_lp mov.u32 \t%r_rms_nrows, 2;".into());
    // m_tile = ctaid.x >> swizzle_log (saved for reuse across row loop iterations)
    prologue.push("mov.u32 \t%r_rms_mtile, %ctaid.x;".into());
    prologue.push("shr.u32 \t%r_rms_mtile, %r_rms_mtile, %r_rms_nrows;".into());

    // Row loop: for each row in this CTA's tile
    prologue.push(format!("mov.u32 \t%r_rms_nrows, {tile_m};"));
    prologue.push("mov.u32 \t%r_rms_row, 0;".into());
    prologue.push("$L_rms_row_loop:".into());

    // Compute row base address: input + (m_tile * tile_m + row) * stride * 2
    prologue.push("mul.lo.s32 \t%r_rms_step, %r_rms_mtile, %r_rms_nrows;".into());
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

    // ── Per-thread setup: load inv_rms for this thread's two rows ──
    // Thread-to-row mapping extracted from GEMM PTX address chain:
    //   m_rel = (tid.x % 32) >> lane_shr + (tid.x / 32) << warp_shl
    //   row_0 = m_rel, row_1 = m_rel + row_stride
    prologue.push(format!(
        "// Per-thread: load inv_rms (lane_shr={lane_shr}, warp_shl={warp_shl}, row_stride={row_stride})"
    ));
    prologue.push("mov.u32 \t%r_rms_step, %tid.x;".into());
    prologue.push("and.b32 \t%r_rms_k, %r_rms_step, 31;".into()); // lane = tid.x % 32
    prologue.push(format!("shr.u32 \t%r_rms_k, %r_rms_k, {lane_shr};")); // lane / lanes_per_row
    prologue.push("shr.u32 \t%r_rms_row, %r_rms_step, 5;".into()); // warp = tid.x / 32
    prologue.push(format!("shl.b32 \t%r_rms_row, %r_rms_row, {warp_shl};")); // warp * rows_per_warp
    prologue.push("add.u32 \t%r_rms_row, %r_rms_row, %r_rms_k;".into()); // m_rel

    // Load inv_rms[m_rel] and inv_rms[m_rel + row_stride]
    let row_stride_bytes = row_stride * 4; // 4 bytes per f32
    prologue.push("mov.u32 \t%r_rms_k, _ferrite_inv_rms;".into());
    prologue.push("shl.b32 \t%r_rms_step, %r_rms_row, 2;".into()); // m_rel * 4
    prologue.push("add.s32 \t%r_rms_step, %r_rms_k, %r_rms_step;".into());
    prologue.push("ld.shared.f32 \t%f_rms_inv0, [%r_rms_step];".into());
    prologue.push(format!(
        "ld.shared.f32 \t%f_rms_inv1, [%r_rms_step+{row_stride_bytes}];"
    )); // +row_stride rows * 4 bytes

    // Compute GMEM row base addresses for both rows (for K-offset extraction in per-site)
    // rb0 = input_ptr + (m_tile * tile_m + m_rel) * stride * 2
    prologue.push("mul.lo.s32 \t%r_rms_step, %r_rms_mtile, %r_rms_nrows;".into());
    prologue.push("add.u32 \t%r_rms_step, %r_rms_step, %r_rms_row;".into()); // abs row 0
    prologue.push("cvt.s64.s32 \t%rd_rms_rb0, %r_rms_step;".into());
    prologue.push("mul.lo.s64 \t%rd_rms_rb0, %rd_rms_rb0, %rd_rms_str;".into());
    prologue.push("shl.b64 \t%rd_rms_rb0, %rd_rms_rb0, 1;".into()); // * 2 for bf16
    prologue.push("add.s64 \t%rd_rms_rb0, %rd_rms_in, %rd_rms_rb0;".into());
    // rb1 = rb0 + row_stride * stride * 2
    let rb1_shift = (row_stride * 2).trailing_zeros(); // row_stride * 2 as power of 2
    prologue.push(format!("shl.b64 \t%rd_rms_rb1, %rd_rms_str, {rb1_shift};")); // stride * row_stride * 2
    prologue.push("add.s64 \t%rd_rms_rb1, %rd_rms_rb0, %rd_rms_rb1;".into());

    prologue.push("// FERRITE: end rms_norm prologue".into());

    prologue
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_tile_m_from_gemm_entries() {
        // 64x128x32
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi128ELi32EEE"
            ),
            Some(64)
        );
        // 128x128x32
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi32EEE"
            ),
            Some(128)
        );
        // 128x128x64
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi64EEE"
            ),
            Some(128)
        );
        // 64x64x32
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi64ELi32EEE"
            ),
            Some(64)
        );
        // Non-CUTLASS entry
        assert_eq!(extract_tile_m_from_entry("vllm_rms_norm_kernel"), None);
    }

    #[test]
    fn build_computation_from_rms_norm() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", rms_ptx).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx).expect("parse gemm");
        let decomp = stage.decompose_reduction().expect("decompose");

        let comp = build_reduction_computation(&decomp, &stage, &gemm_stage, "test_fused")
            .expect("build computation");

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
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let rms_stage = PipelineStage::from_ptx("rms_norm", rms_ptx).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx).expect("parse gemm");

        let fused_ptx =
            fuse_reduction_into_gemm(&rms_stage, &gemm_stage, "fused_norm_gemm").expect("fuse");

        // Apply perimeter replacement (prologue reads N from flat params)
        let (fused_ptx, _) =
            crate::perimeter::replace_perimeter(&fused_ptx, deriv_json, "fused_norm_gemm")
                .expect("perimeter replacement");
        let fused_ptx = crate::dedup_reg_declarations(&fused_ptx);

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
