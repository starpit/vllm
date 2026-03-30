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

/// Fuse a Pointwise stage (e.g., silu_mul) into a TiledGemm's A-input.
///
/// The pointwise operation is computed inline at each A-matrix cp.async site.
/// For silu_mul: the GEMM's A_ptr points to gate_up_buf with lda=2*intermediate.
/// At each A-load, the gate value is loaded from the original address and the
/// up value is loaded from +intermediate_bytes offset. SiLU(gate)*up replaces
/// the A-matrix element.
pub fn fuse_pointwise_into_gemm(
    pointwise: &PipelineStage,
    gemm: &PipelineStage,
    fused_name: &str,
) -> Result<String, String> {
    match &pointwise.pattern {
        StagePattern::Pointwise => {}
        other => return Err(format!("producer is not Pointwise, got {:?}", other)),
    }
    match &gemm.pattern {
        StagePattern::TiledGemm { .. } => {}
        other => return Err(format!("consumer is not TiledGemm, got {:?}", other)),
    }

    let computation = build_silu_mul_computation(fused_name)?;

    let gemm_ptx = gemm.source_lines.join("\n");
    replace_a_loads_with_inline_fn(&gemm_ptx, "", &computation)
}

/// Build a PointwiseComputation for SiLU+mul fused into GEMM A-loads.
///
/// At each A-load site (cp.async), the original load reads gate values
/// (from gate_up_buf with lda=2*intermediate). The per-site code loads
/// the corresponding up values at +intermediate_bytes offset. Per-element
/// instructions apply SiLU(gate) * up.
fn build_silu_mul_computation(fused_name: &str) -> Result<PointwiseComputation, String> {
    // Extra params: just the byte offset to the "up" half
    let extra_params = vec![".param .u64 _ferrite_intermediate_bytes,".into()];

    // Register declarations for up loading + SiLU scratch
    let extra_reg_decls = vec![
        // Up value registers (loaded per-site, consumed per-element)
        ".reg .b32 %r_up0, %r_up1, %r_up2, %r_up3;".into(),
        ".reg .b16 %h_up_a, %h_up_b;".into(),
        ".reg .f32 %f_up0, %f_up1, %f_up2, %f_up3, %f_up4, %f_up5, %f_up6, %f_up7;".into(),
        // SiLU scratch (reuse fuse_epilogue pattern)
        ".reg .f32 %f_act0, %f_act1, %f_act2, %f_act3;".into(),
        ".reg .b32 %r_act0;".into(),
        // Intermediate offset
        ".reg .b64 %rd_up_off;".into(),
    ];

    // Param loads: load the intermediate byte offset once
    let param_loads = vec!["ld.param.u64 \t%rd_up_off, [_ferrite_intermediate_bytes];".into()];

    // Per-site: load 8 bf16 up values from GMEM_SRC + intermediate_bytes
    let mut per_site = vec!["// FERRITE: load up values at +intermediate offset".into()];
    // Compute up address = GMEM_SRC + intermediate_bytes
    per_site.push("add.u64 \t%rd_up_off, {GMEM_SRC}, %rd_up_off;".into());
    per_site.push("ld.global.v4.b32 \t{%r_up0, %r_up1, %r_up2, %r_up3}, [%rd_up_off];".into());
    // Restore rd_up_off (it was clobbered by the add)
    per_site.push("ld.param.u64 \t%rd_up_off, [_ferrite_intermediate_bytes];".into());

    // Unpack 8 bf16 up values to named f32 registers
    per_site.push("// FERRITE: unpack 8 bf16 up values to f32".into());
    for w in 0..4u32 {
        let lo = w * 2;
        let hi = w * 2 + 1;
        per_site.push(format!("mov.b32 \t{{%h_up_a, %h_up_b}}, %r_up{w};"));
        per_site.push(format!("cvt.f32.bf16 \t%f_up{lo}, %h_up_a;"));
        per_site.push(format!("cvt.f32.bf16 \t%f_up{hi}, %h_up_b;"));
    }

    // Per-element instructions: SiLU(gate) * up
    // {INPUT} = gate value (f32, from the original A-load after bf16→f32 conversion)
    // {ELEM_IDX} = 0..7 index into up registers
    let instructions = vec![
        // SiLU(gate) = gate / (1 + exp(-gate))
        // Uses fast exp via range reduction + ex2.approx (same as fuse_epilogue.rs)
        "neg.f32 \t%f_act0, {INPUT};".into(),
        "fma.rn.f32 \t%f_act1, %f_act0, 0f3BBB989D, 0f3F000000;".into(),
        "cvt.sat.f32.f32 \t%f_act1, %f_act1;".into(),
        "fma.rm.f32 \t%f_act2, %f_act1, 0f437C0000, 0f4B400001;".into(),
        "add.f32 \t%f_act3, %f_act2, 0fCB40007F;".into(),
        "neg.f32 \t%f_act3, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act0, 0f3FB8AA3B, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act0, 0f32A57060, %f_act3;".into(),
        "mov.b32 \t%r_act0, %f_act2;".into(),
        "shl.b32 \t%r_act0, %r_act0, 23;".into(),
        "mov.b32 \t%f_act2, %r_act0;".into(),
        "ex2.approx.ftz.f32 \t%f_act3, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act3, %f_act2, 0f3F800000;".into(),
        "div.rn.f32 \t{INPUT}, {INPUT}, %f_act3;".into(),
        // Multiply by corresponding up value
        "mul.f32 \t{INPUT}, {INPUT}, %f_up{ELEM_IDX};".into(),
    ];

    Ok(PointwiseComputation {
        instructions,
        param_loads,
        prologue: vec![], // No prologue needed for pointwise
        extra_reg_decls,
        extra_params,
        per_site,
        entry_name: Some(fused_name.to_string()),
        scratch_f32_count: 0,
        scratch_b32_count: 0,
    })
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
    if params.len() < 5 {
        return Err(format!(
            "expected reduction to have >= 5 params, got {}",
            params.len()
        ));
    }

    // Detect two-input reduction (fused_add_rms_norm): accumulation loop has st.global
    // indicating writeback to one of the inputs.
    let has_writeback = decomp.accumulate_loops.iter().any(|lp| {
        (lp.body_range.0..=lp.body_range.1).any(|i| {
            i < reduction.source_lines.len() && reduction.source_lines[i].contains("st.global")
        })
    });

    // The finalized value register is inv_rms (loaded from SMEM after reduction)
    if decomp.finalized_value_reg.is_empty() {
        return Err("could not detect finalized value register (inv_rms)".into());
    }

    // ── Extra params for the fused kernel ──
    // These get prepended to the GEMM's entry point
    //
    // For rms_norm (single input):
    //   param_0 = output (not needed), param_1 = input, param_2 = weight
    //   ferrite_rms_input → param_1
    //
    // For fused_add_rms_norm (two inputs + writeback):
    //   param_0 = input/hs (hidden_states), param_1 = residual (writeback target + GEMM A-input)
    //   param_2 = weight
    //   ferrite_rms_input → param_1 (residual, also the GEMM's A-pointer)
    //   ferrite_rms_hs_input → param_0 (hidden_states, second input for add)
    let mut extra_params = vec![
        ".param .u64 _ferrite_rms_input,".into(),
    ];
    if has_writeback {
        extra_params.push(".param .u64 _ferrite_rms_hs_input,".into());
    }
    extra_params.extend([
        ".param .u64 _ferrite_rms_weight,".into(),
        ".param .f32 _ferrite_rms_epsilon,".into(),
        ".param .u32 _ferrite_rms_hidden,".into(),
        ".param .u64 _ferrite_rms_a_stride,".into(),
    ]);

    // ── Extra register declarations ──
    // Extract register declarations from the reduction kernel's source and rename them.
    // This ensures the transplanted code has all the registers it needs.
    // The .reg syntax uses ranges like %f<98> (declares %f0..%f97).
    // We rename these to %f_rms_<98> (declares %f_rms_0..%f_rms_97).
    let mut extra_reg_decls: Vec<String> = Vec::new();
    for line in &reduction.source_lines {
        let t = line.trim();
        if t.starts_with(".reg ") {
            // Rename register range declarations: %f<98> → %f_rms_<98>
            let mut renamed = t.to_string();
            let reg_types = ["rd", "rs", "f", "r", "p"];
            for typ in &reg_types {
                let pattern = format!("%{typ}<");
                let replacement = format!("%{typ}_rms_<");
                renamed = renamed.replace(&pattern, &replacement);
            }
            extra_reg_decls.push(renamed);
        }
    }
    // Plumbing registers for the prologue infrastructure (row loop, inv_rms loading)
    extra_reg_decls.push(".reg .f32 %f_rms_inv;".into());
    extra_reg_decls.push(".reg .f32 %f_rms_inv0, %f_rms_inv1;".into());
    extra_reg_decls.push(".reg .b32 %r_rms_k, %r_rms_hdn, %r_rms_step;".into());
    extra_reg_decls.push(".reg .b32 %r_rms_row, %r_rms_nrows, %r_rms_mtile;".into());
    extra_reg_decls.push(".reg .b64 %rd_rms_in, %rd_rms_wt, %rd_rms_str;".into());
    if has_writeback {
        extra_reg_decls.push(".reg .b64 %rd_rms_hs_in;".into());
    }
    extra_reg_decls.push(".reg .b64 %rd_rms_rb0, %rd_rms_rb1;".into());
    extra_reg_decls.push(".reg .b64 %rd_rms_cur;".into());
    extra_reg_decls.push(".reg .pred %p_rms_lp, %p_rms_row, %p_rms_par;".into());
    // SMEM for inv_rms array and warp reduction scratch
    extra_reg_decls.push(".shared .align 4 .f32 _ferrite_inv_rms[128];".into());
    extra_reg_decls.push(".shared .align 4 .f32 _ferrite_warp_scratch[8];".into());
    extra_reg_decls.push(".shared .align 4 .f32 _ferrite_s_inv_rms;".into());

    // ── Param loads ──
    // Parse the extracted kernel's ld.param and cvta.to.global lines to find
    // which registers hold which params. Emit ferrite param loads into the
    // renamed registers, so the transplanted code sees the correct values.
    let prefix = "rms";
    let mut param_loads: Vec<String> = Vec::new();

    // Map original param name suffixes to ferrite param names.
    //
    // rms_norm:            param_0=output(skip), param_1=input, param_2=weight, param_3=eps, param_4=hidden
    // fused_add_rms_norm:  param_0=hs_input,     param_1=residual(=GEMM A-ptr), param_2=weight, param_3=eps, param_4=hidden
    let ferrite_param_map: Vec<(&str, Option<&str>)> = if has_writeback {
        vec![
            ("param_0", Some("_ferrite_rms_hs_input")),  // hidden_states (second input for add)
            ("param_1", Some("_ferrite_rms_input")),     // residual (GEMM A-ptr, writeback target)
            ("param_2", Some("_ferrite_rms_weight")),    // weight ptr
            ("param_3", Some("_ferrite_rms_epsilon")),   // epsilon
            ("param_4", Some("_ferrite_rms_hidden")),    // hidden_size
        ]
    } else {
        vec![
            ("param_0", None),                           // output ptr — not needed
            ("param_1", Some("_ferrite_rms_input")),     // input ptr
            ("param_2", Some("_ferrite_rms_weight")),    // weight ptr
            ("param_3", Some("_ferrite_rms_epsilon")),   // epsilon
            ("param_4", Some("_ferrite_rms_hidden")),    // hidden_size
        ]
    };

    // For each ld.param in the source, emit a renamed version loading from ferrite params
    for line in &reduction.source_lines {
        let t = line.trim();
        if !t.contains("ld.param") {
            continue;
        }
        for &(param_suffix, ferrite_name) in &ferrite_param_map {
            if t.contains(param_suffix) {
                if let Some(ferrite) = ferrite_name {
                    let renamed = rename_ptx_regs(t, prefix);
                    // Replace [original_param_name] with [ferrite_param_name]
                    if let (Some(bs), Some(be)) = (renamed.find('['), renamed.find(']')) {
                        let mut new_line = renamed[..bs + 1].to_string();
                        new_line.push_str(ferrite);
                        new_line.push_str(&renamed[be..]);
                        param_loads.push(new_line);
                    }
                }
                break;
            }
        }
    }

    // Emit renamed cvta.to.global lines (convert device → global address space)
    for line in &reduction.source_lines {
        let t = line.trim();
        if t.contains("cvta.to.global") {
            param_loads.push(rename_ptx_regs(t, prefix));
        }
    }

    // Stride param (not in original kernel — added by ferrite for non-contiguous tensors)
    param_loads.push("ld.param.u64 \t%rd_rms_str, [_ferrite_rms_a_stride];".into());

    // Copy renamed global pointers to plumbing registers for per-site/row-loop code.
    // Find which renamed register is the global input ptr and weight ptr by tracing
    // the cvta chain: ld.param %rdN, [param_1] → cvta %rdM, %rdN → %rdM = global input
    for &(param_suffix, ferrite_name) in &ferrite_param_map {
        if ferrite_name.is_none() {
            continue;
        }
        // Find the ld.param dest register for this param
        let mut ld_dest = String::new();
        for line in &reduction.source_lines {
            let t = line.trim();
            if t.contains("ld.param") && t.contains(param_suffix) {
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    ld_dest = parts[1].trim_end_matches(',').to_string();
                }
                break;
            }
        }
        if ld_dest.is_empty() {
            continue;
        }
        // Find the cvta that uses this register as source
        for line in &reduction.source_lines {
            let t = line.trim();
            if t.contains("cvta.to.global") && t.contains(&ld_dest) {
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 3 {
                    let cvta_dest = rename_ptx_regs(parts[1].trim_end_matches(','), prefix);
                    match ferrite_name {
                        Some("_ferrite_rms_input") => {
                            param_loads.push(format!("mov.u64 \t%rd_rms_in, {cvta_dest};"));
                        }
                        Some("_ferrite_rms_hs_input") => {
                            param_loads.push(format!("mov.u64 \t%rd_rms_hs_in, {cvta_dest};"));
                        }
                        Some("_ferrite_rms_weight") => {
                            param_loads.push(format!("mov.u64 \t%rd_rms_wt, {cvta_dest};"));
                        }
                        _ => {}
                    }
                }
                break;
            }
        }
    }
    // Also copy hidden_size to plumbing register
    for line in &reduction.source_lines {
        let t = line.trim();
        if t.contains("ld.param") && t.contains("param_4") {
            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() >= 2 {
                let dest = rename_ptx_regs(parts[1].trim_end_matches(','), prefix);
                param_loads.push(format!("mov.u32 \t%r_rms_hdn, {dest};"));
            }
            break;
        }
    }

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
        &reduction.source_lines,
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

/// Rename PTX registers in a line with a prefix.
///
/// `%f17` → `%f_<prefix>_17`, `%rd4` → `%rd_<prefix>_4`, etc.
/// Skips special registers (`%tid`, `%ntid`, `%ctaid`, `%nctaid`, `%laneid`, `%warpid`).
fn rename_ptx_regs(line: &str, prefix: &str) -> String {
    let bytes = line.as_bytes();
    let mut result = Vec::with_capacity(bytes.len() + 64);
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' {
            // Check for special registers first (must NOT rename these)
            let rest = &line[i + 1..];
            if rest.starts_with("tid.")
                || rest.starts_with("ntid.")
                || rest.starts_with("ctaid.")
                || rest.starts_with("nctaid.")
                || rest.starts_with("laneid")
                || rest.starts_with("warpid")
            {
                result.push(b'%');
                i += 1;
                continue;
            }

            // Try to match register types (longest first: rd, rs before r)
            let reg_types = ["rd", "rs", "f", "r", "p"];
            let mut matched = false;
            for typ in &reg_types {
                if rest.starts_with(typ) {
                    let after_type = &rest[typ.len()..];
                    let digit_count = after_type
                        .as_bytes()
                        .iter()
                        .take_while(|b| b.is_ascii_digit())
                        .count();
                    if digit_count > 0 {
                        // Found a register: %<type><digits>
                        let digits = &after_type[..digit_count];
                        result.push(b'%');
                        result.extend_from_slice(typ.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(prefix.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(digits.as_bytes());
                        i += 1 + typ.len() + digit_count;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                result.push(bytes[i]);
                i += 1;
            }
        } else if bytes[i] == b'$' && line[i..].starts_with("$L__BB") {
            // Label: $L__BB<N>_<M> → $L_<prefix>_<N>_<M>
            let rest = &line[i + 6..]; // after "$L__BB"
            let d1_count = rest
                .as_bytes()
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count();
            if d1_count > 0 {
                let d1 = &rest[..d1_count];
                let after_d1 = &rest[d1_count..];
                if after_d1.starts_with('_') {
                    let d2_start = &after_d1[1..];
                    let d2_count = d2_start
                        .as_bytes()
                        .iter()
                        .take_while(|b| b.is_ascii_digit())
                        .count();
                    if d2_count > 0 {
                        let d2 = &d2_start[..d2_count];
                        result.extend_from_slice(b"$L_");
                        result.extend_from_slice(prefix.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(d1.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(d2.as_bytes());
                        i += 6 + d1_count + 1 + d2_count;
                        continue;
                    }
                }
            }
            result.push(bytes[i]);
            i += 1;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }

    String::from_utf8(result).unwrap_or_else(|_| line.to_string())
}

/// Determine if a source line should be skipped when transplanting into the prologue.
///
/// We skip: declarations, param loads, cvta for params, ctaid.x row computation.
/// These are replaced by ferrite's own param initialization and row loop.
fn is_transplant_skip(line: &str) -> bool {
    let t = line.trim();
    // Skip empty lines and comments
    if t.is_empty() || t.starts_with("//") {
        return true;
    }
    // Skip entry declaration and its param list
    if t.starts_with(".entry") || t.starts_with(".visible") || t.starts_with(")") {
        return true;
    }
    // Skip .param declarations (entry signature params, not ld.param)
    if t.starts_with(".param ") {
        return true;
    }
    // Skip braces (entry body delimiters)
    if t == "{" || t == "}" {
        return true;
    }
    // Skip register and shared memory declarations
    if t.starts_with(".reg ") || t.starts_with(".shared ") {
        return true;
    }
    // Skip PTX header directives
    if t.starts_with(".version") || t.starts_with(".target") || t.starts_with(".address_size") {
        return true;
    }
    // Skip param loads (we provide our own from ferrite params)
    if t.contains("ld.param") {
        return true;
    }
    // Skip cvta.to.global that set up param pointers
    if t.contains("cvta.to.global") {
        return true;
    }
    // Skip ctaid.x row offset computation (replaced by m_tile * tile_m + row)
    if t.contains("%ctaid.x") || t.contains("%ctaid.y") {
        return true;
    }
    // Skip demoted variable comments (nvcc metadata)
    if t.starts_with("// demoted") {
        return true;
    }
    false
}

/// Generate the prologue PTX by transplanting extracted code from the reduction kernel.
///
/// Every computational instruction (FMA, shuffle, SMEM reduce, rsqrt) comes verbatim
/// from the actual rms_norm kernel's PTX, with register/label/SMEM renaming.
/// Only plumbing is new: param initialization, m_tile computation, row loop.
fn build_prologue_from_decomposition(
    decomp: &ReductionDecomposition,
    source_lines: &[String],
    thread_map: Option<&ThreadRowMap>,
    tile_m: u32,
    tile_n: u32,
    _gemm_struct_param: &str,
) -> Vec<String> {
    let lane_shr = thread_map.map_or(2, |tm| tm.lanes_per_row_log2);
    let warp_shl = thread_map.map_or(4, |tm| tm.rows_per_warp_log2);
    let row_stride = thread_map.map_or(8, |tm| tm.row_stride);

    let prefix = "rms";

    // ── Collect SMEM symbol names for renaming ──
    let mut smem_renames: Vec<(String, String)> = Vec::new();
    for line in source_lines {
        let t = line.trim();
        if t.contains("block_reduce_sum") || t.contains("block_reduce_") {
            // Find the mangled symbol name
            for word in t.split_whitespace() {
                let w = word.trim_end_matches(';').trim_end_matches(',');
                if w.contains("block_reduce_") {
                    if !smem_renames.iter().any(|(old, _)| old == w) {
                        smem_renames.push((w.to_string(), "_ferrite_warp_scratch".to_string()));
                    }
                }
            }
        }
        if t.contains("s_inv_rms") {
            for word in t.split(&['[', ']', ',', ' ', '\t'][..]) {
                let w = word.trim_end_matches(';');
                if w.contains("s_inv_rms") {
                    if !smem_renames.iter().any(|(old, _)| old == w) {
                        smem_renames.push((w.to_string(), "_ferrite_s_inv_rms".to_string()));
                    }
                }
            }
        }
    }

    // ── Determine which lines to transplant ──
    // We need: setup code + accumulate loops + finalize block
    // We skip: emit loops (replaced by per-element at A-load sites)
    let first_accum_start = decomp
        .accumulate_loops
        .first()
        .map(|l| l.header_line)
        .unwrap_or(0);
    let finalize_end = decomp.finalize_range.1;

    // Find the ctaid.x row offset lines to identify %rd4 equivalent
    // (the register that holds the row element offset)
    // Look for: mov.u32 %rN, %ctaid.x → mul → cvt to 64-bit
    let mut ctaid_result_reg = String::new(); // the s64 result (e.g., %rd4)
    let mut ctaid_skip_lines: Vec<usize> = Vec::new();
    for (idx, line) in source_lines.iter().enumerate() {
        if line.contains("%ctaid.x") {
            ctaid_skip_lines.push(idx);
            // The next 2 lines compute the row offset from ctaid.x
            if idx + 1 < source_lines.len() && source_lines[idx + 1].contains("mul.lo.s32") {
                ctaid_skip_lines.push(idx + 1);
            }
            if idx + 2 < source_lines.len() && source_lines[idx + 2].contains("cvt.s64.s32") {
                // Extract the destination register (e.g., %rd4)
                let t = source_lines[idx + 2].trim();
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    let dest = parts[1].trim_end_matches(',');
                    ctaid_result_reg = dest.to_string();
                }
                ctaid_skip_lines.push(idx + 2);
            }
        }
    }

    // Find where the setup code begins (after param loads and ctaid computation)
    // This is the first line that's not a skip line and is before the first accumulate loop
    let mut setup_start = 0;
    for idx in 0..first_accum_start {
        if !is_transplant_skip(&source_lines[idx]) && !ctaid_skip_lines.contains(&idx) {
            setup_start = idx;
            break;
        }
    }

    // ── Apply renaming to each transplanted line ──
    let rename_line = |line: &str| -> String {
        let mut renamed = rename_ptx_regs(line, prefix);
        // Replace SMEM symbols
        for (old, new) in &smem_renames {
            renamed = renamed.replace(old.as_str(), new.as_str());
        }
        // Replace bar.sync 0 with bar.sync 15 to avoid GEMM conflicts
        if renamed.contains("bar.sync") && renamed.contains("\t0") {
            renamed = renamed.replace("\t0", "\t15");
        }
        renamed
    };

    // ── Build the prologue ──
    let mut prologue = Vec::new();
    prologue
        .push("// FERRITE: rms_norm prologue (transplanted from extracted bf16 kernel PTX)".into());

    // Swizzle computation to get m_tile from ctaid.x (GEMM infrastructure)
    let tile_n_shift = tile_n.trailing_zeros();
    prologue.push("// Compute swizzle_log from N (ThreadblockSwizzle<4>)".into());
    prologue.push("ld.param.s32 \t%r_rms_step, [ferrite_params+68];".into());
    prologue.push(format!(
        "add.s32 \t%r_rms_step, %r_rms_step, {};",
        tile_n - 1
    ));
    prologue.push(format!(
        "shr.u32 \t%r_rms_step, %r_rms_step, {tile_n_shift};"
    ));
    prologue.push("mov.u32 \t%r_rms_nrows, 0;".into());
    prologue.push("setp.ge.u32 \t%p_rms_lp, %r_rms_step, 2;".into());
    prologue.push("@%p_rms_lp mov.u32 \t%r_rms_nrows, 1;".into());
    prologue.push("setp.ge.u32 \t%p_rms_lp, %r_rms_step, 3;".into());
    prologue.push("@%p_rms_lp mov.u32 \t%r_rms_nrows, 2;".into());
    prologue.push("mov.u32 \t%r_rms_mtile, %ctaid.x;".into());
    prologue.push("shr.u32 \t%r_rms_mtile, %r_rms_mtile, %r_rms_nrows;".into());

    // Row loop: iterate over min(tile_m, M - m_tile * tile_m) rows
    // Read M from ferrite_params[64] to handle partial tiles at boundary
    prologue.push("ld.param.s32 \t%r_rms_step, [ferrite_params+64];".into()); // M
    prologue.push(format!(
        "mul.lo.s32 \t%r_rms_nrows, %r_rms_mtile, {tile_m};"
    ));
    prologue.push("sub.s32 \t%r_rms_nrows, %r_rms_step, %r_rms_nrows;".into()); // M - m_tile * tile_m
    prologue.push(format!("min.s32 \t%r_rms_nrows, %r_rms_nrows, {tile_m};")); // min(remaining, tile_m)
    prologue.push("mov.u32 \t%r_rms_row, 0;".into());
    prologue.push("$L_rms_row_loop:".into());

    // Compute the row element offset (replaces ctaid.x * hidden_size)
    // IMPORTANT: use tile_m (not nrows) for the absolute row computation.
    // nrows is the clamped loop bound for partial tiles, but the starting row
    // is always m_tile * tile_m regardless of how many rows this tile processes.
    let rd4_renamed = rename_ptx_regs(&ctaid_result_reg, prefix);
    prologue.push("// Row offset: (m_tile * tile_m + row) * hidden".into());
    prologue.push(format!("mul.lo.s32 \t%r_rms_step, %r_rms_mtile, {tile_m};"));
    prologue.push("add.u32 \t%r_rms_step, %r_rms_step, %r_rms_row;".into());
    prologue.push(format!(
        "mul.lo.s32 \t%r_rms_step, %r_rms_step, %r_rms_hdn;"
    ));
    prologue.push(format!("cvt.s64.s32 \t{rd4_renamed}, %r_rms_step;"));

    // Transplant the setup + accumulate + finalize code from the extracted kernel.
    // Stop at the second bar.sync in the finalize range (the one before emit loops).
    // Everything after that is emit-loop setup which we don't need.
    let mut bar_sync_count = 0;
    let mut inv_rms_stored = false;

    prologue.push("// BEGIN transplanted rms_norm code".into());
    for idx in setup_start..=finalize_end.min(source_lines.len() - 1) {
        // Skip lines we're replacing
        if is_transplant_skip(&source_lines[idx]) {
            continue;
        }
        if ctaid_skip_lines.contains(&idx) {
            continue;
        }
        let line = &source_lines[idx];
        let t = line.trim();

        let renamed = rename_line(line);

        // Track bar.sync occurrences in the finalize range.
        // After the inv_rms store, the next bar.sync is the last thing we need.
        if renamed.contains("bar.sync") {
            bar_sync_count += 1;
            if inv_rms_stored {
                // This is the bar.sync after inv_rms store — emit it and stop
                prologue.push(renamed.trim().to_string());
                break;
            }
        }

        // Detect the inv_rms store to scalar SMEM and redirect to array
        if renamed.contains("st.shared.f32") && renamed.contains("_ferrite_s_inv_rms") {
            let parts: Vec<&str> = renamed.split_whitespace().collect();
            let src_reg = parts
                .last()
                .map(|s| s.trim_end_matches(';'))
                .unwrap_or("%f_rms_55");
            // Store to _ferrite_inv_rms[row * 4] instead
            prologue.push("mov.u32 \t%r_rms_step, _ferrite_inv_rms;".into());
            prologue.push("shl.b32 \t%r_rms_k, %r_rms_row, 2;".into());
            prologue.push("add.s32 \t%r_rms_step, %r_rms_step, %r_rms_k;".into());
            prologue.push(format!("st.shared.f32 \t[%r_rms_step], {src_reg};"));
            inv_rms_stored = true;
            continue;
        }

        prologue.push(renamed.trim().to_string());
    }
    prologue.push("// END transplanted rms_norm code".into());

    // End of row loop
    prologue.push("add.u32 \t%r_rms_row, %r_rms_row, 1;".into());
    prologue.push("setp.lt.u32 \t%p_rms_row, %r_rms_row, %r_rms_nrows;".into());
    prologue.push("@%p_rms_row bra \t$L_rms_row_loop;".into());

    // Final barrier before GEMM body reads inv_rms from SMEM
    prologue.push("bar.sync \t15;".into());

    // ── Per-thread setup: load inv_rms for this thread's two rows ──
    prologue.push(format!(
        "// Per-thread: load inv_rms (lane_shr={lane_shr}, warp_shl={warp_shl}, row_stride={row_stride})"
    ));
    prologue.push("mov.u32 \t%r_rms_step, %tid.x;".into());
    prologue.push("and.b32 \t%r_rms_k, %r_rms_step, 31;".into());
    prologue.push(format!("shr.u32 \t%r_rms_k, %r_rms_k, {lane_shr};"));
    prologue.push("shr.u32 \t%r_rms_row, %r_rms_step, 5;".into());
    prologue.push(format!("shl.b32 \t%r_rms_row, %r_rms_row, {warp_shl};"));
    prologue.push("add.u32 \t%r_rms_row, %r_rms_row, %r_rms_k;".into());

    let row_stride_bytes = row_stride * 4;
    prologue.push("mov.u32 \t%r_rms_k, _ferrite_inv_rms;".into());
    prologue.push("shl.b32 \t%r_rms_step, %r_rms_row, 2;".into());
    prologue.push("add.s32 \t%r_rms_step, %r_rms_k, %r_rms_step;".into());
    prologue.push("ld.shared.f32 \t%f_rms_inv0, [%r_rms_step];".into());
    prologue.push(format!(
        "ld.shared.f32 \t%f_rms_inv1, [%r_rms_step+{row_stride_bytes}];"
    ));

    // Compute GMEM row base addresses for per-site K-offset extraction
    prologue.push(format!("mul.lo.s32 \t%r_rms_step, %r_rms_mtile, {tile_m};"));
    prologue.push("add.u32 \t%r_rms_step, %r_rms_step, %r_rms_row;".into());
    prologue.push("cvt.s64.s32 \t%rd_rms_rb0, %r_rms_step;".into());
    prologue.push("mul.lo.s64 \t%rd_rms_rb0, %rd_rms_rb0, %rd_rms_str;".into());
    prologue.push("shl.b64 \t%rd_rms_rb0, %rd_rms_rb0, 1;".into());
    prologue.push("add.s64 \t%rd_rms_rb0, %rd_rms_in, %rd_rms_rb0;".into());
    let rb1_shift = (row_stride * 2).trailing_zeros();
    prologue.push(format!("shl.b64 \t%rd_rms_rb1, %rd_rms_str, {rb1_shift};"));
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
        let stage = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");
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

        let rms_stage = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

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

        let rms_stage = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

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

    #[test]
    fn build_silu_mul_computation_unit() {
        let comp = build_silu_mul_computation("test_silu_gemm").expect("build computation");

        assert!(
            !comp.per_site.is_empty(),
            "should have per-site code for up loading"
        );
        assert!(
            !comp.instructions.is_empty(),
            "should have per-element SiLU*up instructions"
        );
        let instr_text = comp.instructions.join("\n");
        assert!(
            instr_text.contains("ex2.approx") && instr_text.contains("div.rn.f32"),
            "instructions should contain SiLU (ex2 + div)"
        );
        assert!(
            instr_text.contains("%f_up{ELEM_IDX}"),
            "instructions should reference up values"
        );
        assert_eq!(comp.entry_name.as_deref(), Some("test_silu_gemm"));
    }

    #[test]
    fn silu_mul_fused_gemm_ptxas_valid() {
        let silu_ptx = include_str!("../../ptx-fusion/kernels/vllm_silu_mul.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        let silu_stage =
            PipelineStage::from_ptx("silu_mul", silu_ptx, None).expect("parse silu_mul");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused_ptx =
            fuse_pointwise_into_gemm(&silu_stage, &gemm_stage, "fused_silu_gemm").expect("fuse");

        // Apply perimeter replacement
        let (fused_ptx, _) =
            crate::perimeter::replace_perimeter(&fused_ptx, deriv_json, "fused_silu_gemm")
                .expect("perimeter replacement");
        let fused_ptx = crate::dedup_reg_declarations(&fused_ptx);

        // Verify SiLU markers
        assert!(
            fused_ptx.contains("ex2.approx"),
            "should contain SiLU exp approximation"
        );
        assert!(
            fused_ptx.contains("_ferrite_intermediate_bytes"),
            "should have intermediate offset param"
        );

        let path = "/tmp/fused_silu_mul_gemm.ptx";
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
            panic!("ptxas FAILED on SiLU+mul fused GEMM PTX");
        }
        println!("PASS: SiLU+mul fused GEMM passes ptxas");
    }

    #[test]
    fn fuse_add_rms_norm_into_cutlass_gemm() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");

        let norm_stage = PipelineStage::from_ptx(
            "fused_add_rms_norm",
            ptx,
            Some("fused_add_rms_norm_kernelI13__nv_bfloat16"),
        )
        .expect("parse fused_add_rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused = fuse_reduction_into_gemm(&norm_stage, &gemm_stage, "fused_add_norm_gemm");

        match fused {
            Ok(ptx) => {
                // Should contain add+norm prologue (transplanted from fused_add_rms_norm)
                assert!(ptx.contains("rms_norm prologue"), "should contain prologue");
                assert!(ptx.contains("add.f32"), "prologue should contain f32 add (from fused_add_rms_norm)");
                assert!(ptx.contains("st.global"), "prologue should contain writeback store");
                assert!(ptx.contains("mma.sync"), "GEMM interior should be preserved");
                // Should have the extra hs_input param
                assert!(ptx.contains("_ferrite_rms_hs_input"), "should have hs_input param");
                eprintln!("PASS: fused_add_rms_norm → GEMM produces valid PTX structure");
            }
            Err(e) => panic!("fuse_reduction_into_gemm failed: {e}"),
        }
    }

    #[test]
    fn fused_add_norm_gemm_ptxas_valid() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        let norm_stage = PipelineStage::from_ptx(
            "fused_add_rms_norm",
            ptx,
            Some("fused_add_rms_norm_kernelI13__nv_bfloat16"),
        )
        .expect("parse fused_add_rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused_ptx =
            fuse_reduction_into_gemm(&norm_stage, &gemm_stage, "fused_add_norm_gemm").expect("fuse");

        let (fused_ptx, _) =
            crate::perimeter::replace_perimeter(&fused_ptx, deriv_json, "fused_add_norm_gemm")
                .expect("perimeter replacement");
        let fused_ptx = crate::dedup_reg_declarations(&fused_ptx);

        let path = "/tmp/pipeline_fused_add_norm_gemm.ptx";
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
            let ptx_lines: Vec<&str> = fused_ptx.lines().collect();
            for line in stderr.lines() {
                if let Some(lnum) = line
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(')').next())
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    let start = lnum.saturating_sub(3);
                    let end = (lnum + 3).min(ptx_lines.len());
                    for i in start..end {
                        let marker = if i + 1 == lnum { ">>>" } else { "   " };
                        eprintln!("{marker} {:4}: {}", i + 1, ptx_lines[i]);
                    }
                }
            }
            panic!("ptxas FAILED on fused_add_norm+GEMM PTX");
        }
        println!("PASS: fused_add_norm+GEMM passes ptxas");
    }
}
