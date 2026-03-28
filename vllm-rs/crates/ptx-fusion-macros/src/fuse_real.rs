//! Fusion engine for real nvcc-compiled PTX kernels.
//!
//! Handles vectorized loads/stores, nvcc address patterns (cvta, add.s64),
//! and multi-pass kernels with SMEM reductions.

use std::collections::BTreeMap;

use crate::parser::{KernelParam, PtxParser};

pub struct RealFuseBinding {
    /// Param name (or substring) in A that is the output
    pub a_output_param: String,
    /// Param name (or substring) in B that is the input
    pub b_input_param: String,
}

/// How nvcc represents the row element offset in PTX.
#[derive(Debug, Clone)]
pub(crate) enum RowOffset {
    /// Pattern: mul.lo.s32 %r -> cvt.s64.s32 %rd
    /// row_global_base = cvta + (%rd << 2)
    Converted {
        reg_64: String,
        #[allow(dead_code)]
        reg_32: String,
    },
    /// Pattern: mul.lo.s32 %r -> mul.wide.s32 %rd, %r, 4
    /// row_global_base = cvta + %rd (already byte-scaled)
    WideMul { reg_64: String, reg_32: String },
}

pub struct RealFusedKernel {
    pub ptx: String,
}

/// Fuse two real nvcc-compiled kernels via SMEM handoff.
///
/// Both kernels must be "one block per row" — blockIdx.x selects the row,
/// threads cooperatively process elements within the row.
///
/// The approach for address rewriting:
/// 1. Compute `row_global_base = global_base + row_elem_offset * 4` once
/// 2. For each st.global/ld.global on the bound param, replace with:
///    `smem_addr = smem_base + (global_addr - row_global_base)` (truncated to 32-bit)
///
/// This works without understanding the address chain structure — we just
/// subtract the row's global base to get the row-local byte offset.
pub fn fuse_real_kernels(
    ptx_a: &str,
    ptx_b: &str,
    fused_name: &str,
    binding: &RealFuseBinding,
    smem_elements: usize,
) -> Result<RealFusedKernel, String> {
    let proto_a = PtxParser::parse(ptx_a)?;
    let proto_b = PtxParser::parse(ptx_b)?;
    let lines_a: Vec<&str> = ptx_a.lines().collect();
    let lines_b: Vec<&str> = ptx_b.lines().collect();

    // Trace param registers
    let reg_to_param_a = PtxParser::trace_param_registers_pub(&lines_a, &proto_a.params);
    let reg_to_param_b = PtxParser::trace_param_registers_pub(&lines_b, &proto_b.params);

    // Find A's output param and its base register chain
    let a_output_param_name = find_param_by_substring(&proto_a.params, &binding.a_output_param)
        .ok_or_else(|| format!("no param matching '{}' in kernel A", binding.a_output_param))?;
    let b_input_param_name = find_param_by_substring(&proto_b.params, &binding.b_input_param)
        .ok_or_else(|| format!("no param matching '{}' in kernel B", binding.b_input_param))?;

    // Find address registers for the bound params
    let a_output_addr_regs: Vec<String> = reg_to_param_a
        .iter()
        .filter(|(_, p)| **p == a_output_param_name)
        .map(|(r, _)| r.clone())
        .collect();
    let b_input_addr_regs: Vec<String> = reg_to_param_b
        .iter()
        .filter(|(_, p)| **p == b_input_param_name)
        .map(|(r, _)| r.clone())
        .collect();

    if a_output_addr_regs.is_empty() {
        return Err(format!(
            "could not trace address registers for A's output param '{a_output_param_name}'"
        ));
    }
    if b_input_addr_regs.is_empty() {
        return Err(format!(
            "could not trace address registers for B's input param '{b_input_param_name}'"
        ));
    }

    // Find the cvta register (the global-space base pointer) for A's output
    let a_output_cvta_reg = find_cvta_register(&lines_a, &a_output_param_name)
        .ok_or("could not find cvta register for A's output")?;

    // Find the row element offset register
    let a_row_offset =
        find_row_offset_register(&lines_a).ok_or("could not find row offset register in A")?;

    // Same for B
    let b_input_cvta_reg = find_cvta_register(&lines_b, &b_input_param_name)
        .ok_or("could not find cvta register for B's input")?;
    let b_row_offset =
        find_row_offset_register(&lines_b).ok_or("could not find row offset register in B")?;

    // Extract bodies
    let body_a_lines = extract_body_lines(ptx_a)?;
    let body_b_lines = extract_body_lines(ptx_b)?;

    // Compute register offsets for B
    let reg_offsets = compute_register_offsets(&proto_a.registers, &proto_b.registers);

    // B's registers after renaming
    let b_input_addr_regs_renamed: Vec<String> = b_input_addr_regs
        .iter()
        .map(|r| offset_single_register(r, &reg_offsets))
        .collect();
    let b_input_cvta_renamed = offset_single_register(&b_input_cvta_reg, &reg_offsets);
    let b_row_offset_renamed = match &b_row_offset {
        RowOffset::Converted { reg_64, .. } => RowOffset::Converted {
            reg_64: offset_single_register(reg_64, &reg_offsets),
            reg_32: String::new(),
        },
        RowOffset::WideMul { reg_64, reg_32 } => RowOffset::WideMul {
            reg_64: offset_single_register(reg_64, &reg_offsets),
            reg_32: offset_single_register(reg_32, &reg_offsets),
        },
    };

    // Merged params
    let merged_params = merge_params(
        &proto_a,
        &proto_b,
        &a_output_param_name,
        &b_input_param_name,
    );
    let merged_regs =
        compute_merged_reg_decls(&proto_a.registers, &proto_b.registers, &reg_offsets);

    // Max barrier from A
    let max_barrier_a = proto_a.barriers.iter().max().copied().unwrap_or(0);
    let fusion_barrier = max_barrier_a + 1;
    let b_barrier_offset = fusion_barrier + 1;

    let smem_handoff = "_ferrite_handoff";

    // Collect all .shared declarations from both kernels
    let a_shared_decls = extract_shared_decls(ptx_a);
    let b_shared_decls = extract_shared_decls(ptx_b);

    // Find the label suffix used by each kernel (nvcc uses $L__BB0_N, $L__BB1_N etc.)
    let _a_label_prefix = find_label_prefix(&body_a_lines).unwrap_or("$L__BB0".to_string());
    let b_label_prefix = find_label_prefix(&body_b_lines).unwrap_or("$L__BB0".to_string());

    // ── Build fused PTX ──
    let mut out = String::new();

    // Use the version/target from kernel A
    for line in ptx_a.lines() {
        let t = line.trim();
        if t.starts_with(".version") || t.starts_with(".target") || t.starts_with(".address_size") {
            out.push_str(t);
            out.push('\n');
        }
        if t.starts_with(".address_size") {
            break;
        }
    }
    out.push('\n');

    // Top-level .shared declarations from both kernels
    for decl in &a_shared_decls {
        out.push_str(decl);
        out.push('\n');
    }
    for decl in &b_shared_decls {
        // Prefix B's shared names to avoid collision
        let renamed = prefix_shared_names(decl, "_b_");
        out.push_str(&renamed);
        out.push('\n');
    }
    out.push('\n');

    // Entry point
    out.push_str(&format!(".visible .entry {fused_name}(\n"));
    for (i, p) in merged_params.iter().enumerate() {
        let comma = if i + 1 < merged_params.len() { "," } else { "" };
        out.push_str(&format!("\t.param {} {}{}\n", p.ptx_type, p.name, comma));
    }
    out.push_str(")\n{\n");

    // Register declarations
    for (ty, prefix, count) in &merged_regs {
        out.push_str(&format!("\t.reg {ty} \t{prefix}<{count}>;\n"));
    }
    // Ferrite scratch registers
    out.push_str("\t.reg .u32 \t%r_fe<8>;\n");
    out.push_str("\t.reg .b64 \t%rd_fe<4>;\n");

    // SMEM declarations from inside kernel bodies (skip ones already declared at top level)
    let top_level_names: Vec<String> = a_shared_decls
        .iter()
        .chain(b_shared_decls.iter())
        .filter_map(|d| extract_decl_name(d))
        .collect();
    emit_body_shared_decls_dedup(&mut out, &body_a_lines, &top_level_names);
    // B's body shared decls get prefixed to avoid collision
    emit_body_shared_decls_renamed_dedup(&mut out, &body_b_lines, "_b_", &top_level_names);

    // Handoff buffer
    out.push_str(&format!(
        "\t.shared .align 16 .f32 {smem_handoff}[{smem_elements}];\n"
    ));
    out.push('\n');

    // ── Phase A ──
    out.push_str(&format!("\t// ===== PHASE A: {} =====\n", proto_a.name));

    // Track whether we've emitted the row_global_base computation yet
    let mut emitted_row_base = false;

    for line in &body_a_lines {
        let trimmed = line.trim();

        // Skip .shared declarations inside the body (already handled above)
        if trimmed.starts_with(".shared") || trimmed.starts_with("// demoted") {
            continue;
        }

        // Skip ld.param for the bound output (it's gone from the fused params)
        // Instead, zero the register so cvta doesn't fault
        if trimmed.contains("ld.param") && trimmed.contains(&a_output_param_name) {
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let dest = parts[1].trim_end_matches(',');
                out.push_str(&format!(
                    "\t// FERRITE: [{}] removed (SMEM handoff)\n",
                    a_output_param_name
                ));
                out.push_str(&format!("\tmov.u64 \t{dest}, 0;\n"));
            }
            continue;
        }

        // After the cvta for the output base, emit the row_global_base computation
        // (we need %rd1 and %rd4 to be defined first)
        if !emitted_row_base
            && trimmed.contains("cvta.to.global")
            && trimmed.contains(&a_output_cvta_reg)
        {
            out.push_str(&format!("\t{trimmed}\n"));
            continue;
        }

        // Detect when the row offset register is defined, then emit row_global_base
        if !emitted_row_base {
            let trigger = match &a_row_offset {
                RowOffset::Converted { reg_64, .. } => {
                    trimmed.contains("cvt.s64.s32") && trimmed.contains(reg_64.as_str())
                }
                RowOffset::WideMul { reg_64, .. } => {
                    trimmed.starts_with("mul.wide.s32") && trimmed.contains(reg_64.as_str())
                }
            };
            if trigger {
                out.push_str(&format!("\t{trimmed}\n"));
                out.push_str("\t// FERRITE: compute row_global_base for SMEM handoff\n");
                match &a_row_offset {
                    RowOffset::Converted { reg_64, .. } => {
                        out.push_str(&format!("\tshl.b64 \t%rd_fe0, {reg_64}, 2;\n"));
                        out.push_str(&format!(
                            "\tadd.s64 \t%rd_fe1, {a_output_cvta_reg}, %rd_fe0;\n"
                        ));
                    }
                    RowOffset::WideMul { reg_64, .. } => {
                        // mul.wide already did *4, so just add to cvta base
                        out.push_str(&format!(
                            "\tadd.s64 \t%rd_fe1, {a_output_cvta_reg}, {reg_64};\n"
                        ));
                    }
                }
                emitted_row_base = true;
                continue;
            }
        }

        // Rewrite st.global on the bound output → st.shared via SMEM handoff
        if is_global_store(trimmed) && is_addr_in_set(trimmed, &a_output_addr_regs) {
            emit_smem_store_real(&mut out, trimmed, smem_handoff);
            continue;
        }

        // Redirect labels and ret for phase A
        if trimmed == "ret;" {
            continue;
        }
        if is_label(trimmed) && trimmed.contains("ret") {
            out.push_str(&format!("\t{trimmed}\n"));
            continue;
        }

        out.push_str(&format!("\t{trimmed}\n"));
    }

    // Fusion barrier
    out.push_str(&format!("\n\tbar.sync \t{};\n\n", fusion_barrier));

    // ── Phase B ──
    out.push_str(&format!(
        "\t// ===== PHASE B: {} (registers offset) =====\n",
        proto_b.name
    ));

    let b_label_replacement = "$L__BB1".to_string();
    let mut emitted_b_row_base = false;

    for line in &body_b_lines {
        let trimmed = line.trim();

        // Skip .shared declarations inside B's body
        if trimmed.starts_with(".shared") || trimmed.starts_with("// demoted") {
            continue;
        }

        let renamed = offset_all_registers(trimmed, &reg_offsets);
        let renamed = rename_b_shared_refs(&renamed, &b_shared_decls);
        let renamed = offset_barriers(&renamed, &proto_b.barriers, b_barrier_offset);
        let renamed = renamed.replace(&b_label_prefix, &b_label_replacement);
        let rtrimmed = renamed.trim();

        // Skip ld.param for the bound input; zero the register
        if trimmed.contains("ld.param") && trimmed.contains(&b_input_param_name) {
            let parts: Vec<&str> = rtrimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let dest = parts[1].trim_end_matches(',');
                out.push_str(&format!(
                    "\t// FERRITE: [{}] removed (SMEM handoff)\n",
                    b_input_param_name
                ));
                out.push_str(&format!("\tmov.u64 \t{dest}, 0;\n"));
            }
            continue;
        }

        // After the row offset register is defined, emit row_global_base for B
        if !emitted_b_row_base {
            let trigger = match &b_row_offset_renamed {
                RowOffset::Converted { reg_64, .. } => {
                    rtrimmed.contains("cvt.s64.s32") && rtrimmed.contains(reg_64.as_str())
                }
                RowOffset::WideMul { reg_64, .. } => {
                    rtrimmed.starts_with("mul.wide.s32") && rtrimmed.contains(reg_64.as_str())
                }
            };
            if trigger {
                out.push_str(&format!("\t{rtrimmed}\n"));
                out.push_str("\t// FERRITE: compute row_global_base for B's input\n");
                match &b_row_offset_renamed {
                    RowOffset::Converted { reg_64, .. } => {
                        out.push_str(&format!("\tshl.b64 \t%rd_fe2, {reg_64}, 2;\n"));
                        out.push_str(&format!(
                            "\tadd.s64 \t%rd_fe3, {b_input_cvta_renamed}, %rd_fe2;\n"
                        ));
                    }
                    RowOffset::WideMul { reg_64, .. } => {
                        out.push_str(&format!(
                            "\tadd.s64 \t%rd_fe3, {b_input_cvta_renamed}, {reg_64};\n"
                        ));
                    }
                }
                emitted_b_row_base = true;
                continue;
            }
        }

        // Rewrite ld.global on the bound input → ld.shared from SMEM handoff
        if is_global_load(rtrimmed) && is_addr_in_set(rtrimmed, &b_input_addr_regs_renamed) {
            emit_smem_load_real(&mut out, rtrimmed, smem_handoff);
            continue;
        }

        out.push_str(&format!("\t{rtrimmed}\n"));
    }

    out.push_str("}\n");

    Ok(RealFusedKernel { ptx: out })
}

// ── Address analysis helpers ──

/// Find the param name matching a substring.
pub(crate) fn find_param_by_substring(params: &[KernelParam], substr: &str) -> Option<String> {
    params
        .iter()
        .find(|p| p.name.contains(substr))
        .map(|p| p.name.clone())
}

/// Find the register assigned by `cvta.to.global.u64` for a given param.
/// Pattern: ld.param.u64 %rdX, [param]; cvta.to.global.u64 %rdY, %rdX;
pub(crate) fn find_cvta_register(lines: &[&str], param_name: &str) -> Option<String> {
    // First find which register the param is loaded into
    let mut param_reg = None;
    for line in lines {
        let t = line.trim();
        if t.contains("ld.param") && t.contains(param_name) {
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                param_reg = Some(parts[1].trim_end_matches(',').to_string());
                break;
            }
        }
    }
    let param_reg = param_reg?;

    // Then find the cvta that converts it
    for line in lines {
        let t = line.trim();
        if t.starts_with("cvta.to.global") && t.contains(&param_reg) {
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 3 {
                let src = parts[2].trim_end_matches(';');
                if src == param_reg {
                    return Some(parts[1].trim_end_matches(',').to_string());
                }
            }
        }
    }

    None
}

/// Find the row element offset register, searching near a specific param's cvta.
///
/// When `near_param` is Some, searches for ctaid.x AFTER the cvta line for that param.
/// This is needed for fused PTX where multiple phases each have their own ctaid.x.
pub(crate) fn find_row_offset_near(lines: &[&str], near_param: Option<&str>) -> Option<RowOffset> {
    // Find the starting line: either 0 or after the cvta for the target param
    let start_line = if let Some(param_name) = near_param {
        let mut cvta_line = 0;
        let mut param_reg = None;
        for (i, line) in lines.iter().enumerate() {
            let t = line.trim();
            if t.contains("ld.param") && t.contains(param_name) {
                let parts: Vec<&str> = t
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                    .collect();
                if parts.len() >= 2 {
                    param_reg = Some(parts[1].trim_end_matches(',').to_string());
                }
            }
            if let Some(ref pr) = param_reg
                && t.starts_with("cvta.to.global")
                && t.contains(pr.as_str())
            {
                cvta_line = i;
                break;
            }
        }
        cvta_line
    } else {
        0
    };

    find_row_offset_in_range(lines, start_line)
}

/// Find the row element offset register.
///
/// Two nvcc patterns:
/// - Pattern A: mul.lo.s32 %rN, ctaid_reg, size_reg -> cvt.s64.s32 %rdM, %rN
/// - Pattern B: mul.lo.s32 %rN, ctaid_reg, size_reg -> mul.wide.s32 %rdM, %rN, 4
fn find_row_offset_register(lines: &[&str]) -> Option<RowOffset> {
    find_row_offset_in_range(lines, 0)
}

fn find_row_offset_in_range(lines: &[&str], start: usize) -> Option<RowOffset> {
    // Find the register holding ctaid.x, starting from `start`
    let mut ctaid_reg = None;
    for line in lines.iter().skip(start) {
        let t = line.trim();
        if t.contains("ctaid.x") && (t.contains("mov.u32") || t.contains("mov.b32")) {
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 3 {
                ctaid_reg = Some(parts[1].trim_end_matches(',').to_string());
                break;
            }
        }
    }
    let ctaid_reg = ctaid_reg?;

    // Find mul.lo.s32 using ctaid_reg
    let mut mul_dst = None;
    for line in lines {
        let t = line.trim();
        if t.starts_with("mul.lo.s32") && t.contains(&ctaid_reg) {
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                mul_dst = Some(parts[1].trim_end_matches(',').to_string());
                break;
            }
        }
    }
    let mul_dst = mul_dst?;

    // Pattern A: cvt.s64.s32 from mul_dst
    for line in lines {
        let t = line.trim();
        if t.contains("cvt.s64.s32") && t.contains(&mul_dst) {
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 3 {
                let src = parts[2].trim_end_matches(';');
                if src == mul_dst {
                    return Some(RowOffset::Converted {
                        reg_64: parts[1].trim_end_matches(',').to_string(),
                        reg_32: mul_dst,
                    });
                }
            }
        }
    }

    // Pattern B: mul.wide.s32 %rdM, mul_dst, 4
    for line in lines {
        let t = line.trim();
        if t.starts_with("mul.wide.s32") && t.contains(&mul_dst) && t.contains(", 4;") {
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                return Some(RowOffset::WideMul {
                    reg_64: parts[1].trim_end_matches(',').to_string(),
                    reg_32: mul_dst,
                });
            }
        }
    }

    None
}

// ── Store/Load rewriting ──

fn is_global_store(instr: &str) -> bool {
    instr.contains("st.global")
}

fn is_global_load(instr: &str) -> bool {
    instr.contains("ld.global")
}

fn is_addr_in_set(instruction: &str, addr_regs: &[String]) -> bool {
    if let Some(bs) = instruction.find('[') {
        let be = instruction.find(']').unwrap_or(instruction.len());
        let addr = &instruction[bs + 1..be];
        let base = addr.split('+').next().unwrap_or(addr).trim();
        addr_regs.iter().any(|r| r == base)
    } else {
        false
    }
}

fn is_label(s: &str) -> bool {
    s.ends_with(':') || s.starts_with('$')
}

/// Rewrite a st.global instruction to st.shared using row_global_base subtraction.
/// Uses %rd_fe1 as the row_global_base (computed before phase A).
fn emit_smem_store_real(out: &mut String, instruction: &str, smem_name: &str) {
    // Parse: st.global[.v4].f32 [%rdX], {values} or [%rdX], %fY
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0]; // st.global.v4.f32 or st.global.f32
    let rest = if parts.len() > 1 { parts[1].trim() } else { "" };

    let shared_op = op.replace("st.global", "st.shared");

    // Extract address register and value(s)
    if let Some(bracket_start) = rest.find('[') {
        let bracket_end = rest.find(']').unwrap_or(rest.len());
        let addr_reg = rest[bracket_start + 1..bracket_end].trim();
        let values_part = rest[bracket_end + 1..]
            .trim()
            .trim_start_matches(',')
            .trim();

        out.push_str("\t// FERRITE: st.global -> st.shared (SMEM handoff)\n");
        out.push_str(&format!("\tsub.s64 \t%rd_fe0, {addr_reg}, %rd_fe1;\n"));
        out.push_str("\tcvt.u32.u64 \t%r_fe0, %rd_fe0;\n");
        out.push_str(&format!("\tmov.u32 \t%r_fe1, {smem_name};\n"));
        out.push_str("\tadd.u32 \t%r_fe2, %r_fe1, %r_fe0;\n");
        out.push_str(&format!("\t{shared_op} \t[%r_fe2], {values_part}\n"));
    } else {
        out.push_str(&format!("\t{instruction}\n"));
    }
}

/// Rewrite a ld.global instruction to ld.shared using row_global_base subtraction.
/// Uses %rd_fe3 as the row_global_base (computed before phase B).
fn emit_smem_load_real(out: &mut String, instruction: &str, smem_name: &str) {
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0]; // ld.global.nc.v4.f32 or ld.global.nc.f32
    let rest = if parts.len() > 1 { parts[1].trim() } else { "" };

    // For shared: drop .nc, replace .global with .shared
    let shared_op = op.replace(".nc", "").replace("ld.global", "ld.shared");

    // Extract dest and address
    if let Some(bracket_start) = rest.find('[') {
        let bracket_end = rest.find(']').unwrap_or(rest.len());
        let addr_reg = rest[bracket_start + 1..bracket_end].trim();
        let dest_part = rest[..bracket_start].trim().trim_end_matches(',').trim();
        let after = rest[bracket_end + 1..].trim();

        out.push_str("\t// FERRITE: ld.global -> ld.shared (SMEM handoff)\n");
        out.push_str(&format!("\tsub.s64 \t%rd_fe0, {addr_reg}, %rd_fe3;\n"));
        out.push_str("\tcvt.u32.u64 \t%r_fe0, %rd_fe0;\n");
        out.push_str(&format!("\tmov.u32 \t%r_fe1, {smem_name};\n"));
        out.push_str("\tadd.u32 \t%r_fe2, %r_fe1, %r_fe0;\n");
        if after.is_empty() {
            out.push_str(&format!("\t{shared_op} \t{dest_part}, [%r_fe2];\n"));
        } else {
            out.push_str(&format!("\t{shared_op} \t{dest_part}, [%r_fe2]{after}\n"));
        }
    } else {
        out.push_str(&format!("\t{instruction}\n"));
    }
}

// ── Body extraction ──

pub(crate) fn extract_body_lines(ptx: &str) -> Result<Vec<String>, String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let mut start = None;
    let mut end = None;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if (t == "{" || t.ends_with('{')) && start.is_none() {
            start = Some(i + 1);
        }
        if t == "}" {
            end = Some(i);
        }
    }
    let s = start.ok_or("no '{'")?;
    let e = end.ok_or("no '}'")?;
    Ok(lines[s..e]
        .iter()
        .filter(|l| {
            let t = l.trim();
            !t.starts_with(".reg") && !t.is_empty()
        })
        .map(|l| l.to_string())
        .collect())
}

fn extract_shared_decls(ptx: &str) -> Vec<String> {
    ptx.lines()
        .filter(|l| {
            let t = l.trim();
            // Top-level .shared declarations (not inside entry body)
            t.starts_with(".shared") || (t.starts_with(".extern") && t.contains(".shared"))
        })
        .map(|l| l.to_string())
        .collect()
}

fn emit_body_shared_decls_dedup(
    out: &mut String,
    body_lines: &[String],
    already_declared: &[String],
) {
    for line in body_lines {
        let t = line.trim();
        if t.starts_with(".shared")
            && let Some(name) = extract_decl_name(t)
            && !already_declared.contains(&name)
        {
            out.push_str(&format!("\t{t}\n"));
        }
    }
}

fn emit_body_shared_decls_renamed_dedup(
    out: &mut String,
    body_lines: &[String],
    prefix: &str,
    already_declared: &[String],
) {
    for line in body_lines {
        let t = line.trim();
        if t.starts_with(".shared")
            && let Some(name) = extract_decl_name(t)
            && !already_declared.contains(&name)
        {
            let renamed = prefix_shared_names(t, prefix);
            out.push_str(&format!("\t{renamed}\n"));
        }
    }
}

pub(crate) fn prefix_shared_names_pub(decl: &str, prefix: &str) -> String {
    prefix_shared_names(decl, prefix)
}

fn prefix_shared_names(decl: &str, prefix: &str) -> String {
    // Find the symbol name and prefix it
    let parts: Vec<&str> = decl.split_whitespace().collect();
    let mut result = decl.to_string();
    for part in &parts {
        let clean = part.trim_end_matches(';').trim_end_matches(']');
        if !clean.starts_with('.')
            && !clean.starts_with('[')
            && clean.contains(|c: char| c.is_alphabetic())
            && !clean.starts_with("//")
        {
            // This is likely the symbol name
            let name = if let Some(bracket) = clean.find('[') {
                &clean[..bracket]
            } else {
                clean
            };
            if !name.is_empty() && !name.starts_with(prefix) {
                result = result.replace(name, &format!("{prefix}{name}"));
                break;
            }
        }
    }
    result
}

fn rename_b_shared_refs(line: &str, b_shared_decls: &[String]) -> String {
    let mut result = line.to_string();
    for decl in b_shared_decls {
        if let Some(name) = extract_decl_name(decl)
            && result.contains(&name)
        {
            result = result.replace(&name, &format!("_b_{name}"));
        }
    }
    result
}

pub(crate) fn extract_decl_name_pub(decl: &str) -> Option<String> {
    extract_decl_name(decl)
}

fn extract_decl_name(decl: &str) -> Option<String> {
    let parts: Vec<&str> = decl.split_whitespace().collect();
    for part in &parts {
        let clean = part.trim_end_matches(';');
        if !clean.starts_with('.') && clean.contains(|c: char| c.is_alphabetic()) {
            let name = if let Some(bracket) = clean.find('[') {
                &clean[..bracket]
            } else {
                clean
            };
            return Some(name.to_string());
        }
    }
    None
}

pub(crate) fn find_label_prefix(body_lines: &[String]) -> Option<String> {
    for line in body_lines {
        let t = line.trim();
        if let Some(pos) = t.find("$L__BB") {
            let rest = &t[pos..];
            // Extract $L__BBN (number after BB)
            let end = rest
                .find('_')
                .unwrap_or(rest.find(':').unwrap_or(rest.len()));
            if end > 6 {
                return Some(rest[..end].to_string());
            }
        }
    }
    None
}

// ── Register offset helpers (same as in fuse.rs) ──

pub(crate) fn compute_register_offsets(
    a_regs: &[(String, usize)],
    b_regs: &[(String, usize)],
) -> BTreeMap<String, usize> {
    let a_map: BTreeMap<&str, usize> = a_regs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let mut offsets = BTreeMap::new();
    for (reg_type, _) in b_regs {
        let a_count = a_map.get(reg_type.as_str()).copied().unwrap_or(0);
        offsets.insert(reg_type.clone(), a_count);
    }
    offsets
}

pub(crate) fn compute_merged_reg_decls(
    a_regs: &[(String, usize)],
    b_regs: &[(String, usize)],
    offsets: &BTreeMap<String, usize>,
) -> Vec<(String, String, usize)> {
    let type_to_prefix: &[(&str, &str)] = &[
        (".b32", "%r"),
        (".b64", "%rd"),
        (".f32", "%f"),
        (".f64", "%fd"),
        (".u32", "%r"),
        (".u64", "%rd"),
        (".pred", "%p"),
        (".b16", "%rs"),
    ];
    let b_map: BTreeMap<&str, usize> = b_regs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let a_map: BTreeMap<&str, usize> = a_regs.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    let mut all_types: BTreeMap<String, ()> = BTreeMap::new();
    for (ty, _) in a_regs {
        all_types.insert(ty.clone(), ());
    }
    for (ty, _) in b_regs {
        all_types.insert(ty.clone(), ());
    }

    let mut result = Vec::new();
    for ty in all_types.keys() {
        if let Some((_, prefix)) = type_to_prefix.iter().find(|(t, _)| *t == ty.as_str()) {
            let a_count = a_map.get(ty.as_str()).copied().unwrap_or(0);
            let b_count = b_map.get(ty.as_str()).copied().unwrap_or(0);
            let offset = offsets.get(ty.as_str()).copied().unwrap_or(a_count);
            result.push((ty.clone(), prefix.to_string(), offset + b_count));
        }
    }
    result
}

pub(crate) fn merge_params(
    proto_a: &crate::parser::KernelProtocol,
    proto_b: &crate::parser::KernelProtocol,
    a_output_name: &str,
    b_input_name: &str,
) -> Vec<KernelParam> {
    let mut params: Vec<KernelParam> = proto_a
        .params
        .iter()
        .filter(|p| p.name != a_output_name)
        .cloned()
        .collect();
    for bp in &proto_b.params {
        if bp.name == b_input_name {
            continue;
        }
        if !params.iter().any(|p| p.name == bp.name) {
            params.push(bp.clone());
        }
    }
    for (i, p) in params.iter_mut().enumerate() {
        p.index = i;
    }
    params
}

pub(crate) fn offset_all_registers(line: &str, offsets: &BTreeMap<String, usize>) -> String {
    // Process in order: longer prefixes first
    let prefixes = [
        (".b64", "%rd"),
        (".u64", "%rd"),
        (".f32", "%f"),
        (".pred", "%p"),
        (".b16", "%rs"),
        (".b32", "%r"),
        (".u32", "%r"),
    ];
    let mut result = line.to_string();
    for (ty, prefix) in &prefixes {
        if let Some(offset) = offsets.get(*ty)
            && *offset > 0
        {
            result = offset_registers_in_text(&result, prefix, *offset);
        }
    }
    result
}

fn offset_registers_in_text(text: &str, prefix: &str, offset: usize) -> String {
    if offset == 0 {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len() + 64);
    let mut remaining = text;
    while let Some(pos) = remaining.find(prefix) {
        if prefix == "%r" && remaining[pos..].starts_with("%rd") {
            result.push_str(&remaining[..pos + 3]);
            remaining = &remaining[pos + 3..];
            continue;
        }
        if prefix == "%r" && remaining[pos..].starts_with("%rs") {
            result.push_str(&remaining[..pos + 3]);
            remaining = &remaining[pos + 3..];
            continue;
        }
        if prefix == "%r" && remaining[pos..].starts_with("%r_fe") {
            result.push_str(&remaining[..pos + 5]);
            remaining = &remaining[pos + 5..];
            continue;
        }
        result.push_str(&remaining[..pos]);
        let after_prefix = &remaining[pos + prefix.len()..];
        let digit_end = after_prefix
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after_prefix.len());
        if digit_end > 0
            && let Ok(num) = after_prefix[..digit_end].parse::<usize>()
        {
            result.push_str(prefix);
            result.push_str(&(num + offset).to_string());
            remaining = &after_prefix[digit_end..];
            continue;
        }
        result.push_str(prefix);
        remaining = after_prefix;
    }
    result.push_str(remaining);
    result
}

pub(crate) fn offset_single_register(reg: &str, offsets: &BTreeMap<String, usize>) -> String {
    let prefixes = [
        ("%rd", ".b64"),
        ("%rs", ".b16"),
        ("%f", ".f32"),
        ("%p", ".pred"),
        ("%r", ".b32"),
    ];
    for (prefix, ty) in &prefixes {
        if prefix == &"%r"
            && (reg.starts_with("%rd") || reg.starts_with("%rs") || reg.starts_with("%r_fe"))
        {
            continue;
        }
        if let Some(num_str) = reg.strip_prefix(prefix)
            && let Ok(num) = num_str.parse::<usize>()
        {
            let off = offsets.get(*ty).copied().unwrap_or(0);
            return format!("{prefix}{}", num + off);
        }
    }
    reg.to_string()
}

pub(crate) fn offset_barriers(line: &str, b_barriers: &[usize], new_start: usize) -> String {
    let mut result = line.to_string();
    let mut sorted: Vec<usize> = b_barriers.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    for (i, &old_id) in sorted.iter().enumerate() {
        result = result.replace(
            &format!("bar.sync \t{old_id}"),
            &format!("bar.sync \t{}", new_start + i),
        );
    }
    result
}
