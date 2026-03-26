//! Chain a third phase onto an already-fused kernel via SMEM handoff.
//!
//! Phase B's output stores are redirected to a second SMEM buffer.
//! Phase C's input loads read from that buffer.
//! Both handoffs (Phase A->B and Phase B->C) go through SMEM.

use crate::fuse_real::{
    RowOffset, compute_register_offsets, extract_body_lines, find_cvta_register, find_label_prefix,
    find_param_by_substring, find_row_offset_near, offset_all_registers, offset_barriers,
    offset_single_register,
};
use crate::parser::{KernelParam, PtxParser};
#[allow(unused_imports)]
use std::collections::BTreeMap;

/// Append a third phase to an already-fused kernel, with SMEM handoff.
///
/// Phase B's output stores are redirected to SMEM. Phase C's input loads
/// read from that SMEM buffer. Both inter-phase handoffs eliminate GMEM traffic.
pub fn append_phase_smem(
    fused_ptx: &str,
    new_kernel_ptx: &str,
    b_output_param: &str,
    c_input_param: &str,
    new_name: &str,
    smem_elements: usize,
) -> Result<String, String> {
    let proto_fused = PtxParser::parse(fused_ptx)?;
    let proto_c = PtxParser::parse(new_kernel_ptx)?;
    let lines_fused: Vec<&str> = fused_ptx.lines().collect();
    let lines_c: Vec<&str> = new_kernel_ptx.lines().collect();

    // Find the actual param names
    let b_output_name = find_param_by_substring(&proto_fused.params, b_output_param)
        .ok_or_else(|| format!("no param matching '{b_output_param}' in fused kernel"))?;
    let c_input_name = find_param_by_substring(&proto_c.params, c_input_param)
        .ok_or_else(|| format!("no param matching '{c_input_param}' in Phase C"))?;

    // Trace address registers for Phase B's output and Phase C's input
    let reg_to_param_fused =
        PtxParser::trace_param_registers_pub(&lines_fused, &proto_fused.params);
    let reg_to_param_c = PtxParser::trace_param_registers_pub(&lines_c, &proto_c.params);

    let b_output_addr_regs: Vec<String> = reg_to_param_fused
        .iter()
        .filter(|(_, p)| **p == b_output_name)
        .map(|(r, _)| r.clone())
        .collect();
    let c_input_addr_regs: Vec<String> = reg_to_param_c
        .iter()
        .filter(|(_, p)| **p == c_input_name)
        .map(|(r, _)| r.clone())
        .collect();

    if b_output_addr_regs.is_empty() {
        return Err(format!(
            "could not trace address registers for Phase B output '{b_output_name}'"
        ));
    }
    if c_input_addr_regs.is_empty() {
        return Err(format!(
            "could not trace address registers for Phase C input '{c_input_name}'"
        ));
    }

    // Find cvta and row offset for Phase B's OUTPUT
    let b_output_cvta = find_cvta_register(&lines_fused, &b_output_name)
        .ok_or("could not find cvta for Phase B output")?;
    // Trace backward from the output stores to find the 32-bit row offset register
    let b_output_row_reg_32 =
        find_row_offset_for_stores(&lines_fused, &b_output_addr_regs, &b_output_cvta)
            .ok_or("could not find row offset for Phase B output stores")?;

    // Find cvta and row offset for Phase C's input (fresh kernel, normal search)
    let c_input_cvta = find_cvta_register(&lines_c, &c_input_name)
        .ok_or("could not find cvta for Phase C input")?;
    let c_row_offset =
        find_row_offset_near(&lines_c, None).ok_or("could not find row offset in Phase C")?;

    // Register offsets for Phase C
    let reg_offsets = compute_register_offsets(&proto_fused.registers, &proto_c.registers);

    // Renamed C registers
    let c_input_addr_regs_renamed: Vec<String> = c_input_addr_regs
        .iter()
        .map(|r| offset_single_register(r, &reg_offsets))
        .collect();
    let c_input_cvta_renamed = offset_single_register(&c_input_cvta, &reg_offsets);
    let c_row_offset_renamed = match &c_row_offset {
        RowOffset::Converted { reg_64, .. } => RowOffset::Converted {
            reg_64: offset_single_register(reg_64, &reg_offsets),
            reg_32: String::new(),
        },
        RowOffset::WideMul { reg_64, reg_32 } => RowOffset::WideMul {
            reg_64: offset_single_register(reg_64, &reg_offsets),
            reg_32: offset_single_register(reg_32, &reg_offsets),
        },
    };

    // Merged params with collision renaming
    let (merged_params, param_renames) = merge_params_chain(
        &proto_fused.params,
        &proto_c.params,
        &b_output_name,
        &c_input_name,
    );

    // Barriers
    let max_barrier_fused = proto_fused.barriers.iter().max().copied().unwrap_or(1);
    let chain_barrier = max_barrier_fused + 1;
    let c_barrier_offset = chain_barrier + 1;

    // Phase C body and labels
    let body_c_lines = extract_body_lines(new_kernel_ptx)?;
    let c_label_prefix = find_label_prefix(&body_c_lines).unwrap_or("$L__BB0".to_string());
    let c_label_replacement = "$L__BB2".to_string();

    let smem_handoff_2 = "_ferrite_handoff_2";

    // Build output PTX
    let mut out = String::new();

    // Version/target
    for line in fused_ptx.lines() {
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

    // Top-level .shared from fused PTX
    for line in fused_ptx.lines() {
        let t = line.trim();
        if t.contains(".entry") {
            break;
        }
        if t.starts_with(".shared") || (t.starts_with(".extern") && t.contains(".shared")) {
            out.push_str(t);
            out.push('\n');
        }
    }
    out.push('\n');

    // Entry with merged params
    out.push_str(&format!(".visible .entry {new_name}(\n"));
    for (i, p) in merged_params.iter().enumerate() {
        let comma = if i + 1 < merged_params.len() { "," } else { "" };
        out.push_str(&format!("\t.param {} {}{}\n", p.ptx_type, p.name, comma));
    }
    out.push_str(")\n{\n");

    // Register declarations: expand for Phase C
    let fused_reg_lines = extract_reg_lines(fused_ptx);
    emit_expanded_reg_decls(&mut out, &fused_reg_lines, &proto_c.registers, &reg_offsets);

    // Chain-specific scratch registers (separate from fuse_real's %r_fe/%rd_fe)
    out.push_str("\t// FERRITE: chain scratch registers for SMEM handoff 2\n");
    out.push_str("\t.reg .u32 \t%r_ch<4>;\n");
    out.push_str("\t.reg .b64 \t%rd_ch<2>;\n");

    // Body-level .shared from fused kernel + second handoff buffer
    let fused_body = extract_body_lines(fused_ptx)?;
    for line in &fused_body {
        let t = line.trim();
        if t.starts_with(".shared") {
            out.push_str(&format!("\t{t}\n"));
        }
    }
    out.push_str(&format!(
        "\t.shared .align 16 .f32 {smem_handoff_2}[{smem_elements}];\n"
    ));
    out.push('\n');

    // ── Phases A + B (with B's output stores redirected to SMEM) ──
    // Emit row_global_base for Phase B's output when the row offset reg is defined
    let mut emitted_b_output_row_base = false;

    for line in &fused_body {
        let t = line.trim();
        if t.starts_with(".shared") || t.starts_with(".reg") {
            continue;
        }
        if t == "ret;" {
            continue;
        }

        // Detect when Phase B's row offset register is defined (mul.lo.s32 that produces it)
        if !emitted_b_output_row_base
            && t.starts_with("mul.lo.s32")
            && t.contains(&b_output_row_reg_32)
        {
            // Check this is the defining instruction (dest = our register)
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 && parts[1].trim_end_matches(',') == b_output_row_reg_32 {
                out.push_str(&format!("\t{t}\n"));
                out.push_str(
                    "\t// FERRITE: row_global_base for Phase B output -> SMEM handoff 2\n",
                );
                out.push_str(&format!(
                    "\tmul.wide.s32 \t%rd_ch0, {b_output_row_reg_32}, 4;\n"
                ));
                out.push_str(&format!("\tadd.s64 \t%rd_ch1, {b_output_cvta}, %rd_ch0;\n"));
                emitted_b_output_row_base = true;
                continue;
            }
        }

        // Intercept ld.param for the removed output param (now goes through SMEM)
        if t.contains("ld.param") && t.contains(&b_output_name) {
            let parts: Vec<&str> = t
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let dest = parts[1].trim_end_matches(',');
                out.push_str(&format!(
                    "\t// FERRITE: [{b_output_name}] removed (SMEM handoff 2)\n"
                ));
                out.push_str(&format!("\tmov.u64 \t{dest}, 0;\n"));
            }
            continue;
        }

        // Redirect Phase B's output st.global -> st.shared
        if is_global_store(t) && is_addr_in_set(t, &b_output_addr_regs) {
            emit_smem_store(&mut out, t, smem_handoff_2);
            continue;
        }

        out.push_str(&format!("\t{t}\n"));
    }

    // ── Chain barrier ──
    out.push_str(&format!("\n\tbar.sync \t{chain_barrier};\n\n"));

    // ── Phase C (reads from SMEM handoff 2) ──
    out.push_str(&format!(
        "\t// ===== PHASE C: {} (registers offset, SMEM input) =====\n",
        proto_c.name
    ));

    let mut emitted_c_row_base = false;

    for line in &body_c_lines {
        let trimmed = line.trim();
        if trimmed.starts_with(".shared") || trimmed.starts_with(".reg") {
            continue;
        }

        let mut renamed = offset_all_registers(trimmed, &reg_offsets);
        renamed = offset_barriers(&renamed, &proto_c.barriers, c_barrier_offset);
        renamed = renamed.replace(&c_label_prefix, &c_label_replacement);
        for (old_name, new_name_r) in &param_renames {
            renamed = renamed.replace(old_name.as_str(), new_name_r.as_str());
        }
        let rtrimmed = renamed.trim();

        // Skip ld.param for the bound input; zero the register
        if trimmed.contains("ld.param") && trimmed.contains(&c_input_name) {
            let parts: Vec<&str> = rtrimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let dest = parts[1].trim_end_matches(',');
                out.push_str(&format!(
                    "\t// FERRITE: [{c_input_name}] removed (SMEM handoff 2)\n"
                ));
                out.push_str(&format!("\tmov.u64 \t{dest}, 0;\n"));
            }
            continue;
        }

        // After cvta for C's input, detect row offset and emit row_global_base
        if !emitted_c_row_base {
            let trigger = match &c_row_offset_renamed {
                RowOffset::Converted { reg_64, .. } => {
                    rtrimmed.contains("cvt.s64.s32") && rtrimmed.contains(reg_64.as_str())
                }
                RowOffset::WideMul { reg_64, .. } => {
                    rtrimmed.starts_with("mul.wide.s32") && rtrimmed.contains(reg_64.as_str())
                }
            };
            if trigger {
                out.push_str(&format!("\t{rtrimmed}\n"));
                out.push_str(
                    "\t// FERRITE: row_global_base for Phase C input from SMEM handoff 2\n",
                );
                match &c_row_offset_renamed {
                    RowOffset::Converted { reg_64, .. } => {
                        out.push_str(&format!("\tshl.b64 \t%rd_ch0, {reg_64}, 2;\n"));
                        out.push_str(&format!(
                            "\tadd.s64 \t%rd_ch1, {c_input_cvta_renamed}, %rd_ch0;\n"
                        ));
                    }
                    RowOffset::WideMul { reg_64, .. } => {
                        out.push_str(&format!(
                            "\tadd.s64 \t%rd_ch1, {c_input_cvta_renamed}, {reg_64};\n"
                        ));
                    }
                }
                emitted_c_row_base = true;
                continue;
            }
        }

        // Redirect C's input ld.global -> ld.shared from SMEM handoff 2
        if is_global_load(rtrimmed) && is_addr_in_set(rtrimmed, &c_input_addr_regs_renamed) {
            emit_smem_load(&mut out, rtrimmed, smem_handoff_2);
            continue;
        }

        out.push_str(&format!("\t{rtrimmed}\n"));
    }

    out.push_str("}\n");
    Ok(out)
}

// ── Helpers ──

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

/// Rewrite st.global -> st.shared using row_global_base subtraction.
/// Uses %rd_fe2 as scratch, %rd_fe3 as row_global_base.
fn emit_smem_store(out: &mut String, instruction: &str, smem_name: &str) {
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0];
    let rest = if parts.len() > 1 { parts[1].trim() } else { "" };
    let shared_op = op.replace("st.global", "st.shared");

    if let Some(bs) = rest.find('[') {
        let be = rest.find(']').unwrap_or(rest.len());
        let addr_reg = rest[bs + 1..be].trim();
        let values_part = rest[be + 1..].trim().trim_start_matches(',').trim();

        out.push_str("\t// FERRITE: st.global -> st.shared (SMEM handoff 2)\n");
        out.push_str(&format!("\tsub.s64 \t%rd_ch0, {addr_reg}, %rd_ch1;\n"));
        out.push_str("\tcvt.u32.u64 \t%r_ch0, %rd_ch0;\n");
        out.push_str(&format!("\tmov.u32 \t%r_ch1, {smem_name};\n"));
        out.push_str("\tadd.u32 \t%r_ch2, %r_ch1, %r_ch0;\n");
        out.push_str(&format!("\t{shared_op} \t[%r_ch2], {values_part}\n"));
    } else {
        out.push_str(&format!("\t{instruction}\n"));
    }
}

/// Rewrite ld.global -> ld.shared using row_global_base subtraction.
fn emit_smem_load(out: &mut String, instruction: &str, smem_name: &str) {
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0];
    let rest = if parts.len() > 1 { parts[1].trim() } else { "" };
    let shared_op = op.replace(".nc", "").replace("ld.global", "ld.shared");

    if let Some(bs) = rest.find('[') {
        let be = rest.find(']').unwrap_or(rest.len());
        let addr_reg = rest[bs + 1..be].trim();
        let dest_part = rest[..bs].trim().trim_end_matches(',').trim();
        let after = rest[be + 1..].trim();

        out.push_str("\t// FERRITE: ld.global -> ld.shared (SMEM handoff 2)\n");
        out.push_str(&format!("\tsub.s64 \t%rd_ch0, {addr_reg}, %rd_ch1;\n"));
        out.push_str("\tcvt.u32.u64 \t%r_ch0, %rd_ch0;\n");
        out.push_str(&format!("\tmov.u32 \t%r_ch1, {smem_name};\n"));
        out.push_str("\tadd.u32 \t%r_ch2, %r_ch1, %r_ch0;\n");
        if after.is_empty() {
            out.push_str(&format!("\t{shared_op} \t{dest_part}, [%r_ch2];\n"));
        } else {
            out.push_str(&format!("\t{shared_op} \t{dest_part}, [%r_ch2]{after}\n"));
        }
    } else {
        out.push_str(&format!("\t{instruction}\n"));
    }
}

/// Merge params: keep all fused params (minus B's output), add C's params (minus bound input).
/// If C's param names conflict, prefix with "_c_".
fn merge_params_chain(
    fused_params: &[KernelParam],
    c_params: &[KernelParam],
    b_output_name: &str,
    c_input_name: &str,
) -> (Vec<KernelParam>, Vec<(String, String)>) {
    // Remove B's output param (it goes through SMEM now)
    let mut params: Vec<KernelParam> = fused_params
        .iter()
        .filter(|p| p.name != b_output_name)
        .cloned()
        .collect();
    let mut renames = Vec::new();
    for cp in c_params {
        if cp.name == c_input_name {
            continue;
        }
        if params.iter().any(|p| p.name == cp.name) {
            let new_name = format!("_c_{}", cp.name);
            renames.push((cp.name.clone(), new_name.clone()));
            let mut renamed = cp.clone();
            renamed.name = new_name;
            params.push(renamed);
        } else {
            params.push(cp.clone());
        }
    }
    for (i, p) in params.iter_mut().enumerate() {
        p.index = i;
    }
    (params, renames)
}

/// Emit expanded register declarations covering fused + Phase C registers.
fn emit_expanded_reg_decls(
    out: &mut String,
    fused_reg_lines: &[String],
    c_regs: &[(String, usize)],
    offsets: &std::collections::BTreeMap<String, usize>,
) {
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

    let c_map: std::collections::BTreeMap<&str, usize> =
        c_regs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let mut emitted_types = std::collections::BTreeSet::new();

    for line in fused_reg_lines {
        let t = line.trim();
        if !t.starts_with(".reg") {
            continue;
        }

        let is_standard = t.contains('<')
            && type_to_prefix.iter().any(|(_, prefix)| {
                if let Some(pos) = t.find(prefix) {
                    let after = &t[pos + prefix.len()..];
                    after.starts_with('<')
                } else {
                    false
                }
            });

        if is_standard {
            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() >= 3 {
                let ty = parts[1];
                if let Some(angle_start) = t.find('<')
                    && let Some(angle_end) = t.find('>')
                {
                    let current: usize = t[angle_start + 1..angle_end].parse().unwrap_or(0);
                    if let Some(c_count) = c_map.get(ty) {
                        let offset = offsets.get(ty).copied().unwrap_or(0);
                        let new_count = current.max(offset + c_count);
                        if !emitted_types.contains(ty) {
                            let prefix = type_to_prefix
                                .iter()
                                .find(|(tt, _)| *tt == ty)
                                .map(|(_, p)| *p)
                                .unwrap_or("%r");
                            out.push_str(&format!("\t.reg {ty} \t{prefix}<{new_count}>;\n"));
                            emitted_types.insert(ty.to_string());
                        }
                    } else if !emitted_types.contains(ty) {
                        out.push_str(&format!("\t{t}\n"));
                        emitted_types.insert(ty.to_string());
                    }
                }
            }
        } else {
            out.push_str(&format!("\t{t}\n"));
        }
    }

    // SiLU scratch if present
    if fused_reg_lines.iter().any(|l| l.contains("%f_act")) && !out.contains("%f_act") {
        out.push_str("\t.reg .f32 \t%f_act<4>;\n");
    }
    if fused_reg_lines.iter().any(|l| l.contains("%r_act")) && !out.contains("%r_act") {
        out.push_str("\t.reg .b32 \t%r_act<2>;\n");
    }

    // Phase C register types not already present
    for (ty, count) in c_regs {
        if !emitted_types.contains(ty.as_str()) {
            let offset = offsets.get(ty.as_str()).copied().unwrap_or(0);
            if let Some((_, prefix)) = type_to_prefix.iter().find(|(t, _)| *t == ty.as_str()) {
                out.push_str(&format!("\t.reg {ty} \t{prefix}<{}>;\n", offset + count));
                emitted_types.insert(ty.clone());
            }
        }
    }
}

/// Find the row offset register by tracing backward from the output stores.
///
/// Given stores like `st.global.f32 [%rd52], %f118`, traces:
///   %rd52 = add.s64(%rd33, %rd51)     -- %rd33 is cvta, %rd51 is byte offset
///   %rd51 = mul.wide.s32(%r112, 4)    -- element index * 4
///   %r112 = add.s32(%r113, %r88)      -- col + row_offset
///   %r88  = mul.lo.s32(%r85, %r103)   -- ctaid.x * N (the row offset we need)
/// Returns the 32-bit register holding row * stride (e.g., %r88 = ctaid.x * N).
fn find_row_offset_for_stores(
    lines: &[&str],
    output_addr_regs: &[String],
    cvta_reg: &str,
) -> Option<String> {
    // Build a map: register -> defining instruction
    let mut reg_defs: BTreeMap<String, String> = BTreeMap::new();
    for line in lines {
        let t = line.trim();
        // Instructions that define a register: the dest is the first operand after the opcode
        let ops = [
            "add.s64",
            "add.s32",
            "add.u32",
            "mul.wide.s32",
            "mul.wide.u32",
            "mul.lo.s32",
            "cvt.s64.s32",
            "mov.u32",
            "mov.b32",
        ];
        for op in ops {
            if t.starts_with(op) {
                let parts: Vec<&str> = t
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                    .collect();
                if parts.len() >= 2 {
                    let dest = parts[1].trim_end_matches(',').to_string();
                    reg_defs.insert(dest, t.to_string());
                }
                break;
            }
        }
    }

    // Find a store that uses one of the output addr regs
    let mut store_addr_reg = None;
    for line in lines {
        let t = line.trim();
        if is_global_store(t)
            && is_addr_in_set(t, output_addr_regs)
            && let Some(bs) = t.find('[')
        {
            let be = t.find(']').unwrap_or(t.len());
            let addr = t[bs + 1..be].trim();
            let base = addr.split('+').next().unwrap_or(addr).trim();
            store_addr_reg = Some(base.to_string());
            break;
        }
    }
    let store_addr = store_addr_reg?;

    // Trace backward: store_addr = add.s64(cvta, byte_offset)
    let addr_def = reg_defs.get(&store_addr)?;
    if !addr_def.contains("add.s64") {
        return None;
    }
    let parts: Vec<&str> = addr_def
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    // add.s64 %dest, %src1, %src2
    if parts.len() < 4 {
        return None;
    }
    let src1 = parts[2].trim_end_matches(',');
    let src2 = parts[3].trim_end_matches(';');
    // One of src1/src2 is the cvta, the other is the byte offset
    let byte_offset_reg = if src1 == cvta_reg {
        src2
    } else if src2 == cvta_reg {
        src1
    } else {
        return None;
    };

    // byte_offset = mul.wide.s32(%element_index, 4)
    let offset_def = reg_defs.get(byte_offset_reg)?;
    if !offset_def.contains("mul.wide") {
        return None;
    }
    let parts: Vec<&str> = offset_def
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() < 3 {
        return None;
    }
    let elem_index_reg = parts[2].trim_end_matches(',');

    // element_index = add.s32(col, row_offset) — or might be the mul result directly
    let index_def = reg_defs.get(elem_index_reg)?;
    let row_offset_reg = if index_def.contains("add.s32") {
        let parts: Vec<&str> = index_def
            .split([',', ' ', '\t'])
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() < 4 {
            return None;
        }
        // One operand is the column (from tid.x loop), the other is row*N
        // The row*N one is the result of mul.lo.s32 with ctaid.x
        let s1 = parts[2].trim_end_matches(',');
        let s2 = parts[3].trim_end_matches(';');
        // Check which one traces back to mul.lo.s32
        if let Some(def) = reg_defs.get(s2) {
            if def.contains("mul.lo.s32") {
                s2.to_string()
            } else {
                s1.to_string()
            }
        } else {
            s1.to_string()
        }
    } else {
        elem_index_reg.to_string()
    };

    // row_offset = mul.lo.s32(ctaid_reg, stride_reg) — verify it involves ctaid.x
    let row_def = reg_defs.get(&row_offset_reg)?;
    if !row_def.contains("mul.lo.s32") {
        return None;
    }

    // Return the 32-bit row offset register — we'll handle 64-bit conversion ourselves
    Some(row_offset_reg)
}

fn extract_reg_lines(ptx: &str) -> Vec<String> {
    let mut in_body = false;
    let mut regs = Vec::new();
    for line in ptx.lines() {
        let t = line.trim();
        if t == "{" || t.ends_with('{') {
            in_body = true;
            continue;
        }
        if t == "}" {
            break;
        }
        if in_body && t.starts_with(".reg") {
            regs.push(t.to_string());
        }
    }
    regs
}
