use std::collections::BTreeMap;

use crate::parser::{KernelParam, PtxParser};

/// Binding for register-level fusion: A's output feeds B's input directly via registers.
/// Only valid when A and B are elementwise with matching per-thread data layout.
pub struct RegFuseBinding {
    pub a_output_param: String,
    pub b_input_param: String,
}

pub struct RegFusedKernel {
    pub ptx: String,
}

/// Fuse two elementwise PTX kernels via register handoff.
///
/// Requirements:
/// - Both kernels are elementwise (thread i processes element i)
/// - A stores one value per thread to its output param
/// - B loads one value per thread from its input param
/// - No SMEM, no barriers needed — the value stays in a register
///
/// Strategy:
/// 1. Run A's body but replace `st.global` on the output with nothing (value stays in register)
/// 2. Run B's body but replace `ld.global` on the input with a `mov` from A's output register
/// 3. Skip the address computation for the eliminated load/store
pub fn regfuse_kernels(
    ptx_a: &str,
    ptx_b: &str,
    fused_name: &str,
    binding: &RegFuseBinding,
) -> Result<RegFusedKernel, String> {
    let proto_a = PtxParser::parse(ptx_a)?;
    let proto_b = PtxParser::parse(ptx_b)?;
    let lines_a: Vec<&str> = ptx_a.lines().collect();
    let lines_b: Vec<&str> = ptx_b.lines().collect();

    // Validate
    let _a_store = proto_a
        .global_stores
        .iter()
        .find(|s| s.param_name == binding.a_output_param)
        .ok_or_else(|| format!("A has no global store to '{}'", binding.a_output_param))?;
    let _b_load = proto_b
        .global_loads
        .iter()
        .find(|l| l.param_name == binding.b_input_param)
        .ok_or_else(|| format!("B has no global load from '{}'", binding.b_input_param))?;

    // Both must be elementwise (no SMEM, no barriers, no MMA)
    if !proto_a.smem_regions.is_empty() || !proto_a.barriers.is_empty() {
        return Err("register fusion requires A to be elementwise (no SMEM/barriers)".into());
    }
    if !proto_b.smem_regions.is_empty() || !proto_b.barriers.is_empty() {
        return Err("register fusion requires B to be elementwise (no SMEM/barriers)".into());
    }

    // Find A's store instruction to identify the value register
    let body_a = extract_body_lines(ptx_a)?;
    let body_b = extract_body_lines(ptx_b)?;

    let reg_to_param_a = PtxParser::trace_param_registers_pub(&lines_a, &proto_a.params);
    let reg_to_param_b = PtxParser::trace_param_registers_pub(&lines_b, &proto_b.params);

    let a_output_addr_regs: Vec<String> = reg_to_param_a
        .iter()
        .filter(|(_, p)| **p == binding.a_output_param)
        .map(|(r, _)| r.clone())
        .collect();
    let b_input_addr_regs: Vec<String> = reg_to_param_b
        .iter()
        .filter(|(_, p)| **p == binding.b_input_param)
        .map(|(r, _)| r.clone())
        .collect();

    // Find the value register in A's store: st.global.f32 [%rdX], %fY → %fY
    let a_value_reg = find_store_value_register(&body_a, &a_output_addr_regs)
        .ok_or("could not find value register in A's store")?;

    // Find the dest register in B's load: ld.global.f32 %fZ, [%rdW] → %fZ
    let b_dest_reg = find_load_dest_register(&body_b, &b_input_addr_regs)
        .ok_or("could not find dest register in B's load")?;

    // Compute register offsets for B
    let reg_offsets = compute_register_offsets(&proto_a.registers, &proto_b.registers);

    // After renaming, the value register in A stays the same, but B's dest gets offset
    let b_dest_reg_renamed = offset_single_register(&b_dest_reg, &reg_offsets);

    // Merged params
    let merged_params = merge_params(&proto_a, &proto_b, binding);
    let merged_regs =
        compute_merged_reg_decls(&proto_a.registers, &proto_b.registers, &reg_offsets);

    // ── Build fused PTX ──
    let mut out = String::new();
    out.push_str(".version 7.0\n.target sm_80\n.address_size 64\n\n");

    out.push_str(&format!(".visible .entry {fused_name}(\n"));
    for (i, p) in merged_params.iter().enumerate() {
        let comma = if i + 1 < merged_params.len() { "," } else { "" };
        out.push_str(&format!("    .param {} {}{}\n", p.ptx_type, p.name, comma));
    }
    out.push_str(")\n{\n");

    for (ty, prefix, count) in &merged_regs {
        out.push_str(&format!("    .reg {ty} {prefix}<{count}>;\n"));
    }

    // ── Phase A: emit body, skip the st.global and its address computation ──
    out.push_str(&format!(
        "\n    // ===== PHASE A: {} (elementwise) =====\n",
        proto_a.name
    ));

    for line in &body_a {
        let trimmed = line.trim();

        // Skip ld.param for the output param
        if trimmed.contains("ld.param")
            && trimmed.contains(&format!("[{}]", binding.a_output_param))
        {
            out.push_str(&format!(
                "    // FERRITE: skipped ld.param [{}] (register handoff)\n",
                binding.a_output_param
            ));
            continue;
        }

        // Skip the address computation for the output (add.u64 producing the output addr reg)
        if trimmed.starts_with("add.u64") && produces_any(trimmed, &a_output_addr_regs) {
            out.push_str("    // FERRITE: skipped output addr computation (register handoff)\n");
            continue;
        }

        // Skip the st.global entirely — value stays in register
        if trimmed.contains("st.global") && is_addr_in_set(trimmed, &a_output_addr_regs) {
            out.push_str(&format!(
                "    // FERRITE: st.global eliminated - value stays in {a_value_reg}\n"
            ));
            continue;
        }

        // Redirect EXIT
        if trimmed == "ret;" {
            continue;
        }
        if trimmed.starts_with("EXIT") && trimmed.ends_with(':') {
            out.push_str("PHASE_A_EXIT:\n");
            continue;
        }
        if trimmed.contains("bra EXIT") {
            out.push_str(&format!(
                "    {}\n",
                trimmed.replace("bra EXIT", "bra PHASE_A_EXIT")
            ));
            continue;
        }

        out.push_str(&format!("    {trimmed}\n"));
    }

    // ── Phase B: emit body with registers offset, replace ld.global with mov from A's register ──
    out.push_str(&format!(
        "\n    // ===== PHASE B: {} (register handoff: {} -> {}) =====\n",
        proto_b.name, a_value_reg, b_dest_reg_renamed
    ));

    for line in &body_b {
        let trimmed = line.trim();
        let renamed = offset_all_registers(trimmed, &reg_offsets);
        let rtrimmed = renamed.trim();

        // Skip ld.param for the input param
        if trimmed.contains("ld.param") && trimmed.contains(&format!("[{}]", binding.b_input_param))
        {
            out.push_str(&format!(
                "    // FERRITE: skipped ld.param [{}] (register handoff)\n",
                binding.b_input_param
            ));
            continue;
        }

        // Skip the address computation for the input
        let b_input_addr_regs_renamed: Vec<String> = b_input_addr_regs
            .iter()
            .map(|r| offset_single_register(r, &reg_offsets))
            .collect();
        if rtrimmed.starts_with("add.u64") && produces_any(rtrimmed, &b_input_addr_regs_renamed) {
            out.push_str("    // FERRITE: skipped input addr computation (register handoff)\n");
            continue;
        }

        // Also skip mul.wide.u32 that feeds the address computation
        // (it computes the byte offset, which we no longer need for the input)
        if trimmed.contains("ld.global") && is_addr_in_set(rtrimmed, &b_input_addr_regs_renamed) {
            // Replace ld.global with mov from A's output register
            out.push_str("    // FERRITE: ld.global eliminated - register handoff\n");
            out.push_str(&format!(
                "    mov.f32 {b_dest_reg_renamed}, {a_value_reg};\n"
            ));
            continue;
        }

        // Redirect EXIT
        if rtrimmed == "ret;" {
            out.push_str("    bra FUSED_EXIT;\n");
            continue;
        }
        if trimmed.starts_with("EXIT") && trimmed.ends_with(':') {
            out.push_str("PHASE_B_EXIT:\n");
            continue;
        }
        if rtrimmed.contains("bra EXIT") {
            out.push_str(&format!(
                "    {}\n",
                rtrimmed.replace("bra EXIT", "bra PHASE_B_EXIT")
            ));
            continue;
        }

        out.push_str(&format!("    {rtrimmed}\n"));
    }

    out.push_str("\nFUSED_EXIT:\n    ret;\n}\n");

    Ok(RegFusedKernel { ptx: out })
}

// ── Helpers ──

fn extract_body_lines(ptx: &str) -> Result<Vec<String>, String> {
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
            !t.starts_with(".reg") && !t.starts_with(".shared") && !t.is_empty()
        })
        .map(|l| l.to_string())
        .collect())
}

/// Find the value register in a st.global instruction.
/// st.global.f32 [%rd6], %f7; → returns "%f7"
fn find_store_value_register(body: &[String], addr_regs: &[String]) -> Option<String> {
    for line in body {
        let trimmed = line.trim();
        if trimmed.contains("st.global") && is_addr_in_set(trimmed, addr_regs) {
            // st.global.f32 [%rd6], %f7;
            // The value is the last token before the semicolon
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if let Some(last) = parts.last() {
                return Some(last.trim_end_matches(';').to_string());
            }
        }
    }
    None
}

/// Find the dest register in a ld.global instruction.
/// ld.global.f32 %f1, [%rd3]; → returns "%f1"
fn find_load_dest_register(body: &[String], addr_regs: &[String]) -> Option<String> {
    for line in body {
        let trimmed = line.trim();
        if trimmed.contains("ld.global") && is_addr_in_set(trimmed, addr_regs) {
            // ld.global.f32 %f1, [%rd3];
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                return Some(parts[1].trim_end_matches(',').to_string());
            }
        }
    }
    None
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

/// Check if an instruction produces (writes to) any register in the set.
fn produces_any(instruction: &str, regs: &[String]) -> bool {
    // The destination is typically the second token (after the opcode):
    // add.u64 %rd6, %rd1, %rd3;  → dest is %rd6
    let parts: Vec<&str> = instruction
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() >= 2 {
        let dest = parts[1].trim_end_matches(',');
        regs.iter().any(|r| r == dest)
    } else {
        false
    }
}

fn compute_register_offsets(
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

fn compute_merged_reg_decls(
    a_regs: &[(String, usize)],
    b_regs: &[(String, usize)],
    offsets: &BTreeMap<String, usize>,
) -> Vec<(String, String, usize)> {
    let type_to_prefix: &[(&str, &str)] = &[
        (".f32", "%f"),
        (".f64", "%fd"),
        (".u32", "%r"),
        (".u64", "%rd"),
        (".pred", "%p"),
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

fn offset_all_registers(line: &str, offsets: &BTreeMap<String, usize>) -> String {
    let prefixes = [
        (".u64", "%rd"),
        (".f32", "%f"),
        (".pred", "%p"),
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
    let mut result = String::with_capacity(text.len() + 32);
    let mut remaining = text;
    while let Some(pos) = remaining.find(prefix) {
        if prefix == "%r" && remaining[pos..].starts_with("%rd") {
            result.push_str(&remaining[..pos + 3]);
            remaining = &remaining[pos + 3..];
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

fn offset_single_register(reg: &str, offsets: &BTreeMap<String, usize>) -> String {
    let prefixes = [
        ("%rd", ".u64"),
        ("%f", ".f32"),
        ("%p", ".pred"),
        ("%r", ".u32"),
    ];
    for (prefix, ty) in &prefixes {
        if prefix == &"%r" && reg.starts_with("%rd") {
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

fn merge_params(
    proto_a: &crate::parser::KernelProtocol,
    proto_b: &crate::parser::KernelProtocol,
    binding: &RegFuseBinding,
) -> Vec<KernelParam> {
    let mut params: Vec<KernelParam> = proto_a
        .params
        .iter()
        .filter(|p| p.name != binding.a_output_param)
        .cloned()
        .collect();
    for bp in &proto_b.params {
        if bp.name == binding.b_input_param {
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
