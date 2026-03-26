use std::collections::BTreeMap;

use crate::parser::{KernelParam, PtxParser};

/// A fusion binding: kernel A's output param feeds kernel B's input param via SMEM.
pub struct FuseBinding {
    pub a_output_param: String,
    pub b_input_param: String,
}

/// Result of fusing two PTX kernels.
pub struct FusedKernel {
    pub ptx: String,
}

// Dedicated registers for SMEM address computation in the fused kernel.
// These are allocated beyond both A's and B's register ranges.
const FERRITE_REG_U32_COUNT: usize = 4; // %r_fe0..%r_fe3
const FERRITE_REG_PREFIX: &str = "%r_fe";

/// Fuse two sequential PTX kernels via SMEM handoff.
///
/// Produces a single .entry that runs A's body, then a bar.sync, then B's body.
/// A's store to the bound output param is rewritten to st.shared.
/// B's load from the bound input param is rewritten to ld.shared.
/// B's registers are offset to avoid collision with A's.
pub fn fuse_kernels(
    ptx_a: &str,
    ptx_b: &str,
    fused_name: &str,
    binding: &FuseBinding,
) -> Result<FusedKernel, String> {
    let proto_a = PtxParser::parse(ptx_a)?;
    let proto_b = PtxParser::parse(ptx_b)?;
    let lines_a: Vec<&str> = ptx_a.lines().collect();
    let lines_b: Vec<&str> = ptx_b.lines().collect();

    // Validate: A stores to the bound param, B loads from the bound param
    proto_a
        .global_stores
        .iter()
        .find(|s| s.param_name == binding.a_output_param)
        .ok_or_else(|| {
            format!(
                "kernel A ({}) has no global store to param '{}'",
                proto_a.name, binding.a_output_param
            )
        })?;
    proto_b
        .global_loads
        .iter()
        .find(|l| l.param_name == binding.b_input_param)
        .ok_or_else(|| {
            format!(
                "kernel B ({}) has no global load from param '{}'",
                proto_b.name, binding.b_input_param
            )
        })?;

    // Trace which registers map to which params
    let reg_to_param_a = PtxParser::trace_param_registers_pub(&lines_a, &proto_a.params);
    let reg_to_param_b = PtxParser::trace_param_registers_pub(&lines_b, &proto_b.params);

    // Address registers for the bound params
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

    // Compute register offsets for B (shift past A's register ranges)
    let reg_offsets = compute_register_offsets(&proto_a.registers, &proto_b.registers);

    // B's input addr regs after renaming
    let b_input_addr_regs_renamed: Vec<String> = b_input_addr_regs
        .iter()
        .map(|r| offset_single_register(r, &reg_offsets))
        .collect();

    // Extract bodies
    let body_a_lines = extract_body_lines(ptx_a)?;
    let body_b_lines = extract_body_lines(ptx_b)?;

    // Find the byte-offset register used by A's store.
    // In rms_norm: st.global.f32 [%rd6], %f7
    //   %rd6 = output_base(%rd1) + byte_offset(%rd3)
    //   byte_offset = mul.wide.u32 %rd3, %r4, 4
    // We need to find which register holds the byte offset (not the base pointer).
    let a_offset_reg = find_byte_offset_register(
        &body_a_lines,
        &a_output_addr_regs,
        &reg_to_param_a,
        &binding.a_output_param,
    );

    // Same for B's load
    let b_offset_reg = find_byte_offset_register(
        &body_b_lines,
        &b_input_addr_regs,
        &reg_to_param_b,
        &binding.b_input_param,
    );
    // After renaming:
    let b_offset_reg_renamed = b_offset_reg
        .as_ref()
        .map(|r| offset_single_register(r, &reg_offsets));

    // SMEM handoff buffer
    let smem_handoff = "_ferrite_handoff";
    let smem_count = 1024u32; // max elements (covers up to 1024-thread blocks)

    // Merged params
    let merged_params = merge_params(&proto_a, &proto_b, binding);

    // Max barrier from A
    let max_barrier_a = proto_a.barriers.iter().max().copied().unwrap_or(0);
    let fusion_barrier = max_barrier_a + 1;
    let b_barrier_offset = fusion_barrier + 1;

    // Total register counts
    let merged_regs =
        compute_merged_reg_decls(&proto_a.registers, &proto_b.registers, &reg_offsets);

    // ── Build fused PTX ──

    let mut out = String::new();
    out.push_str(".version 7.0\n.target sm_80\n.address_size 64\n\n");

    // Entry point
    out.push_str(&format!(".visible .entry {fused_name}(\n"));
    for (i, p) in merged_params.iter().enumerate() {
        let comma = if i + 1 < merged_params.len() { "," } else { "" };
        out.push_str(&format!("    .param {} {}{}\n", p.ptx_type, p.name, comma));
    }
    out.push_str(")\n{\n");

    // Register declarations
    for (ty, prefix, count) in &merged_regs {
        out.push_str(&format!("    .reg {ty} {prefix}<{count}>;\n"));
    }
    // Ferrite scratch registers for SMEM addressing
    out.push_str(&format!(
        "    .reg .u32 {FERRITE_REG_PREFIX}<{FERRITE_REG_U32_COUNT}>;\n"
    ));

    // SMEM declarations: A's original
    for region in &proto_a.smem_regions {
        out.push_str(&format!(
            "    .shared .align {} {} {}[{}];\n",
            region.align, region.elem_type, region.name, region.count
        ));
    }
    // B's original (prefixed)
    for region in &proto_b.smem_regions {
        out.push_str(&format!(
            "    .shared .align {} {} _b_{}[{}];\n",
            region.align, region.elem_type, region.name, region.count
        ));
    }
    // Handoff buffer
    out.push_str(&format!(
        "    .shared .align 16 .f32 {smem_handoff}[{smem_count}];\n"
    ));

    // ── Phase A ──
    out.push_str(&format!("\n    // ===== PHASE A: {} =====\n", proto_a.name));
    for line in &body_a_lines {
        let trimmed = line.trim();

        // Skip ld.param for the bound output param (it's now internal SMEM).
        // Zero-init the register so downstream address math doesn't read undef.
        if trimmed.contains("ld.param")
            && trimmed.contains(&format!("[{}]", binding.a_output_param))
        {
            // Extract the dest register: ld.param.u64 %rd1, [output];
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let dest = parts[1].trim_end_matches(',');
                out.push_str(&format!(
                    "    // FERRITE: [{}] is now SMEM handoff\n",
                    binding.a_output_param
                ));
                out.push_str(&format!("    mov.u64 {dest}, 0;\n"));
            }
            continue;
        }

        // Rewrite A's st.global on the bound output → st.shared via handoff
        if trimmed.contains("st.global") && is_addr_in_set(trimmed, &a_output_addr_regs) {
            emit_smem_store(&mut out, trimmed, smem_handoff, a_offset_reg.as_deref());
            continue;
        }

        // Remove ret, redirect EXIT
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

    // Fusion barrier
    out.push_str(&format!("\n    bar.sync {};\n", fusion_barrier));

    // ── Phase B ──
    out.push_str(&format!(
        "\n    // ===== PHASE B: {} (registers offset +{:?}) =====\n",
        proto_b.name, reg_offsets
    ));

    for line in &body_b_lines {
        let trimmed = line.trim();

        // Rename B's registers
        let renamed = offset_all_registers(trimmed, &reg_offsets);
        let renamed = rename_smem_refs(&renamed, &proto_b.smem_regions);
        let renamed = offset_barrier_refs(&renamed, &proto_b.barriers, b_barrier_offset);

        let rtrimmed = renamed.trim();

        // Skip ld.param for the bound input param (it's now in SMEM)
        if rtrimmed.contains("ld.param")
            && line
                .trim()
                .contains(&format!("[{}]", binding.b_input_param))
        {
            out.push_str(&format!(
                "    // FERRITE: skipped ld.param [{}] (now from SMEM)\n",
                binding.b_input_param
            ));
            continue;
        }

        // Rewrite B's ld.global on the bound input → ld.shared from handoff
        if rtrimmed.contains("ld.global") && is_addr_in_set(rtrimmed, &b_input_addr_regs_renamed) {
            emit_smem_load(
                &mut out,
                rtrimmed,
                smem_handoff,
                b_offset_reg_renamed.as_deref(),
            );
            continue;
        }

        // Redirect EXIT / ret
        if rtrimmed == "ret;" {
            out.push_str("    bra FUSED_EXIT;\n");
            continue;
        }
        if line.trim().starts_with("EXIT") && line.trim().ends_with(':') {
            out.push_str("PHASE_B_EXIT:\n");
            continue;
        }
        if rtrimmed.contains("bra EXIT")
            || (line.trim().contains("bra EXIT") && rtrimmed.contains("bra"))
        {
            out.push_str(&format!(
                "    {}\n",
                rtrimmed.replace("bra EXIT", "bra PHASE_B_EXIT")
            ));
            continue;
        }

        out.push_str(&format!("    {rtrimmed}\n"));
    }

    out.push_str("\nFUSED_EXIT:\n    ret;\n}\n");

    Ok(FusedKernel { ptx: out })
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

/// Find the register holding the byte offset for a given param's address.
/// E.g., for rms_norm's output: %rd6 = %rd1 + %rd3, where %rd1 = output base.
/// The byte offset register is %rd3 (the non-base operand of the add.u64).
fn find_byte_offset_register(
    body_lines: &[String],
    addr_regs: &[String],
    reg_to_param: &BTreeMap<String, String>,
    param_name: &str,
) -> Option<String> {
    // Find the add.u64 that produces the address register
    // The address register is used in st.global/ld.global
    // Find which addr_reg is the final address (appears in brackets of ld/st.global)
    let final_addr = addr_regs.last()?;

    // Find: add.u64 <final_addr>, <base>, <offset>
    for line in body_lines {
        let t = line.trim();
        if !t.starts_with("add.u64") {
            continue;
        }
        let parts: Vec<&str> = t
            .split([',', ' ', '\t'])
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() >= 4 {
            let dst = parts[1].trim_end_matches(',');
            if dst != final_addr {
                continue;
            }
            let src1 = parts[2].trim_end_matches(',');
            let src2 = parts[3].trim_end_matches(';');

            // The base is the one that maps to the param; the other is the offset
            if reg_to_param
                .get(src1)
                .map(|p| p == param_name)
                .unwrap_or(false)
            {
                return Some(src2.to_string());
            }
            if reg_to_param
                .get(src2)
                .map(|p| p == param_name)
                .unwrap_or(false)
            {
                return Some(src1.to_string());
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

/// Emit a st.shared sequence using the byte offset register.
fn emit_smem_store(out: &mut String, instruction: &str, smem_name: &str, offset_reg: Option<&str>) {
    // Extract value register: st.global.f32 [%rdX], %fY;
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0]; // st.global.f32
    let rest = if parts.len() > 1 { parts[1] } else { "" };

    // Get the value part (after ], )
    let val = if let Some(bracket_end) = rest.find(']') {
        rest[bracket_end + 1..]
            .trim()
            .trim_start_matches(',')
            .trim()
    } else {
        ""
    };

    let shared_op = op.replace("st.global", "st.shared");

    match offset_reg {
        Some(off_reg) => {
            out.push_str("    // FERRITE: st.global -> st.shared (handoff)\n");
            out.push_str(&format!(
                "    cvt.u32.u64 {FERRITE_REG_PREFIX}0, {off_reg};\n"
            ));
            out.push_str(&format!(
                "    mov.u32 {FERRITE_REG_PREFIX}1, {smem_name};\n"
            ));
            out.push_str(&format!(
                "    add.u32 {FERRITE_REG_PREFIX}2, {FERRITE_REG_PREFIX}1, {FERRITE_REG_PREFIX}0;\n"
            ));
            out.push_str(&format!("    {shared_op} [{FERRITE_REG_PREFIX}2], {val}\n"));
        }
        None => {
            // Fallback: can't determine offset register, emit with original address
            // (won't work but makes the issue visible)
            out.push_str("    // FERRITE WARNING: could not determine byte offset register\n");
            out.push_str(&format!("    {shared_op} {rest}\n"));
        }
    }
}

/// Emit a ld.shared sequence using the byte offset register.
fn emit_smem_load(out: &mut String, instruction: &str, smem_name: &str, offset_reg: Option<&str>) {
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0];
    let rest = if parts.len() > 1 { parts[1] } else { "" };

    // Get the dest register: ld.global.f32 %fX, [%rdY];
    let dest = if let Some(comma) = rest.find(',') {
        rest[..comma].trim()
    } else {
        ""
    };

    let shared_op = op.replace("ld.global", "ld.shared");

    match offset_reg {
        Some(off_reg) => {
            out.push_str("    // FERRITE: ld.global -> ld.shared (handoff)\n");
            out.push_str(&format!(
                "    cvt.u32.u64 {FERRITE_REG_PREFIX}0, {off_reg};\n"
            ));
            out.push_str(&format!(
                "    mov.u32 {FERRITE_REG_PREFIX}1, {smem_name};\n"
            ));
            out.push_str(&format!(
                "    add.u32 {FERRITE_REG_PREFIX}2, {FERRITE_REG_PREFIX}1, {FERRITE_REG_PREFIX}0;\n"
            ));
            out.push_str(&format!(
                "    {shared_op} {dest}, [{FERRITE_REG_PREFIX}2];\n"
            ));
        }
        None => {
            out.push_str("    // FERRITE WARNING: could not determine byte offset register\n");
            out.push_str(&format!("    {shared_op} {rest}\n"));
        }
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

    let mut result = Vec::new();
    // Collect all register types
    let mut all_types: BTreeMap<String, ()> = BTreeMap::new();
    for (ty, _) in a_regs {
        all_types.insert(ty.clone(), ());
    }
    for (ty, _) in b_regs {
        all_types.insert(ty.clone(), ());
    }

    for ty in all_types.keys() {
        let prefix = type_to_prefix
            .iter()
            .find(|(t, _)| *t == ty.as_str())
            .map(|(_, p)| *p);
        if let Some(prefix) = prefix {
            let a_count = a_map.get(ty.as_str()).copied().unwrap_or(0);
            let b_count = b_map.get(ty.as_str()).copied().unwrap_or(0);
            let offset = offsets.get(ty.as_str()).copied().unwrap_or(a_count);
            let total = offset + b_count;
            result.push((ty.clone(), prefix.to_string(), total));
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
    // Process longer prefixes first (%rd before %r)
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
        // For %r, make sure we're not matching %rd
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
    // Process longer prefixes first
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
    binding: &FuseBinding,
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

fn rename_smem_refs(line: &str, b_smem_regions: &[crate::parser::SmemRegion]) -> String {
    let mut result = line.to_string();
    for region in b_smem_regions {
        result = result.replace(&region.name, &format!("_b_{}", region.name));
    }
    result
}

fn offset_barrier_refs(line: &str, b_barriers: &[usize], new_start: usize) -> String {
    let mut result = line.to_string();
    let mut sorted: Vec<usize> = b_barriers.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    for (i, &old_id) in sorted.iter().enumerate() {
        result = result.replace(
            &format!("bar.sync {old_id}"),
            &format!("bar.sync {}", new_start + i),
        );
    }
    result
}
