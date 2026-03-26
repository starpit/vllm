//! Chain a third phase onto an already-fused kernel via GMEM handoff.
//!
//! Phase C reads from the same GMEM buffer that Phase B wrote to.
//! No SMEM rewriting needed -- just param sharing, register offsetting,
//! and a barrier between Phase B and Phase C.

use crate::fuse_real::{
    compute_register_offsets, extract_body_lines, find_label_prefix, find_param_by_substring,
    offset_all_registers, offset_barriers,
};
use crate::parser::{KernelParam, PtxParser};

/// Append a third phase to an already-fused kernel, with GMEM handoff.
///
/// Phase C reads its input from the same GMEM buffer that the fused kernel's
/// Phase B wrote to. The bound params are merged (C's input = B's output).
pub fn append_phase_gmem(
    fused_ptx: &str,
    new_kernel_ptx: &str,
    shared_output_param: &str,
    new_input_param: &str,
    new_name: &str,
) -> Result<String, String> {
    let proto_fused = PtxParser::parse(fused_ptx)?;
    let proto_c = PtxParser::parse(new_kernel_ptx)?;

    // Find the actual param names
    let shared_param_name = find_param_by_substring(&proto_fused.params, shared_output_param)
        .ok_or_else(|| format!("no param matching '{shared_output_param}' in fused kernel"))?;
    let c_input_param_name = find_param_by_substring(&proto_c.params, new_input_param)
        .ok_or_else(|| format!("no param matching '{new_input_param}' in Phase C"))?;

    // Compute register offsets: Phase C registers start after all fused registers
    let reg_offsets = compute_register_offsets(&proto_fused.registers, &proto_c.registers);

    // Merged params: fused params + C params (minus the bound input, with renames for conflicts)
    let (merged_params, param_renames) =
        merge_params_chain(&proto_fused.params, &proto_c.params, &c_input_param_name);

    // Max barrier from fused kernel
    let max_barrier_fused = proto_fused.barriers.iter().max().copied().unwrap_or(1);
    let chain_barrier = max_barrier_fused + 1;
    let c_barrier_offset = chain_barrier + 1;

    // Extract Phase C body
    let body_c_lines = extract_body_lines(new_kernel_ptx)?;
    let c_label_prefix = find_label_prefix(&body_c_lines).unwrap_or("$L__BB0".to_string());
    let c_label_replacement = "$L__BB2".to_string();

    // Build the output PTX
    let mut out = String::new();

    // Version/target from fused PTX
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

    // Top-level .shared declarations from fused PTX (before entry)
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

    // Entry point with merged params
    out.push_str(&format!(".visible .entry {new_name}(\n"));
    for (i, p) in merged_params.iter().enumerate() {
        let comma = if i + 1 < merged_params.len() { "," } else { "" };
        out.push_str(&format!("\t.param {} {}{}\n", p.ptx_type, p.name, comma));
    }
    out.push_str(")\n{\n");

    // Register declarations: expand the fused kernel's existing declarations
    // to accommodate Phase C's additional registers.
    // extract_body_lines strips .reg lines, so we read them directly from the PTX.
    let fused_reg_lines = extract_reg_lines(fused_ptx);
    let fused_body = extract_body_lines(fused_ptx)?;
    emit_expanded_reg_decls(&mut out, &fused_reg_lines, &proto_c.registers, &reg_offsets);

    // Body-level .shared declarations from fused kernel
    for line in &fused_body {
        let t = line.trim();
        if t.starts_with(".shared") {
            out.push_str(&format!("\t{t}\n"));
        }
    }
    out.push('\n');

    // ── Phases A + B (from fused kernel body, skip .reg and .shared) ──
    for line in &fused_body {
        let t = line.trim();
        if t.starts_with(".shared") || t.starts_with(".reg") {
            continue;
        }
        if t == "ret;" {
            continue;
        }
        out.push_str(&format!("\t{t}\n"));
    }

    // ── Chain barrier ──
    out.push_str(&format!("\n\tbar.sync \t{chain_barrier};\n\n"));

    // ── Phase C ──
    out.push_str(&format!(
        "\t// ===== PHASE C: {} (registers offset) =====\n",
        proto_c.name
    ));

    for line in &body_c_lines {
        let trimmed = line.trim();

        if trimmed.starts_with(".shared") || trimmed.starts_with(".reg") {
            continue;
        }

        let mut renamed = offset_all_registers(trimmed, &reg_offsets);
        renamed = offset_barriers(&renamed, &proto_c.barriers, c_barrier_offset);
        renamed = renamed.replace(&c_label_prefix, &c_label_replacement);

        // Apply param renames (for collision avoidance)
        for (old_name, new_name) in &param_renames {
            renamed = renamed.replace(old_name.as_str(), new_name.as_str());
        }
        let rtrimmed = renamed.trim();

        // Redirect C's bound input param load to the shared output param
        if trimmed.contains("ld.param") && trimmed.contains(&c_input_param_name) {
            let redirected = rtrimmed.replace(&c_input_param_name, &shared_param_name);
            out.push_str("\t// FERRITE: Phase C reads from Phase B output (GMEM handoff)\n");
            out.push_str(&format!("\t{redirected}\n"));
            continue;
        }

        out.push_str(&format!("\t{rtrimmed}\n"));
    }

    out.push_str("}\n");

    Ok(out)
}

/// Merge params: keep all fused params, add C's params (minus the bound input).
/// If C's param names conflict with existing ones, prefix with "_c_" and record the rename.
fn merge_params_chain(
    fused_params: &[KernelParam],
    c_params: &[KernelParam],
    c_input_name: &str,
) -> (Vec<KernelParam>, Vec<(String, String)>) {
    let mut params: Vec<KernelParam> = fused_params.to_vec();
    let mut renames = Vec::new(); // (old_name, new_name)
    for cp in c_params {
        if cp.name == c_input_name {
            continue;
        }
        if params.iter().any(|p| p.name == cp.name) {
            // Name collision — rename with _c_ prefix
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

/// Emit expanded register declarations that cover both the fused kernel's
/// registers and Phase C's additional registers.
fn emit_expanded_reg_decls(
    out: &mut String,
    fused_reg_lines: &[String],
    c_regs: &[(String, usize)],
    offsets: &std::collections::BTreeMap<String, usize>,
) {
    // Collect existing .reg declarations from the fused body
    // Expand standard numbered registers to include Phase C, pass through special ones
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

        // Check if this is a standard numbered register: .reg .TYPE %PREFIX<COUNT>;
        // vs a named register: .reg .TYPE %name;
        // Check if this is a standard register (%r<N>, %rd<N>, %f<N>, %p<N>)
        // vs a named group (%r_fe<8>, %f_act<4>) or a single named register (%r_ptile)
        let is_standard = t.contains('<')
            && type_to_prefix.iter().any(|(_, prefix)| {
                // Match: .reg .TYPE \t%PREFIX<COUNT>;
                // The prefix must appear right before < with no extra chars
                if let Some(pos) = t.find(prefix) {
                    let after = &t[pos + prefix.len()..];
                    after.starts_with('<')
                } else {
                    false
                }
            });

        if is_standard {
            // Standard numbered register — expand if needed
            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() >= 3 {
                let ty = parts[1];
                if let Some(angle_start) = t.find('<')
                    && let Some(angle_end) = t.find('>')
                {
                    let current_count: usize = t[angle_start + 1..angle_end].parse().unwrap_or(0);

                    if let Some(c_count) = c_map.get(ty) {
                        let offset = offsets.get(ty).copied().unwrap_or(0);
                        let needed = offset + c_count;
                        let new_count = current_count.max(needed);
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
            // Named register group (%r_fe<8>, %f_act<4>) or single (%r_ptile) — pass through
            out.push_str(&format!("\t{t}\n"));
        }
    }

    // Also emit the SiLU scratch (.reg .f32 %f_act<4>, .reg .b32 %r_act<2>) if not present
    // These are injected by fuse_epilogue but not in the parsed register counts
    let has_f_act = fused_reg_lines.iter().any(|l| l.contains("%f_act"));
    let has_r_act = fused_reg_lines.iter().any(|l| l.contains("%r_act"));
    if has_f_act && !out.contains("%f_act") {
        out.push_str("\t.reg .f32 \t%f_act<4>;\n");
    }
    if has_r_act && !out.contains("%r_act") {
        out.push_str("\t.reg .b32 \t%r_act<2>;\n");
    }

    // Emit any Phase C register types not already present
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

/// Extract all .reg declaration lines from PTX (inside the entry body).
fn extract_reg_lines(ptx: &str) -> Vec<String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let mut in_body = false;
    let mut regs = Vec::new();
    for line in lines {
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
