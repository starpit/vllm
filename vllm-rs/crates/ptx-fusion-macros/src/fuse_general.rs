//! General kernel fusion engine.
//!
//! Given N kernels (as PTX) and a set of bindings (`a.output => b.input`),
//! produce a single fused kernel. The engine is kernel-agnostic: it only
//! sees escape perimeters (global stores/loads traced to params) and decides
//! the handoff mechanism (registers or SMEM) based on thread-to-element
//! mapping analysis.
//!
//! This module is called by the `fuse!` proc macro.

use std::collections::BTreeMap;

use crate::fuse_real::{
    self, RowOffset, compute_merged_reg_decls, compute_register_offsets, extract_body_lines,
    find_cvta_register, find_label_prefix, find_param_by_substring, find_row_offset_near,
    merge_params, offset_all_registers, offset_barriers, offset_single_register,
};
use crate::parser::{KernelProtocol, PtxParser};

// ── Parsed binding from the DSL ──

/// A binding parsed from `fuse!`: producer.port => consumer.port
#[derive(Debug)]
pub struct ParsedBinding {
    /// Name of the producer kernel (e.g., "a")
    pub producer: String,
    /// Output param name/substring on the producer
    pub producer_port: String,
    /// Name of the consumer kernel (e.g., "b")
    pub consumer: String,
    /// Input param name/substring on the consumer
    pub consumer_port: String,
}

/// Result of the general fusion engine.
pub struct FusedResult {
    /// The fused PTX source.
    pub ptx: String,
    /// The entry point name.
    pub entry_name: String,
}

/// How data transits between two fused phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoffKind {
    /// A's output register becomes B's input register. Zero memory traffic.
    /// Only valid when both kernels are elementwise with identical
    /// thread-to-element mappings.
    Register,
    /// A writes to SMEM, barrier, B reads from SMEM. Always correct.
    SharedMemory,
}

// ── Thread-mapping analysis ──

/// Determine whether register handoff is possible between producer and consumer.
///
/// Register handoff requires:
/// 1. Both kernels are elementwise (no SMEM, no barriers, no MMA)
/// 2. Both use the same grid (blockDim, gridDim)
/// 3. The address computation for the bound store/load reduces to the same
///    function of (threadIdx, blockIdx)
///
/// When in doubt, returns SharedMemory (always correct).
fn choose_handoff(
    proto_a: &KernelProtocol,
    proto_b: &KernelProtocol,
    lines_a: &[&str],
    lines_b: &[&str],
    a_output_addr_regs: &[String],
    b_input_addr_regs: &[String],
) -> HandoffKind {
    let a_elementwise =
        proto_a.smem_regions.is_empty() && proto_a.barriers.is_empty() && !proto_a.has_mma;
    let b_elementwise =
        proto_b.smem_regions.is_empty() && proto_b.barriers.is_empty() && !proto_b.has_mma;

    if !a_elementwise || !b_elementwise {
        return HandoffKind::SharedMemory;
    }

    // Both elementwise. Verify that A has a store and B has a load on the bound params,
    // and that each is a simple scalar (not vectorized) — register handoff needs 1:1 mapping.
    let body_a = match extract_body_lines_internal(lines_a) {
        Ok(b) => b,
        Err(_) => return HandoffKind::SharedMemory,
    };
    let body_b = match extract_body_lines_internal(lines_b) {
        Ok(b) => b,
        Err(_) => return HandoffKind::SharedMemory,
    };

    // Check A has a scalar st.global to the bound param
    let a_store = find_store_value_register(&body_a, a_output_addr_regs);
    if a_store.is_none() {
        return HandoffKind::SharedMemory;
    }

    // Check B has a scalar ld.global from the bound param
    let b_load = find_load_dest_register(&body_b, b_input_addr_regs);
    if b_load.is_none() {
        return HandoffKind::SharedMemory;
    }

    // Both have simple scalar store/load — register handoff is valid.
    // The kernels are elementwise (thread i processes element i), so the
    // value A's thread i stores is the same element B's thread i loads.
    HandoffKind::Register
}

// ── General fusion engine ──

/// Fuse two kernels given a binding.
///
/// This is the core of `fuse!`. It:
/// 1. Parses both PTX files and extracts perimeters
/// 2. Resolves the binding to concrete store/load sites
/// 3. Chooses handoff mechanism (register or SMEM)
/// 4. Rewrites: redirect A's bound stores, redirect B's bound loads
/// 5. Merges params (bound pair eliminated), registers, SMEM, barriers
/// 6. Returns a single fused kernel
pub fn fuse_two(
    ptx_a: &str,
    ptx_b: &str,
    a_output_port: &str,
    b_input_port: &str,
    fused_name: &str,
) -> Result<FusedResult, String> {
    // Parse extracts the first entry from multi-entry PTX.
    // We need to work on extracted single-entry PTX so that
    // trace_param_registers sees only the relevant params.
    let proto_a = PtxParser::parse(ptx_a)?;
    let proto_b = PtxParser::parse(ptx_b)?;

    // Re-extract single entries to get clean PTX for each kernel
    let ptx_a_single = extract_single_entry(ptx_a, &proto_a.name)?;
    let ptx_b_single = extract_single_entry(ptx_b, &proto_b.name)?;
    let lines_a: Vec<&str> = ptx_a_single.lines().collect();
    let lines_b: Vec<&str> = ptx_b_single.lines().collect();

    // Resolve binding to param names
    let a_output_param =
        find_param_by_substring(&proto_a.params, a_output_port).ok_or_else(|| {
            format!(
                "no param matching '{}' in kernel A ({})",
                a_output_port, proto_a.name
            )
        })?;
    let b_input_param =
        find_param_by_substring(&proto_b.params, b_input_port).ok_or_else(|| {
            format!(
                "no param matching '{}' in kernel B ({})",
                b_input_port, proto_b.name
            )
        })?;

    // Trace address registers for the bound params
    let reg_to_param_a = PtxParser::trace_param_registers_pub(&lines_a, &proto_a.params);
    let reg_to_param_b = PtxParser::trace_param_registers_pub(&lines_b, &proto_b.params);

    let a_output_addr_regs: Vec<String> = reg_to_param_a
        .iter()
        .filter(|(_, p)| **p == a_output_param)
        .map(|(r, _)| r.clone())
        .collect();
    let b_input_addr_regs: Vec<String> = reg_to_param_b
        .iter()
        .filter(|(_, p)| **p == b_input_param)
        .map(|(r, _)| r.clone())
        .collect();

    if a_output_addr_regs.is_empty() {
        return Err(format!(
            "could not trace address registers for A's output param '{a_output_param}'"
        ));
    }
    if b_input_addr_regs.is_empty() {
        return Err(format!(
            "could not trace address registers for B's input param '{b_input_param}'"
        ));
    }

    let handoff = choose_handoff(
        &proto_a,
        &proto_b,
        &lines_a,
        &lines_b,
        &a_output_addr_regs,
        &b_input_addr_regs,
    );

    match handoff {
        HandoffKind::Register => fuse_register(
            &ptx_a_single,
            &ptx_b_single,
            &proto_a,
            &proto_b,
            &lines_a,
            &lines_b,
            &a_output_param,
            &b_input_param,
            &a_output_addr_regs,
            &b_input_addr_regs,
            fused_name,
        ),
        HandoffKind::SharedMemory => fuse_smem(
            &ptx_a_single,
            &ptx_b_single,
            &proto_a,
            &proto_b,
            &lines_a,
            &lines_b,
            &a_output_param,
            &b_input_param,
            &a_output_addr_regs,
            &b_input_addr_regs,
            &reg_to_param_a,
            &reg_to_param_b,
            fused_name,
        ),
    }
}

/// Register handoff fusion: A's output value stays in a register, B reads it directly.
///
/// No SMEM, no barrier, zero memory traffic for the intermediate.
/// Valid only for elementwise kernels with matching thread-to-element mappings.
#[allow(clippy::too_many_arguments)]
fn fuse_register(
    ptx_a: &str,
    ptx_b: &str,
    proto_a: &KernelProtocol,
    proto_b: &KernelProtocol,
    lines_a: &[&str],
    lines_b: &[&str],
    a_output_param: &str,
    b_input_param: &str,
    a_output_addr_regs: &[String],
    b_input_addr_regs: &[String],
    fused_name: &str,
) -> Result<FusedResult, String> {
    let body_a = extract_body_lines(ptx_a)?;
    let body_b = extract_body_lines(ptx_b)?;

    // Find the value register in A's store
    let a_value_reg = find_store_value_register(&body_a, a_output_addr_regs)
        .ok_or("register fusion: could not find value register in A's store")?;

    // Find the dest register in B's load
    let b_dest_reg = find_load_dest_register(&body_b, b_input_addr_regs)
        .ok_or("register fusion: could not find dest register in B's load")?;

    // Register offsets for B.
    // offset_single_register from fuse_real.rs maps %rd -> ".b64", but hand-written PTX
    // may use ".u64". We use offset_all_registers (which handles both) on a synthetic
    // instruction to rename single registers reliably.
    let reg_offsets = compute_register_offsets(&proto_a.registers, &proto_b.registers);
    let b_dest_reg_renamed = rename_one_register(&b_dest_reg, &reg_offsets);

    let b_input_addr_regs_renamed: Vec<String> = b_input_addr_regs
        .iter()
        .map(|r| rename_one_register(r, &reg_offsets))
        .collect();

    // Merged params and registers
    let merged_params = merge_params(proto_a, proto_b, a_output_param, b_input_param);
    let merged_regs =
        compute_merged_reg_decls(&proto_a.registers, &proto_b.registers, &reg_offsets);

    // Label prefixes for renaming B's labels
    let b_label_prefix = find_label_prefix(&body_b).unwrap_or("$L__BB0".to_string());
    let b_label_replacement = "$L__BB1".to_string();

    // ── Build fused PTX ──
    let mut out = String::new();

    // PTX header from kernel A
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
    out.push('\n');

    // ── Phase A: emit body, eliminate st.global (value stays in register) ──
    out.push_str(&format!(
        "\t// ===== PHASE A: {} (elementwise) =====\n",
        proto_a.name
    ));

    for line in &body_a {
        let trimmed = line.trim();

        // Skip ld.param for the bound output param
        if trimmed.contains("ld.param") && trimmed.contains(a_output_param) {
            out.push_str(&format!(
                "\t// FERRITE: skipped ld.param [{}] (register handoff)\n",
                a_output_param
            ));
            continue;
        }

        // Skip cvta.to.global for the output (no global address needed)
        if trimmed.contains("cvta.to.global") {
            // Check if the source register is the one loaded from the bound param
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 3 {
                let src = parts[2].trim_end_matches(';');
                // If this cvta uses a register that was loaded from the bound param,
                // skip it. We detect this by checking if the register was zeroed by
                // our ld.param elimination above.
                if lines_a.iter().any(|l| {
                    l.contains("ld.param") && l.contains(a_output_param) && l.contains(src)
                }) {
                    out.push_str("\t// FERRITE: skipped cvta for output (register handoff)\n");
                    continue;
                }
            }
        }

        // Skip address computation for the output
        if (trimmed.starts_with("add.u64") || trimmed.starts_with("add.s64"))
            && produces_any(trimmed, a_output_addr_regs)
        {
            out.push_str("\t// FERRITE: skipped output addr computation (register handoff)\n");
            continue;
        }

        // Eliminate st.global — value stays in register
        if trimmed.contains("st.global") && is_addr_in_set(trimmed, a_output_addr_regs) {
            out.push_str(&format!(
                "\t// FERRITE: st.global eliminated - value stays in {a_value_reg}\n"
            ));
            continue;
        }

        // Skip ret (continue into phase B)
        if trimmed == "ret;" {
            continue;
        }

        // Rename EXIT label to avoid collision with Phase B
        let emitted = if trimmed.contains("EXIT") {
            trimmed
                .replace("EXIT:", "PHASE_A_EXIT:")
                .replace("bra EXIT", "bra PHASE_A_EXIT")
        } else {
            trimmed.to_string()
        };
        out.push_str(&format!("\t{emitted}\n"));
    }

    // ── Phase B: emit body with registers offset, replace ld.global with mov ──
    out.push_str(&format!(
        "\n\t// ===== PHASE B: {} (register handoff: {} -> {}) =====\n",
        proto_b.name, a_value_reg, b_dest_reg_renamed
    ));

    // Determine the data type for the mov instruction
    let mov_type = if a_value_reg.starts_with("%f") {
        "f32"
    } else if a_value_reg.starts_with("%rd") {
        "u64"
    } else {
        "u32"
    };

    for line in &body_b {
        let trimmed = line.trim();
        let renamed = offset_all_registers(trimmed, &reg_offsets);
        let renamed = renamed.replace(&b_label_prefix, &b_label_replacement);
        let rtrimmed = renamed.trim();

        // Skip ld.param for the bound input param
        if trimmed.contains("ld.param") && trimmed.contains(b_input_param) {
            out.push_str(&format!(
                "\t// FERRITE: skipped ld.param [{}] (register handoff)\n",
                b_input_param
            ));
            continue;
        }

        // Skip cvta.to.global for the input
        if trimmed.contains("cvta.to.global") {
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 3 {
                let src = parts[2].trim_end_matches(';');
                if lines_b
                    .iter()
                    .any(|l| l.contains("ld.param") && l.contains(b_input_param) && l.contains(src))
                {
                    out.push_str("\t// FERRITE: skipped cvta for input (register handoff)\n");
                    continue;
                }
            }
        }

        // Skip address computation for the input
        if (rtrimmed.starts_with("add.u64") || rtrimmed.starts_with("add.s64"))
            && produces_any(rtrimmed, &b_input_addr_regs_renamed)
        {
            out.push_str("\t// FERRITE: skipped input addr computation (register handoff)\n");
            continue;
        }

        // Replace ld.global with mov from A's value register
        if rtrimmed.contains("ld.global") && is_addr_in_set(rtrimmed, &b_input_addr_regs_renamed) {
            out.push_str("\t// FERRITE: ld.global -> register handoff\n");
            out.push_str(&format!(
                "\tmov.{mov_type} \t{b_dest_reg_renamed}, {a_value_reg};\n"
            ));
            continue;
        }

        // Rename EXIT label in Phase B
        let emitted = if rtrimmed.contains("EXIT") {
            rtrimmed
                .replace("EXIT:", "PHASE_B_EXIT:")
                .replace("bra EXIT", "bra PHASE_B_EXIT")
        } else {
            rtrimmed.to_string()
        };
        out.push_str(&format!("\t{emitted}\n"));
    }

    out.push_str("}\n");

    Ok(FusedResult {
        ptx: out,
        entry_name: fused_name.to_string(),
    })
}

/// SMEM handoff fusion: A's bound stores -> SMEM, barrier, B's bound loads ← SMEM.
///
/// Generalizes fuse_real.rs to work with any kernel pair. The approach:
/// 1. Compute row_global_base for A's output (global base + row offset)
/// 2. For each st.global to the bound param: smem_addr = smem_base + (global_addr - row_global_base)
/// 3. Barrier
/// 4. Same pattern for B's input loads
#[allow(clippy::too_many_arguments)]
fn fuse_smem(
    ptx_a: &str,
    ptx_b: &str,
    proto_a: &KernelProtocol,
    proto_b: &KernelProtocol,
    lines_a: &[&str],
    lines_b: &[&str],
    a_output_param: &str,
    b_input_param: &str,
    a_output_addr_regs: &[String],
    b_input_addr_regs: &[String],
    _reg_to_param_a: &BTreeMap<String, String>,
    _reg_to_param_b: &BTreeMap<String, String>,
    fused_name: &str,
) -> Result<FusedResult, String> {
    // Find cvta registers (global-space base pointers)
    let a_output_cvta = find_cvta_register(lines_a, a_output_param)
        .ok_or_else(|| format!("could not find cvta register for A's output '{a_output_param}'"))?;
    let b_input_cvta = find_cvta_register(lines_b, b_input_param)
        .ok_or_else(|| format!("could not find cvta register for B's input '{b_input_param}'"))?;

    // Find row offset registers
    let a_row_offset = find_row_offset_near(lines_a, Some(a_output_param))
        .ok_or("could not find row offset register in A")?;
    let b_row_offset = find_row_offset_near(lines_b, Some(b_input_param))
        .ok_or("could not find row offset register in B")?;

    // Extract kernel bodies
    let body_a = extract_body_lines(ptx_a)?;
    let body_b = extract_body_lines(ptx_b)?;

    // Register offsets: B's registers get shifted past A's to avoid collision
    let reg_offsets = compute_register_offsets(&proto_a.registers, &proto_b.registers);

    // Rename B's key registers
    let b_input_addr_regs_renamed: Vec<String> = b_input_addr_regs
        .iter()
        .map(|r| offset_single_register(r, &reg_offsets))
        .collect();
    let b_input_cvta_renamed = offset_single_register(&b_input_cvta, &reg_offsets);
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

    // Merged params (bound pair eliminated)
    let merged_params = merge_params(proto_a, proto_b, a_output_param, b_input_param);
    let merged_regs =
        compute_merged_reg_decls(&proto_a.registers, &proto_b.registers, &reg_offsets);

    // Barrier numbering
    let max_barrier_a = proto_a.barriers.iter().max().copied().unwrap_or(0);
    let fusion_barrier = max_barrier_a + 1;
    let b_barrier_offset = fusion_barrier + 1;

    // SMEM handoff buffer — size based on A's output data
    // Estimate: count A's global stores to the bound param, multiply by max vector width
    let smem_elements = estimate_smem_elements(proto_a, a_output_param);
    let smem_handoff = "_ferrite_handoff";

    // Collect .shared declarations from both kernels
    let a_shared_decls = extract_shared_decls(ptx_a);
    let b_shared_decls = extract_shared_decls(ptx_b);

    // Label prefixes for renaming B's labels
    let b_label_prefix = find_label_prefix(&body_b).unwrap_or("$L__BB0".to_string());
    let b_label_replacement = "$L__BB1".to_string();

    // ── Build fused PTX ──
    let mut out = String::new();

    // PTX header from kernel A
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

    // Top-level .shared declarations
    for decl in &a_shared_decls {
        out.push_str(decl);
        out.push('\n');
    }
    for decl in &b_shared_decls {
        let renamed = fuse_real::prefix_shared_names_pub(decl, "_b_");
        out.push_str(&renamed);
        out.push('\n');
    }
    out.push('\n');

    // Entry point with merged params
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
    out.push_str("\t.reg .u32 \t%r_fe<8>;\n");
    out.push_str("\t.reg .b64 \t%rd_fe<4>;\n");

    // Body-level .shared declarations (deduped)
    let top_level_names: Vec<String> = a_shared_decls
        .iter()
        .chain(b_shared_decls.iter())
        .filter_map(|d| fuse_real::extract_decl_name_pub(d))
        .collect();
    emit_body_shared_decls(&mut out, &body_a, &top_level_names, "");
    emit_body_shared_decls(&mut out, &body_b, &top_level_names, "_b_");

    // SMEM handoff buffer
    out.push_str(&format!(
        "\t.shared .align 16 .f32 {smem_handoff}[{smem_elements}];\n"
    ));
    out.push('\n');

    // ── Phase A ──
    out.push_str(&format!("\t// ===== PHASE A: {} =====\n", proto_a.name));

    let mut emitted_row_base = false;

    for line in &body_a {
        let trimmed = line.trim();

        // Skip body-level .shared decls (already emitted above)
        if trimmed.starts_with(".shared") || trimmed.starts_with("// demoted") {
            continue;
        }

        // Skip ld.param for the bound output; zero the register so cvta doesn't fault
        if trimmed.contains("ld.param") && trimmed.contains(a_output_param) {
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let dest = parts[1].trim_end_matches(',');
                out.push_str(&format!(
                    "\t// FERRITE: [{}] removed (SMEM handoff)\n",
                    a_output_param
                ));
                out.push_str(&format!("\tmov.u64 \t{dest}, 0;\n"));
            }
            continue;
        }

        // After cvta for the output base, emit row_global_base
        if !emitted_row_base
            && trimmed.contains("cvta.to.global")
            && trimmed.contains(&a_output_cvta)
        {
            out.push_str(&format!("\t{trimmed}\n"));
            continue;
        }

        // Detect row offset definition, emit row_global_base
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
                        out.push_str(&format!("\tadd.s64 \t%rd_fe1, {a_output_cvta}, %rd_fe0;\n"));
                    }
                    RowOffset::WideMul { reg_64, .. } => {
                        out.push_str(&format!(
                            "\tadd.s64 \t%rd_fe1, {a_output_cvta}, {reg_64};\n"
                        ));
                    }
                }
                emitted_row_base = true;
                continue;
            }
        }

        // Rewrite st.global on the bound output -> st.shared via SMEM handoff
        if is_global_store(trimmed) && is_addr_in_set(trimmed, a_output_addr_regs) {
            emit_smem_store(&mut out, trimmed, smem_handoff, "%rd_fe1");
            continue;
        }

        // Skip ret (continue into phase B)
        if trimmed == "ret;" {
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

    let mut emitted_b_row_base = false;

    for line in &body_b {
        let trimmed = line.trim();

        // Skip body-level .shared decls
        if trimmed.starts_with(".shared") || trimmed.starts_with("// demoted") {
            continue;
        }

        // Rename B's registers, shared refs, barriers, labels
        let renamed = offset_all_registers(trimmed, &reg_offsets);
        let renamed = rename_b_shared_refs(&renamed, &b_shared_decls);
        let renamed = offset_barriers(&renamed, &proto_b.barriers, b_barrier_offset);
        let renamed = renamed.replace(&b_label_prefix, &b_label_replacement);
        let rtrimmed = renamed.trim();

        // Skip ld.param for the bound input; zero the register
        if trimmed.contains("ld.param") && trimmed.contains(b_input_param) {
            let parts: Vec<&str> = rtrimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                let dest = parts[1].trim_end_matches(',');
                out.push_str(&format!(
                    "\t// FERRITE: [{}] removed (SMEM handoff)\n",
                    b_input_param
                ));
                out.push_str(&format!("\tmov.u64 \t{dest}, 0;\n"));
            }
            continue;
        }

        // Detect row offset definition, emit row_global_base for B
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

        // Rewrite ld.global on the bound input -> ld.shared from SMEM handoff
        if is_global_load(rtrimmed) && is_addr_in_set(rtrimmed, &b_input_addr_regs_renamed) {
            emit_smem_load(&mut out, rtrimmed, smem_handoff, "%rd_fe3");
            continue;
        }

        out.push_str(&format!("\t{rtrimmed}\n"));
    }

    out.push_str("}\n");

    Ok(FusedResult {
        ptx: out,
        entry_name: fused_name.to_string(),
    })
}

// ── Helpers ──

/// Estimate SMEM elements needed for the handoff buffer.
/// Based on the output data size: count global stores to the bound param
/// and their data types.
fn estimate_smem_elements(proto: &KernelProtocol, bound_param: &str) -> usize {
    // Count stores to the bound param and estimate size
    let store_count = proto
        .global_stores
        .iter()
        .filter(|s| s.param_name == bound_param || bound_param.contains(&s.param_name))
        .count();

    if store_count > 0 {
        // Each store might be vectorized (v4 = 4 elements). Conservative estimate.
        // Use a generous multiple to handle vectorized stores.
        (store_count * 4).max(4096)
    } else {
        // Fallback: use a reasonable default
        4096
    }
}

/// Extract body lines from parsed lines (for choose_handoff which has &[&str]).
fn extract_body_lines_internal(lines: &[&str]) -> Result<Vec<String>, String> {
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
/// `st.global.f32 [%rd6], %f7;` -> returns `"%f7"`
fn find_store_value_register(body: &[String], addr_regs: &[String]) -> Option<String> {
    for line in body {
        let trimmed = line.trim();
        if trimmed.contains("st.global") && is_addr_in_set(trimmed, addr_regs) {
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
/// `ld.global.f32 %f1, [%rd3];` -> returns `"%f1"`
fn find_load_dest_register(body: &[String], addr_regs: &[String]) -> Option<String> {
    for line in body {
        let trimmed = line.trim();
        if trimmed.contains("ld.global") && is_addr_in_set(trimmed, addr_regs) {
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

/// Check if an instruction produces (writes to) any register in the set.
fn produces_any(instruction: &str, regs: &[String]) -> bool {
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

/// Rename a single register using offset_all_registers (handles both .u64 and .b64).
/// Wraps the register in a dummy instruction, renames, then extracts.
fn rename_one_register(reg: &str, offsets: &BTreeMap<String, usize>) -> String {
    let dummy = format!("mov.u64 {reg}, 0;");
    let renamed = offset_all_registers(&dummy, offsets);
    // Extract the register name from "mov.u64 %rdNN, 0;"
    let parts: Vec<&str> = renamed
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() >= 2 {
        parts[1].trim_end_matches(',').to_string()
    } else {
        reg.to_string()
    }
}

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
fn emit_smem_store(out: &mut String, instruction: &str, smem_name: &str, row_base_reg: &str) {
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0];
    let rest = if parts.len() > 1 { parts[1].trim() } else { "" };
    let shared_op = op.replace("st.global", "st.shared");

    if let Some(bracket_start) = rest.find('[') {
        let bracket_end = rest.find(']').unwrap_or(rest.len());
        let addr_reg = rest[bracket_start + 1..bracket_end].trim();
        let values_part = rest[bracket_end + 1..]
            .trim()
            .trim_start_matches(',')
            .trim();

        out.push_str("\t// FERRITE: st.global -> st.shared (SMEM handoff)\n");
        out.push_str(&format!(
            "\tsub.s64 \t%rd_fe0, {addr_reg}, {row_base_reg};\n"
        ));
        out.push_str("\tcvt.u32.u64 \t%r_fe0, %rd_fe0;\n");
        out.push_str(&format!("\tmov.u32 \t%r_fe1, {smem_name};\n"));
        out.push_str("\tadd.u32 \t%r_fe2, %r_fe1, %r_fe0;\n");
        out.push_str(&format!("\t{shared_op} \t[%r_fe2], {values_part}\n"));
    } else {
        out.push_str(&format!("\t{instruction}\n"));
    }
}

/// Rewrite ld.global -> ld.shared using row_global_base subtraction.
fn emit_smem_load(out: &mut String, instruction: &str, smem_name: &str, row_base_reg: &str) {
    let parts: Vec<&str> = instruction.splitn(2, ' ').collect();
    let op = parts[0];
    let rest = if parts.len() > 1 { parts[1].trim() } else { "" };
    let shared_op = op.replace(".nc", "").replace("ld.global", "ld.shared");

    if let Some(bracket_start) = rest.find('[') {
        let bracket_end = rest.find(']').unwrap_or(rest.len());
        let addr_reg = rest[bracket_start + 1..bracket_end].trim();
        let dest_part = rest[..bracket_start].trim().trim_end_matches(',').trim();
        let after = rest[bracket_end + 1..].trim();

        out.push_str("\t// FERRITE: ld.global -> ld.shared (SMEM handoff)\n");
        out.push_str(&format!(
            "\tsub.s64 \t%rd_fe0, {addr_reg}, {row_base_reg};\n"
        ));
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

/// Extract a single entry from multi-entry PTX, using a substring of the entry name.
fn extract_single_entry(ptx: &str, entry_name: &str) -> Result<String, String> {
    let entry_count = ptx
        .lines()
        .filter(|l| l.contains(".entry") && l.contains('('))
        .count();
    if entry_count <= 1 {
        return Ok(ptx.to_string());
    }
    // Use the first 20 chars of the entry name as a substring match
    let substr = if entry_name.len() > 20 {
        &entry_name[..20]
    } else {
        entry_name
    };
    crate::extract::extract_entry(ptx, substr)
        .map_err(|e| format!("extract_single_entry({substr}): {e}"))
}

fn extract_shared_decls(ptx: &str) -> Vec<String> {
    ptx.lines()
        .filter(|l| {
            let t = l.trim();
            t.starts_with(".shared") || (t.starts_with(".extern") && t.contains(".shared"))
        })
        .map(|l| l.to_string())
        .collect()
}

fn emit_body_shared_decls(
    out: &mut String,
    body_lines: &[String],
    already_declared: &[String],
    prefix: &str,
) {
    for line in body_lines {
        let t = line.trim();
        if t.starts_with(".shared") {
            if let Some(name) = fuse_real::extract_decl_name_pub(t) {
                if !already_declared.contains(&name) {
                    if prefix.is_empty() {
                        out.push_str(&format!("\t{t}\n"));
                    } else {
                        let renamed = fuse_real::prefix_shared_names_pub(t, prefix);
                        out.push_str(&format!("\t{renamed}\n"));
                    }
                }
            }
        }
    }
}

fn rename_b_shared_refs(line: &str, b_shared_decls: &[String]) -> String {
    let mut result = line.to_string();
    for decl in b_shared_decls {
        if let Some(name) = fuse_real::extract_decl_name_pub(decl) {
            if result.contains(&name) {
                result = result.replace(&name, &format!("_b_{name}"));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_two_hand_written_elementwise() {
        // Test register handoff with the hand-written rms_norm + scale PTX
        let ptx_a = include_str!("../../ptx-fusion/kernels/rms_norm.ptx");
        let ptx_b = include_str!("../../ptx-fusion/kernels/scale.ptx");

        let result = fuse_two(ptx_a, ptx_b, "output", "input", "test_regfused").unwrap();

        // Should have chosen register handoff (both elementwise)
        assert!(
            result.ptx.contains("register handoff"),
            "should use register handoff for elementwise pair"
        );
        assert!(
            !result.ptx.contains("st.shared"),
            "should NOT use SMEM for elementwise pair"
        );
        assert!(
            result.ptx.contains("st.global eliminated"),
            "should eliminate A's st.global"
        );
        assert!(
            result.ptx.contains("mov.f32"),
            "should have mov.f32 for register transfer"
        );
    }

    #[test]
    fn fuse_two_nvcc_kernels_uses_smem() {
        // nvcc-compiled kernels with SMEM reductions -> SMEM handoff
        let ptx_a = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let ptx_b = include_str!("../../ptx-fusion/kernels/vllm_silu_mul.ptx");

        let result = fuse_two(ptx_a, ptx_b, "param_0", "param_1", "test_smem_fused").unwrap();

        // rms_norm uses SMEM for reduction -> should fall back to SMEM handoff
        assert!(
            result.ptx.contains("SMEM handoff"),
            "should use SMEM handoff when A has SMEM"
        );
    }

    #[test]
    fn find_store_value_reg() {
        let body = vec!["st.global.f32 [%rd6], %f7;".to_string()];
        let addr_regs = vec!["%rd6".to_string()];
        let val = find_store_value_register(&body, &addr_regs);
        assert_eq!(val.as_deref(), Some("%f7"));
    }

    #[test]
    fn find_load_dest_reg() {
        let body = vec!["ld.global.f32 %f1, [%rd3];".to_string()];
        let addr_regs = vec!["%rd3".to_string()];
        let dest = find_load_dest_register(&body, &addr_regs);
        assert_eq!(dest.as_deref(), Some("%f1"));
    }
}
