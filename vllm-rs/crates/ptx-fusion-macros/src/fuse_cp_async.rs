//! Intercept cp.async loads in CUTLASS GEMMs for prologue injection.
//!
//! Replaces A-matrix cp.async.cg.shared.global loads with synchronous
//! SMEM-to-SMEM copies from a handoff buffer, enabling rms_norm → CUTLASS
//! GEMM fusion without an intermediate GMEM round-trip.

use std::collections::BTreeMap;

use crate::parser::PtxParser;

/// Classification of a cp.async instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CpAsyncClass {
    /// This cp.async loads the A matrix (redirect to SMEM)
    AMatrix,
    /// This cp.async loads the B matrix (keep as-is)
    BMatrix,
    /// Unknown — keep as-is
    Unknown,
}

/// Replace A-matrix cp.async loads with SMEM-to-SMEM copies.
///
/// The A-matrix data is already in a SMEM handoff buffer (written by rms_norm).
/// Each A-matrix `cp.async.cg.shared.global [smem_dst], [gmem_src], 16, mask`
/// is replaced with:
///   1. Compute handoff SMEM offset from GMEM address
///   2. ld.shared.v4.b32 from handoff buffer
///   3. st.shared.v4.b32 to CUTLASS's SMEM tile location
///
/// B-matrix cp.async loads are left unchanged.
pub fn redirect_a_matrix_loads(
    ptx: &str,
    a_param_substr: &str,
    smem_handoff: &str,
) -> Result<String, String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let proto = PtxParser::parse(ptx)?;
    let reg_to_param = PtxParser::trace_param_registers_pub(&lines, &proto.params);

    // Auto-detect A vs B: trace all cp.async GMEM sources to their param origins,
    // find the two distinct groups, and take the first-seen group as A (CUTLASS
    // loads A before B in the mainloop).
    let a_param_name = identify_a_matrix_param(&lines, &reg_to_param, a_param_substr)?;

    let a_addr_regs: Vec<String> = reg_to_param
        .iter()
        .filter(|(_, p)| **p == a_param_name)
        .map(|(r, _)| r.clone())
        .collect();

    if a_addr_regs.is_empty() {
        return Err(format!(
            "no address registers traced to A param '{a_param_name}'"
        ));
    }

    // Classify each cp.async as A-matrix or not
    let classifications = classify_cp_async_loads(&lines, &a_addr_regs);

    let a_count = classifications
        .values()
        .filter(|c| **c == CpAsyncClass::AMatrix)
        .count();
    if a_count == 0 {
        return Err("no cp.async loads classified as A-matrix".into());
    }

    // Rewrite: replace A-matrix cp.async with SMEM-to-SMEM copies
    let mut result = Vec::new();
    let mut scratch_needed = false;

    // Find the register that's loaded directly from the A param via ld.param.
    // This is the A pointer/base. We'll initialize %rd_cpa_base from it.
    let a_direct_reg = reg_to_param
        .iter()
        .find(|(_, p)| **p == a_param_name)
        .map(|(r, _)| r.clone())
        .ok_or("no register directly loaded from A param")?;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.contains("cp.async.cg.shared.global")
            && let Some(class) = classifications.get(&i)
            && *class == CpAsyncClass::AMatrix
        {
            scratch_needed = true;
            if let Some((smem_dst, gmem_src, mask)) = parse_cp_async(trimmed) {
                emit_smem_to_smem_copy(&mut result, &smem_dst, &gmem_src, &mask, smem_handoff);
                continue;
            }
        }

        result.push(line.to_string());
    }

    if !scratch_needed {
        return Err("no A-matrix cp.async loads were rewritten".into());
    }

    // Insert scratch register + SMEM declarations
    insert_cp_async_scratch(&mut result, smem_handoff, &a_direct_reg);

    Ok(result.join("\n"))
}

/// Identify the A-matrix param by analyzing cp.async GMEM source traces.
///
/// All cp.async sources trace back to one of two struct-param offsets (A and B).
/// The first group encountered in instruction order is A (CUTLASS loads A first).
/// If `hint` is provided, it's used as a substring match to disambiguate.
fn identify_a_matrix_param(
    lines: &[&str],
    reg_to_param: &BTreeMap<String, String>,
    hint: &str,
) -> Result<String, String> {
    // Collect the param that each cp.async GMEM source traces to, in instruction order
    let mut seen_params = Vec::new();
    for line in lines {
        let t = line.trim();
        if !t.contains("cp.async.cg.shared.global") {
            continue;
        }
        if let Some((_smem, gmem, _mask)) = parse_cp_async(t)
            && let Some(param) = reg_to_param.get(&gmem)
            && !seen_params.contains(param)
        {
            seen_params.push(param.clone());
        }
    }

    if seen_params.is_empty() {
        return Err("no cp.async GMEM sources could be traced to params".into());
    }

    // If hint is provided and matches one param, use it
    if !hint.is_empty()
        && let Some(matched) = seen_params.iter().find(|p| p.contains(hint))
    {
        return Ok(matched.clone());
    }

    // Otherwise: first param seen in cp.async order is A (CUTLASS convention)
    Ok(seen_params[0].clone())
}

/// Classify each cp.async instruction as A-matrix or B-matrix.
///
/// A cp.async is A-matrix if its GMEM source register traces to the A param.
fn classify_cp_async_loads(
    lines: &[&str],
    a_addr_regs: &[String],
) -> BTreeMap<usize, CpAsyncClass> {
    let mut classifications = BTreeMap::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.contains("cp.async.cg.shared.global") {
            continue;
        }

        if let Some((_smem_dst, gmem_src, _mask)) = parse_cp_async(trimmed) {
            // Check if the GMEM source register is in the A-matrix set
            if a_addr_regs.contains(&gmem_src) {
                classifications.insert(i, CpAsyncClass::AMatrix);
            } else {
                classifications.insert(i, CpAsyncClass::BMatrix);
            }
        } else {
            classifications.insert(i, CpAsyncClass::Unknown);
        }
    }

    classifications
}

/// Parse: cp.async.cg.shared.global.L2::128B [%r201], [%rd138], 16, %r819;
/// Returns (smem_dst, gmem_src, mask)
fn parse_cp_async(instr: &str) -> Option<(String, String, String)> {
    // Find the bracket pairs
    let mut brackets = Vec::new();
    let mut i = 0;
    let bytes = instr.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let start = i + 1;
            while i < bytes.len() && bytes[i] != b']' {
                i += 1;
            }
            brackets.push(instr[start..i].trim().to_string());
        }
        i += 1;
    }

    if brackets.len() < 2 {
        return None;
    }

    let smem_dst = brackets[0].clone();
    let gmem_src = brackets[1].clone();

    // Extract mask: last token before semicolon
    let parts: Vec<&str> = instr
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    let mask = parts.last()?.trim_end_matches(';').to_string();

    Some((smem_dst, gmem_src, mask))
}

/// Emit SMEM-to-SMEM copy replacing a cp.async instruction.
///
/// The GMEM address tells us which element we need. We compute the offset
/// into the handoff SMEM buffer and copy 16 bytes to the CUTLASS SMEM tile.
fn emit_smem_to_smem_copy(
    out: &mut Vec<String>,
    smem_dst: &str,
    gmem_src: &str,
    mask: &str,
    smem_handoff: &str,
) {
    out.push("\t// FERRITE: cp.async -> SMEM-to-SMEM (A-matrix from handoff)".to_string());
    // Check mask: if 0, skip (boundary case)
    out.push(format!("\tsetp.ne.u32 \t%p_cpa, {mask}, 0;"));
    // Compute offset in handoff buffer from GMEM address
    // handoff_addr = handoff_base + (gmem_addr - A_row_base)
    // We use %rd_cpa0 for the subtraction, %r_cpa0..5 for SMEM addr + data
    out.push(format!(
        "\t@%p_cpa sub.s64 \t%rd_cpa0, {gmem_src}, %rd_cpa_base;"
    ));
    out.push("\t@%p_cpa cvt.u32.u64 \t%r_cpa0, %rd_cpa0;".to_string());
    out.push(format!("\t@%p_cpa mov.u32 \t%r_cpa1, {smem_handoff};"));
    out.push("\t@%p_cpa add.u32 \t%r_cpa1, %r_cpa1, %r_cpa0;".to_string());
    // Load 16 bytes from handoff SMEM
    out.push(
        "\t@%p_cpa ld.shared.v4.b32 \t{%r_cpa2, %r_cpa3, %r_cpa4, %r_cpa5}, [%r_cpa1];".to_string(),
    );
    // Store to CUTLASS SMEM tile
    out.push(format!(
        "\t@%p_cpa st.shared.v4.b32 \t[{smem_dst}], {{%r_cpa2, %r_cpa3, %r_cpa4, %r_cpa5}};"
    ));
}

/// Insert scratch register + SMEM declarations for cp.async replacement.
fn insert_cp_async_scratch(lines: &mut Vec<String>, smem_handoff: &str, a_direct_reg: &str) {
    // Find first instruction after .reg/.shared declarations
    let mut insert_pos = None;
    let mut in_body = false;

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t == "{" || t.ends_with('{') {
            in_body = true;
        }
        if in_body
            && !t.starts_with(".reg")
            && !t.starts_with(".shared")
            && !t.starts_with(".local")
            && !t.starts_with("//")
            && !t.is_empty()
            && !t.starts_with("{")
        {
            insert_pos = Some(i);
            break;
        }
    }

    if let Some(pos) = insert_pos {
        let decls = vec![
            "\t// FERRITE: cp.async replacement scratch".to_string(),
            "\t.reg .pred \t%p_cpa;".to_string(),
            "\t.reg .b64 \t%rd_cpa0;".to_string(),
            "\t.reg .b64 \t%rd_cpa_base;".to_string(),
            "\t.reg .b32 \t%r_cpa0;".to_string(),
            "\t.reg .b32 \t%r_cpa1;".to_string(),
            "\t.reg .b32 \t%r_cpa2;".to_string(),
            "\t.reg .b32 \t%r_cpa3;".to_string(),
            "\t.reg .b32 \t%r_cpa4;".to_string(),
            "\t.reg .b32 \t%r_cpa5;".to_string(),
            format!("\t.shared .align 16 .b8 {smem_handoff}[65536];"),
            String::new(),
            "\t// FERRITE: initialize A-matrix base for SMEM offset computation".to_string(),
            format!("\tmov.u64 \t%rd_cpa_base, {a_direct_reg};"),
        ];
        for (j, decl) in decls.iter().enumerate() {
            lines.insert(pos + j, decl.clone());
        }
    }
}
