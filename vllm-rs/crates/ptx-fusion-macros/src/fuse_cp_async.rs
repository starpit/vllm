//! Intercept cp.async loads in CUTLASS GEMMs for prologue injection.
//!
//! The escape perimeter of a CUTLASS GEMM includes cp.async.cg.shared.global
//! instructions that copy matrix tiles from GMEM to SMEM. For prologue fusion,
//! the A-matrix data is already in SMEM (written by the prologue phase to the
//! same addresses). The cp.async instructions for A are simply deleted.

use std::collections::BTreeMap;

use crate::parser::PtxParser;

/// Classification of a cp.async instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CpAsyncClass {
    AMatrix,
    BMatrix,
    Unknown,
}

/// Delete A-matrix cp.async loads from a CUTLASS GEMM.
///
/// The prologue phase writes A-matrix data directly into CUTLASS's SMEM tile
/// locations. The cp.async instructions that would have loaded A from GMEM
/// are deleted — the data is already there.
///
/// B-matrix cp.async loads are preserved unchanged.
///
/// Returns the modified PTX and metadata about what was deleted.
pub fn delete_a_matrix_loads(ptx: &str, a_param_hint: &str) -> Result<DeleteResult, String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let proto = PtxParser::parse(ptx)?;
    let reg_to_param = PtxParser::trace_param_registers_pub(&lines, &proto.params);

    // Auto-detect A vs B from cp.async trace
    let a_param_name = identify_a_matrix_param(&lines, &reg_to_param, a_param_hint)?;

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

    let classifications = classify_cp_async_loads(&lines, &a_addr_regs);
    let a_count = classifications
        .values()
        .filter(|c| **c == CpAsyncClass::AMatrix)
        .count();
    if a_count == 0 {
        return Err("no cp.async loads classified as A-matrix".into());
    }

    // Collect metadata about deleted loads (SMEM destinations for prologue)
    let mut deleted_loads = Vec::new();

    // Delete A-matrix cp.async, keep everything else
    let mut result = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.contains("cp.async.cg.shared.global")
            && classifications.get(&i) == Some(&CpAsyncClass::AMatrix)
            && let Some((smem_dst, gmem_src, mask)) = parse_cp_async(trimmed)
        {
            result.push(
                "\t// FERRITE: deleted cp.async for A-matrix (data in SMEM from prologue)"
                    .to_string(),
            );
            deleted_loads.push(DeletedCpAsync {
                smem_dst,
                gmem_src,
                mask,
                line: i,
            });
            continue;
        }

        result.push(line.to_string());
    }

    let b_count = classifications
        .values()
        .filter(|c| **c == CpAsyncClass::BMatrix)
        .count();

    Ok(DeleteResult {
        ptx: result.join("\n"),
        a_param: a_param_name,
        deleted_loads,
        a_loads_deleted: a_count,
        b_loads_preserved: b_count,
    })
}

/// Result of deleting A-matrix cp.async loads.
pub struct DeleteResult {
    /// The modified PTX with A-matrix cp.async deleted.
    pub ptx: String,
    /// The param name that was identified as the A matrix.
    pub a_param: String,
    /// Metadata about each deleted cp.async (for prologue to write to).
    pub deleted_loads: Vec<DeletedCpAsync>,
    /// Count of A-matrix cp.async loads deleted.
    pub a_loads_deleted: usize,
    /// Count of B-matrix cp.async loads preserved.
    pub b_loads_preserved: usize,
}

/// Metadata about a deleted cp.async instruction.
pub struct DeletedCpAsync {
    /// SMEM destination register (e.g., "%r226").
    pub smem_dst: String,
    /// GMEM source register (e.g., "%rd26") — indicates which A elements.
    pub gmem_src: String,
    /// Predicate mask register (e.g., "%r227").
    pub mask: String,
    /// Line number in the original PTX.
    pub line: usize,
}

/// Replace A-matrix cp.async loads with explicit ld.global + st.shared.
///
/// This is the passthrough prologue: same data flow as cp.async, but using
/// synchronous instructions. Each cp.async.cg.shared.global [smem], [gmem], 16, mask
/// becomes:
///   setp.ne.b32 %p_fe, mask, 0;
///   @%p_fe  ld.global.v4.b32 {t0,t1,t2,t3}, [gmem];
///   @!%p_fe mov.b32 t0, 0;  (... x4)
///   st.shared.v4.b32 [smem], {t0,t1,t2,t3};
///
/// The SMEM address computation code is preserved — only the cp.async
/// instruction is replaced. B-matrix cp.async loads are unchanged.
pub fn replace_a_loads_with_explicit(ptx: &str, a_param_hint: &str) -> Result<String, String> {
    let lines: Vec<&str> = ptx.lines().collect();
    let proto = PtxParser::parse(ptx)?;
    let reg_to_param = PtxParser::trace_param_registers_pub(&lines, &proto.params);

    let a_param_name = identify_a_matrix_param(&lines, &reg_to_param, a_param_hint)?;

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

    let classifications = classify_cp_async_loads(&lines, &a_addr_regs);
    let a_count = classifications
        .values()
        .filter(|c| **c == CpAsyncClass::AMatrix)
        .count();
    if a_count == 0 {
        return Err("no cp.async loads classified as A-matrix".into());
    }

    // Find existing register counts to allocate temps
    let (pred_max, b32_max) = find_reg_counts(&lines);
    let pred_tmp = pred_max; // one predicate register, reused
    let b32_base = b32_max; // four b32 registers, reused

    let mut result = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Bump register declarations
        if trimmed.starts_with(".reg .pred") && trimmed.contains(&format!("%p<{pred_max}>")) {
            result.push(line.replace(&format!("%p<{pred_max}>"), &format!("%p<{}>", pred_max + 1)));
            continue;
        }
        if trimmed.starts_with(".reg .b32") && trimmed.contains(&format!("%r<{b32_max}>")) {
            result.push(line.replace(&format!("%r<{b32_max}>"), &format!("%r<{}>", b32_max + 4)));
            continue;
        }

        // Replace A-matrix cp.async with explicit ld+st
        if trimmed.contains("cp.async.cg.shared.global")
            && classifications.get(&i) == Some(&CpAsyncClass::AMatrix)
            && let Some((smem_dst, gmem_src, mask)) = parse_cp_async(trimmed)
        {
            let t0 = format!("%r{}", b32_base);
            let t1 = format!("%r{}", b32_base + 1);
            let t2 = format!("%r{}", b32_base + 2);
            let t3 = format!("%r{}", b32_base + 3);
            let p = format!("%p{pred_tmp}");

            result.push(
                "\t// FERRITE: explicit ld+st replacing cp.async for A-matrix".to_string(),
            );
            result.push(format!("\tsetp.ne.b32 {p}, {mask}, 0;"));
            result.push(format!(
                "\t@{p} ld.global.v4.b32 {{{t0}, {t1}, {t2}, {t3}}}, [{gmem_src}];"
            ));
            result.push(format!("\t@!{p} mov.b32 {t0}, 0;"));
            result.push(format!("\t@!{p} mov.b32 {t1}, 0;"));
            result.push(format!("\t@!{p} mov.b32 {t2}, 0;"));
            result.push(format!("\t@!{p} mov.b32 {t3}, 0;"));
            result.push(format!(
                "\tst.shared.v4.b32 [{smem_dst}], {{{t0}, {t1}, {t2}, {t3}}};"
            ));
            continue;
        }

        result.push(line.to_string());
    }

    Ok(result.join("\n"))
}

/// Parse register declaration counts from PTX.
fn find_reg_counts(lines: &[&str]) -> (usize, usize) {
    let mut pred_max = 0usize;
    let mut b32_max = 0usize;

    for line in lines {
        let t = line.trim();
        // .reg .pred %p<75>;
        if t.starts_with(".reg .pred")
            && let Some(start) = t.find("%p<")
        {
            let rest = &t[start + 3..];
            if let Some(end) = rest.find('>')
                && let Ok(n) = rest[..end].parse::<usize>()
            {
                pred_max = pred_max.max(n);
            }
        }
        // .reg .b32 %r<740>;
        if t.starts_with(".reg .b32")
            && let Some(start) = t.find("%r<")
        {
            let rest = &t[start + 3..];
            if let Some(end) = rest.find('>')
                && let Ok(n) = rest[..end].parse::<usize>()
            {
                b32_max = b32_max.max(n);
            }
        }
    }

    (pred_max, b32_max)
}

// ── Internal helpers ──

fn identify_a_matrix_param(
    lines: &[&str],
    reg_to_param: &BTreeMap<String, String>,
    hint: &str,
) -> Result<String, String> {
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

    if !hint.is_empty()
        && let Some(matched) = seen_params.iter().find(|p| p.contains(hint))
    {
        return Ok(matched.clone());
    }

    // First param seen in cp.async order is A (CUTLASS loads A before B)
    Ok(seen_params[0].clone())
}

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

fn parse_cp_async(instr: &str) -> Option<(String, String, String)> {
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

    let parts: Vec<&str> = instr
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    let mask = parts.last()?.trim_end_matches(';').to_string();

    Some((smem_dst, gmem_src, mask))
}
