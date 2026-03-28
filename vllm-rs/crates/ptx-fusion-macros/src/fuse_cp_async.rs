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
pub(crate) enum CpAsyncClass {
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

            result.push("\t// FERRITE: explicit ld+st replacing cp.async for A-matrix".to_string());
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

/// Fuse rms_norm into CUTLASS GEMM by replacing A-matrix loads with
/// inline normalization.
///
/// The fused kernel:
/// 1. Computes inv_rms for each tile row (prologue phase)
/// 2. At each A-load site: loads input, loads weight, normalizes,
///    converts to bf16, writes to CUTLASS's SMEM tile slot
///
/// The host passes input_ptr in the A_ptr field of the CUTLASS params.
/// Weight_ptr, epsilon, and hidden_size are prepended as extra params.
pub fn fuse_rms_norm_into_cutlass(
    ptx: &str,
    a_param_hint: &str,
    entry_name: &str,
) -> Result<String, String> {
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

    let regs = find_all_reg_counts(&lines);

    // Allocate registers for fusion:
    // Predicates: 1 for mask check, 1 for prologue loop
    let p_mask = regs.pred;
    let p_loop = regs.pred + 1;
    // b32: 4 for ld/st temps, 4 for weight temps, 4 for normalized output,
    //       plus prologue scratch
    let r_base = regs.b32;
    let r_t = |i: usize| format!("%r{}", r_base + i); // 0..3: input, 4..7: weight, 8..11: output
    // f32: 2 inv_rms, 8 unpacked input, 8 unpacked weight, sum_sq, temp
    let f_base = regs.f32_;
    let f_inv_rms_0 = format!("%f{}", f_base); // inv_rms for row m0
    let f_inv_rms_1 = format!("%f{}", f_base + 1); // inv_rms for row m0+8
    let f_sum_sq = format!("%f{}", f_base + 2);
    let f_tmp = |i: usize| format!("%f{}", f_base + 3 + i); // scratch f32
    // b64: rms_weight ptr, row_base_0, row_base_1, weight_addr, loop cursor
    let rd_base = regs.b64;
    let rd_rms_weight = format!("%rd{}", rd_base);
    let rd_row_base_0 = format!("%rd{}", rd_base + 1);
    let rd_row_base_1 = format!("%rd{}", rd_base + 2);
    let rd_weight_addr = format!("%rd{}", rd_base + 3);
    let rd_cursor = format!("%rd{}", rd_base + 4);
    let rd_rms_input = format!("%rd{}", rd_base + 5); // only needed in prologue
    let rd_stride = format!("%rd{}", rd_base + 6);

    let new_pred_count = regs.pred + 2;
    let new_b32_count = regs.b32 + 12;
    let new_f32_count = regs.f32_ + 20; // inv_rms(2) + sum_sq(1) + scratch(17)
    let new_b64_count = regs.b64 + 7;

    // Find the original entry name and param name for replacement
    let (orig_entry_name, orig_param_name) = find_entry_and_param(&lines)?;

    // Build the new param name for the wrapper
    let wrapper_param = format!("_ferrite_fused_{entry_name}_param_0");

    let mut result = Vec::new();
    let mut a_load_index = 0usize; // tracks Load 1 vs Load 2 alternation
    let mut in_entry = false;
    let mut _past_regs = false;
    let mut prologue_emitted = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Replace entry point declaration
        if trimmed.contains(".entry") && trimmed.contains(&orig_entry_name) {
            result.push(format!(
                ".visible .entry {entry_name}(\n\
                 \t.param .u64 _ferrite_rms_weight,\n\
                 \t.param .f32 _ferrite_rms_epsilon,\n\
                 \t.param .u32 _ferrite_rms_hidden,\n\
                 \t.param .align 8 .b8 {wrapper_param}[368]"
            ));
            in_entry = true;
            continue;
        }

        // Skip original param declaration (already replaced above)
        if in_entry && trimmed.contains(&orig_param_name) && trimmed.contains("[368]") {
            continue;
        }

        // Close the param list
        if in_entry && trimmed == ")" {
            result.push(")".to_string());
            in_entry = false;
            continue;
        }

        // Bump register declarations
        if trimmed.starts_with(".reg .pred") && trimmed.contains(&format!("%p<{}>", regs.pred)) {
            result.push(line.replace(
                &format!("%p<{}>", regs.pred),
                &format!("%p<{new_pred_count}>"),
            ));
            continue;
        }
        if trimmed.starts_with(".reg .b32") && trimmed.contains(&format!("%r<{}>", regs.b32)) {
            result.push(line.replace(
                &format!("%r<{}>", regs.b32),
                &format!("%r<{new_b32_count}>"),
            ));
            continue;
        }
        if trimmed.starts_with(".reg .f32") && trimmed.contains(&format!("%f<{}>", regs.f32_)) {
            result.push(line.replace(
                &format!("%f<{}>", regs.f32_),
                &format!("%f<{new_f32_count}>"),
            ));
            // Add b16 register declaration for bf16 unpacking (after f32 decl)
            result.push("\t.reg .b16 \t%h<16>;".to_string());
            continue;
        }
        if trimmed.starts_with(".reg .b64") && trimmed.contains(&format!("%rd<{}>", regs.b64)) {
            result.push(line.replace(
                &format!("%rd<{}>", regs.b64),
                &format!("%rd<{new_b64_count}>"),
            ));
            _past_regs = true;
            continue;
        }

        // Replace the param base pointer reference
        if trimmed.contains("mov.b64") && trimmed.contains(&orig_param_name) {
            result.push(line.replace(&orig_param_name, &wrapper_param));
            continue;
        }

        // Replace any other references to the original param name
        if trimmed.contains(&orig_param_name) {
            result.push(line.replace(&orig_param_name, &wrapper_param));
            continue;
        }

        // Replace A-matrix cp.async with normalized ld+st
        // Emit prologue right before the FIRST A-load replacement
        // (at this point %rd1, %r5, %r194 are all set up by CUTLASS code)
        if trimmed.contains("cp.async.cg.shared.global")
            && classifications.get(&i) == Some(&CpAsyncClass::AMatrix)
            && let Some((smem_dst, gmem_src, mask)) = parse_cp_async(trimmed)
        {
            if !prologue_emitted {
                emit_inv_rms_prologue(
                    &mut result,
                    &rd_rms_input,
                    &rd_rms_weight,
                    &f_inv_rms_0,
                    &f_inv_rms_1,
                    &f_sum_sq,
                    &f_tmp,
                    &rd_row_base_0,
                    &rd_row_base_1,
                    &rd_cursor,
                    &rd_stride,
                    &format!("%p{p_loop}"),
                    &wrapper_param,
                    regs.b32,
                );
                prologue_emitted = true;
            }
            let is_load2 = a_load_index % 2 == 1;
            let inv_rms = if is_load2 { &f_inv_rms_1 } else { &f_inv_rms_0 };
            let row_base = if is_load2 {
                &rd_row_base_1
            } else {
                &rd_row_base_0
            };

            emit_normalized_a_load(
                &mut result,
                &smem_dst,
                &gmem_src,
                &mask,
                inv_rms,
                row_base,
                &rd_rms_weight,
                &rd_weight_addr,
                &r_t,
                &f_tmp,
                &format!("%p{p_mask}"),
            );

            a_load_index += 1;
            continue;
        }

        result.push(line.to_string());
    }

    Ok(result.join("\n"))
}

/// Find the original entry name and param name from PTX.
fn find_entry_and_param(lines: &[&str]) -> Result<(String, String), String> {
    let mut entry_name = String::new();
    let mut param_name = String::new();

    for line in lines {
        let t = line.trim();
        if t.contains(".entry")
            && t.contains('(')
            && let Some(start) = t.find("_ZN")
        {
            let end = t.find('(').unwrap_or(t.len());
            entry_name = t[start..end].trim().to_string();
        }
        if t.contains(".param") && t.contains("[368]") {
            // Extract param name: .param .align 8 .b8 NAME[368]
            let parts: Vec<&str> = t.split_whitespace().collect();
            for (j, p) in parts.iter().enumerate() {
                if p.contains("[368]") {
                    param_name = p.replace("[368]", "");
                    break;
                }
                if j > 0 && parts.get(j + 1).is_some_and(|n| n.contains("[368]")) {
                    // sometimes it's split: NAME [368]
                    param_name = p.to_string();
                }
            }
        }
    }

    if entry_name.is_empty() || param_name.is_empty() {
        return Err("could not find entry name or param name".into());
    }
    Ok((entry_name, param_name))
}

/// Emit the inv_rms prologue: computes inv_rms for 2 rows per thread.
fn emit_inv_rms_prologue(
    out: &mut Vec<String>,
    rd_rms_input: &str,
    rd_rms_weight: &str,
    f_inv_rms_0: &str,
    f_inv_rms_1: &str,
    f_sum_sq: &str,
    f_tmp: &dyn Fn(usize) -> String,
    rd_row_base_0: &str,
    rd_row_base_1: &str,
    rd_cursor: &str,
    rd_stride: &str,
    p_loop: &str,
    _wrapper_param: &str,
    b32_base: usize,
) {
    out.push("\t// -- FERRITE: rms_norm prologue (compute inv_rms per tile row) --".into());

    // Load rms params from prepended declarations
    // A_ptr in CUTLASS params is now input_ptr (set by host)
    // We load weight, epsilon, hidden from our extra params
    let r_hidden = format!("%r{}", b32_base + 8);
    let f_epsilon = f_tmp(10);
    let f_hidden_f = f_tmp(12);

    out.push(format!(
        "\tld.param.u64 {rd_rms_weight}, [_ferrite_rms_weight];"
    ));
    out.push(format!(
        "\tld.param.f32 {f_epsilon}, [_ferrite_rms_epsilon];"
    ));
    out.push(format!("\tld.param.u32 {r_hidden}, [_ferrite_rms_hidden];"));
    // Convert hidden to f32 for division later
    out.push(format!("\tcvt.rn.f32.u32 {f_hidden_f}, {r_hidden};"));

    // Load A_ptr (= input_ptr) from CUTLASS params
    // %rd1 is set up right before this prologue: %rd1 = param_base + 24
    // A_ptr is at [%rd1+40] (same offset the CUTLASS code uses)
    out.push(format!("\tld.param.u64 {rd_rms_input}, [%rd1+40];"));

    // Load A_stride from CUTLASS params: [%rd1+8]
    out.push(format!("\tld.param.u64 {rd_stride}, [%rd1+8];"));

    // Thread-to-row mapping (matches CUTLASS tile mapping)
    // m0 = lane_id/4 + warp_id*16
    // lane_id = tid.x % 32, warp_id = tid.x / 32
    // lane_group = lane_id / 4 = (tid.x % 32) / 4
    // m0 = lane_group + warp_id * 16
    // m1 = m0 + 8
    // Note: m0 here is tile-relative (0..63), not absolute
    //
    // But we need the ABSOLUTE row for GMEM access.
    // Absolute m = m0 + block_m * 64
    // block_m is already factored into the CUTLASS address computation.
    // For the prologue, we need to compute the row base address ourselves:
    //   row_base = input_ptr + absolute_m * stride * 2 (bf16)
    //
    // The CUTLASS code computes: %r194 = lane/4 + warp*16 + block_m*64
    // and: address = input_ptr + (%r194 * stride + k_offset) * 2
    // For k_offset=0: row_base = input_ptr + %r194 * stride * 2
    //
    // We can use %r194 which is already computed before the prologue
    // (line 97 in the PTX: add.s32 %r194, %r193, %r8)
    // And %r192 contains k_batch_offset (which might not be 0 for split-K)
    //
    // Simpler: recompute from tid.x since it's cleaner
    out.push("\t// Thread-to-row: m0 = (tid.x%32)/4 + (tid.x/32)*16".to_string());
    // We reuse %r5 = tid.x (already set by CUTLASS code line 79)
    // But the prologue runs AFTER the CUTLASS setup code, so %r5 is available.
    // Actually, %r5 might be overwritten. Let's use the existing %r194 and %r192.
    //
    // Wait — %r194 is the ABSOLUTE m index (including block_m*64).
    // %r192 is the k_batch_offset.
    // For row_base: row_base = input_ptr + %r194 * stride * 2
    //   = input_ptr + cvt.s64(%r194) * stride * 2
    // But stride is already in byte units? Need to check.
    // From PTX: %rd6 = %rd51 * %rd50 (line 110)
    //   %rd51 = stride, %rd50 = col index
    //   Then %rd52 = %rd6 + %rd5 (add k_offset)
    //   Then %rd53 = %rd52 << 1 (multiply by 2 for bf16)
    // So stride is in ELEMENT units, not bytes.
    // row_base = input_ptr + m * stride * 2
    //
    // Use: cvt.s64 m -> mul.lo stride -> shl 1 -> add input_ptr
    out.push(format!("\tcvt.s64.s32 {rd_row_base_0}, %r194;"));
    out.push(format!(
        "\tmul.lo.s64 {rd_row_base_0}, {rd_stride}, {rd_row_base_0};"
    ));
    out.push(format!("\tshl.b64 {rd_row_base_0}, {rd_row_base_0}, 1;"));
    out.push(format!(
        "\tadd.s64 {rd_row_base_0}, {rd_rms_input}, {rd_row_base_0};"
    ));

    // row_base_1 = row_base_0 + 8 * stride * 2 (m0+8 row)
    // 8 * stride * 2 = stride << 4
    out.push(format!("\tshl.b64 {rd_row_base_1}, {rd_stride}, 4;"));
    out.push(format!(
        "\tadd.s64 {rd_row_base_1}, {rd_row_base_0}, {rd_row_base_1};"
    ));

    // ── Reduction loop: accumulate sum_of_squares for both rows ──
    // Each thread processes hidden/4 bf16 elements (4 threads per row)
    // Thread's k_start = (lane_id % 4) * (hidden / 4)
    // But elements are bf16 packed as b32 (2 per word), so we load v2.b32 = 4 bf16
    //
    // Actually simpler: just iterate over ALL elements, but each of 4 threads
    // handles every 4th group. thread_k = (tid.x % 4) * 8, stride = 32 elements
    // (since 4 threads × 8 elements = 32 elements per group)
    //
    // For hidden=32 (our test case): each thread loads 8 elements (one v4.b32)
    // For hidden=4096: each thread loads 1024 elements (128 v4.b32 loads)

    out.push(format!("\tmov.f32 {f_sum_sq}, 0f00000000;")); // sum = 0
    out.push(format!("\tmov.f32 {}, 0f00000000;", f_tmp(13))); // sum1 = 0

    // k_start for this thread: (lane_id%4) * 8 elements = (lane_id%4) * 16 bytes
    out.push(format!("\tand.b32 {}, %r5, 3;", f_tmp(14))); // reuse f_tmp slot as b32
    // Actually we can't use f32 register for integer ops. We need a b32 temp.
    // Let me use an unallocated b32 register. We allocated r_base..r_base+11.
    // But wait — f_tmp(14) is an f32 register, not b32. Let me fix this.

    // Use the b32 temps for integer arithmetic
    let r_k_start = format!("%r{}", b32_base + 8); // index 8 in our allocation
    let r_k_step = format!("%r{}", b32_base + 9);
    let r_k_end = format!("%r{}", b32_base + 10);
    let _r_itmp = format!("%r{}", b32_base + 11);

    out.push(format!("\tand.b32 {r_k_start}, %r5, 3;")); // lane % 4
    out.push(format!("\tshl.b32 {r_k_start}, {r_k_start}, 4;")); // * 16 bytes (8 bf16 elements)
    out.push(format!("\tmov.u32 {r_k_step}, 64;")); // 4 threads * 16 bytes = 64 byte stride

    // k_end = hidden * 2 bytes (bf16)
    out.push(format!("\tld.param.u32 {r_k_end}, [_ferrite_rms_hidden];"));
    out.push(format!("\tshl.b32 {r_k_end}, {r_k_end}, 1;")); // hidden * 2 bytes

    // Loop cursor: rd_cursor = row_base_0 + k_start
    out.push(format!("\tcvt.u64.u32 {rd_cursor}, {r_k_start};"));
    out.push(format!(
        "\tadd.s64 {rd_cursor}, {rd_row_base_0}, {rd_cursor};"
    ));

    // Row 0 reduction loop
    out.push("$L_ferrite_rms_loop_0:".into());
    // Check: k_start < k_end
    out.push(format!("\tsetp.lt.u32 {p_loop}, {r_k_start}, {r_k_end};"));
    out.push(format!("\t@!{p_loop} bra $L_ferrite_rms_done_0;"));

    // Load 4 b32 = 8 bf16 elements from input[m0, k]
    let r_in = |j: usize| format!("%r{}", b32_base + j);
    out.push(format!(
        "\tld.global.v4.b32 {{{}, {}, {}, {}}}, [{rd_cursor}];",
        r_in(0),
        r_in(1),
        r_in(2),
        r_in(3)
    ));

    // Unpack and accumulate: for each b32, extract 2 bf16, cvt to f32, square, add
    for j in 0..4 {
        let h_lo = format!("%h{}", j * 2);
        let h_hi = format!("%h{}", j * 2 + 1);
        out.push(format!("\tmov.b32 {{{h_lo}, {h_hi}}}, {};", r_in(j)));
        out.push(format!("\tcvt.f32.bf16 {}, {h_lo};", f_tmp(0)));
        out.push(format!("\tcvt.f32.bf16 {}, {h_hi};", f_tmp(1)));
        out.push(format!(
            "\tfma.rn.f32 {f_sum_sq}, {}, {}, {f_sum_sq};",
            f_tmp(0),
            f_tmp(0)
        ));
        out.push(format!(
            "\tfma.rn.f32 {f_sum_sq}, {}, {}, {f_sum_sq};",
            f_tmp(1),
            f_tmp(1)
        ));
    }

    // Advance cursor
    out.push(format!("\tcvt.u64.u32 {rd_rms_input}, {r_k_step};")); // reuse rd as temp
    out.push(format!(
        "\tadd.s64 {rd_cursor}, {rd_cursor}, {rd_rms_input};"
    ));
    out.push(format!("\tadd.u32 {r_k_start}, {r_k_start}, {r_k_step};"));
    out.push("\tbra $L_ferrite_rms_loop_0;".to_string());
    out.push("$L_ferrite_rms_done_0:".into());

    // Row 1 reduction loop (same but using row_base_1)
    out.push(format!("\tand.b32 {r_k_start}, %r5, 3;"));
    out.push(format!("\tshl.b32 {r_k_start}, {r_k_start}, 4;"));
    out.push(format!("\tcvt.u64.u32 {rd_cursor}, {r_k_start};"));
    out.push(format!(
        "\tadd.s64 {rd_cursor}, {rd_row_base_1}, {rd_cursor};"
    ));

    out.push("$L_ferrite_rms_loop_1:".into());
    out.push(format!("\tsetp.lt.u32 {p_loop}, {r_k_start}, {r_k_end};"));
    out.push(format!("\t@!{p_loop} bra $L_ferrite_rms_done_1;"));

    out.push(format!(
        "\tld.global.v4.b32 {{{}, {}, {}, {}}}, [{rd_cursor}];",
        r_in(0),
        r_in(1),
        r_in(2),
        r_in(3)
    ));

    for j in 0..4 {
        let h_lo = format!("%h{}", j * 2);
        let h_hi = format!("%h{}", j * 2 + 1);
        out.push(format!("\tmov.b32 {{{h_lo}, {h_hi}}}, {};", r_in(j)));
        out.push(format!("\tcvt.f32.bf16 {}, {h_lo};", f_tmp(0)));
        out.push(format!("\tcvt.f32.bf16 {}, {h_hi};", f_tmp(1)));
        out.push(format!(
            "\tfma.rn.f32 {}, {}, {}, {};",
            f_tmp(13),
            f_tmp(0),
            f_tmp(0),
            f_tmp(13)
        ));
        out.push(format!(
            "\tfma.rn.f32 {}, {}, {}, {};",
            f_tmp(13),
            f_tmp(1),
            f_tmp(1),
            f_tmp(13)
        ));
    }

    out.push(format!("\tcvt.u64.u32 {rd_rms_input}, {r_k_step};"));
    out.push(format!(
        "\tadd.s64 {rd_cursor}, {rd_cursor}, {rd_rms_input};"
    ));
    out.push(format!("\tadd.u32 {r_k_start}, {r_k_start}, {r_k_step};"));
    out.push("\tbra $L_ferrite_rms_loop_1;".to_string());
    out.push("$L_ferrite_rms_done_1:".into());

    // ── Reduce across 4 threads (lane%4 group) via shfl.sync.bfly ──
    // Butterfly reduction: xor mask 1 then 2
    for xor_mask in [1, 2] {
        out.push(format!(
            "\tshfl.sync.bfly.b32 {}, {f_sum_sq}, {xor_mask}, 31, -1;",
            f_tmp(0)
        ));
        out.push(format!("\tadd.f32 {f_sum_sq}, {f_sum_sq}, {};", f_tmp(0)));
        out.push(format!(
            "\tshfl.sync.bfly.b32 {}, {}, {xor_mask}, 31, -1;",
            f_tmp(0),
            f_tmp(13)
        ));
        out.push(format!(
            "\tadd.f32 {}, {}, {};",
            f_tmp(13),
            f_tmp(13),
            f_tmp(0)
        ));
    }

    // inv_rms_0 = rsqrt(sum_sq_0 / hidden + epsilon)
    out.push(format!(
        "\tdiv.rn.f32 {f_sum_sq}, {f_sum_sq}, {f_hidden_f};"
    ));
    out.push(format!("\tadd.f32 {f_sum_sq}, {f_sum_sq}, {f_epsilon};"));
    out.push(format!("\trsqrt.approx.f32 {f_inv_rms_0}, {f_sum_sq};"));

    // inv_rms_1
    out.push(format!(
        "\tdiv.rn.f32 {}, {}, {f_hidden_f};",
        f_tmp(13),
        f_tmp(13),
    ));
    out.push(format!(
        "\tadd.f32 {}, {}, {f_epsilon};",
        f_tmp(13),
        f_tmp(13),
    ));
    out.push(format!("\trsqrt.approx.f32 {f_inv_rms_1}, {};", f_tmp(13)));

    // Reload rms_input for weight addressing later (was clobbered as temp)
    out.push(format!("\tld.param.u64 {rd_rms_input}, [%rd1+40];"));

    out.push("\tbar.sync \t15;".into()); // barrier 15 (avoid conflict with CUTLASS barrier 0)
    out.push("\t// -- FERRITE: prologue done, inv_rms computed --".into());
}

/// Emit normalized A-load: load input, load weight, normalize, write to SMEM.
fn emit_normalized_a_load(
    out: &mut Vec<String>,
    smem_dst: &str,
    gmem_src: &str,
    mask: &str,
    inv_rms: &str,
    row_base: &str,
    rd_rms_weight: &str,
    rd_weight_addr: &str,
    r_t: &dyn Fn(usize) -> String,
    f_tmp: &dyn Fn(usize) -> String,
    p_mask: &str,
) {
    out.push("\t// FERRITE: normalized A-load (rms_norm fused)".into());
    out.push(format!("\tsetp.ne.b32 {p_mask}, {mask}, 0;"));

    // Load input[m, k:k+8] (same address as original A-load)
    out.push(format!(
        "\t@{p_mask} ld.global.v4.b32 {{{}, {}, {}, {}}}, [{gmem_src}];",
        r_t(0),
        r_t(1),
        r_t(2),
        r_t(3)
    ));

    // Compute weight address: weight_ptr + (gmem_src - row_base)
    out.push(format!(
        "\tsub.s64 {rd_weight_addr}, {gmem_src}, {row_base};"
    ));
    out.push(format!(
        "\tadd.s64 {rd_weight_addr}, {rd_rms_weight}, {rd_weight_addr};"
    ));

    // Load weight[k:k+8]
    out.push(format!(
        "\t@{p_mask} ld.global.v4.b32 {{{}, {}, {}, {}}}, [{rd_weight_addr}];",
        r_t(4),
        r_t(5),
        r_t(6),
        r_t(7)
    ));

    // For each b32 pair: unpack bf16, cvt to f32, multiply input*weight*inv_rms,
    // cvt back to bf16, pack into b32
    for j in 0..4 {
        let h_in_lo = format!("%h{}", 8 + j * 2);
        let h_in_hi = format!("%h{}", 8 + j * 2 + 1);
        let h_wt_lo = format!("%h{}", j * 2);
        let h_wt_hi = format!("%h{}", j * 2 + 1);

        // Unpack input
        out.push(format!(
            "\t@{p_mask} mov.b32 {{{h_in_lo}, {h_in_hi}}}, {};",
            r_t(j)
        ));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 {}, {h_in_lo};", f_tmp(0)));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 {}, {h_in_hi};", f_tmp(1)));

        // Unpack weight
        out.push(format!(
            "\t@{p_mask} mov.b32 {{{h_wt_lo}, {h_wt_hi}}}, {};",
            r_t(4 + j)
        ));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 {}, {h_wt_lo};", f_tmp(2)));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 {}, {h_wt_hi};", f_tmp(3)));

        // Normalize: result = input * weight * inv_rms
        out.push(format!(
            "\t@{p_mask} mul.f32 {}, {}, {};",
            f_tmp(0),
            f_tmp(0),
            f_tmp(2)
        )); // input_lo * weight_lo
        out.push(format!(
            "\t@{p_mask} mul.f32 {}, {}, {inv_rms};",
            f_tmp(0),
            f_tmp(0)
        )); // * inv_rms
        out.push(format!(
            "\t@{p_mask} mul.f32 {}, {}, {};",
            f_tmp(1),
            f_tmp(1),
            f_tmp(3)
        ));
        out.push(format!(
            "\t@{p_mask} mul.f32 {}, {}, {inv_rms};",
            f_tmp(1),
            f_tmp(1)
        ));

        // Convert back to bf16x2 and pack
        out.push(format!(
            "\t@{p_mask} cvt.rn.bf16x2.f32 {}, {}, {};",
            r_t(j),
            f_tmp(1),
            f_tmp(0)
        )); // note: high, low order
    }

    // Zero-fill path for boundary tiles
    out.push(format!("\t@!{p_mask} mov.b32 {}, 0;", r_t(0)));
    out.push(format!("\t@!{p_mask} mov.b32 {}, 0;", r_t(1)));
    out.push(format!("\t@!{p_mask} mov.b32 {}, 0;", r_t(2)));
    out.push(format!("\t@!{p_mask} mov.b32 {}, 0;", r_t(3)));

    // Write to SMEM
    out.push(format!(
        "\tst.shared.v4.b32 [{smem_dst}], {{{}, {}, {}, {}}};",
        r_t(0),
        r_t(1),
        r_t(2),
        r_t(3)
    ));
}

/// All register counts from PTX declarations.
pub(crate) struct RegCounts {
    pub(crate) pred: usize,
    pub(crate) b32: usize,
    pub(crate) f32_: usize,
    pub(crate) b64: usize,
}

/// Parse register declaration counts from PTX.
fn find_reg_counts(lines: &[&str]) -> (usize, usize) {
    let c = find_all_reg_counts(lines);
    (c.pred, c.b32)
}

pub(crate) fn find_all_reg_counts(lines: &[&str]) -> RegCounts {
    let mut counts = RegCounts {
        pred: 0,
        b32: 0,
        f32_: 0,
        b64: 0,
    };

    for line in lines {
        let t = line.trim();
        if let Some(n) = parse_reg_decl(t, ".reg .pred", "%p<") {
            counts.pred = counts.pred.max(n);
        }
        if let Some(n) = parse_reg_decl(t, ".reg .b32", "%r<") {
            counts.b32 = counts.b32.max(n);
        }
        if let Some(n) = parse_reg_decl(t, ".reg .f32", "%f<") {
            counts.f32_ = counts.f32_.max(n);
        }
        if let Some(n) = parse_reg_decl(t, ".reg .b64", "%rd<") {
            counts.b64 = counts.b64.max(n);
        }
    }

    counts
}

fn parse_reg_decl(line: &str, prefix: &str, reg_prefix: &str) -> Option<usize> {
    if !line.starts_with(prefix) {
        return None;
    }
    let start = line.find(reg_prefix)? + reg_prefix.len();
    let rest = &line[start..];
    let end = rest.find('>')?;
    rest[..end].parse().ok()
}

// ── Internal helpers ──

pub(crate) fn identify_a_matrix_param(
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

pub(crate) fn classify_cp_async_loads(
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

pub(crate) fn parse_cp_async(instr: &str) -> Option<(String, String, String)> {
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
