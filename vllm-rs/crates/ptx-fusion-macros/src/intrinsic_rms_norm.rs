//! rms_norm intrinsic: generate rms_norm computation from scratch as PTX,
//! tailored to the GEMM's thread-to-row mapping.
//!
//! Instead of extracting from a .ptx file, we GENERATE the rms_norm PTX
//! because the formula is simple and the thread mapping must match the
//! consumer GEMM's tile layout.
//!
//! The generated code has two parts:
//! - **Prologue**: cooperative reduction (sum of squares -> inv_rms) for
//!   each tile row. Runs once before the first A-load in the GEMM's loop.
//! - **Per-element**: at each A-load site, load input + weight, normalize
//!   (input * weight * inv_rms), write to SMEM.
//!
//! Thread-to-row mapping for CUTLASS bf16 GEMM (PitchLinearWarpRakedThreadMap):
//!   m_row = (tid.x % 32) / 4 + (tid.x / 32) * 16
//!   Each thread handles 2 rows: m_row and m_row + 8
//!   4 threads per row cooperate on the K-dimension reduction

use crate::fuse_cp_async::{
    CpAsyncClass, classify_cp_async_loads, find_all_reg_counts, identify_a_matrix_param,
    parse_cp_async,
};
use crate::parser::PtxParser;

/// Build a fused rms_norm + GEMM kernel.
///
/// Takes the original CUTLASS GEMM PTX and produces a modified kernel that:
/// 1. Has extra params: weight_ptr (u64), epsilon (f32), hidden_size (u32)
/// 2. Computes inv_rms per tile row before the main loop
/// 3. At each A-matrix cp.async site: loads input, loads weight, normalizes, writes to SMEM
///
/// The result can then be passed to `replace_perimeter()` to flatten the CUTLASS params.
pub fn build_rms_norm_gemm(
    gemm_ptx: &str,
    a_param_hint: &str,
    entry_name: &str,
) -> Result<String, String> {
    let lines: Vec<&str> = gemm_ptx.lines().collect();
    let proto = PtxParser::parse(gemm_ptx)?;
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

    // Allocate registers for fusion
    let p_mask = regs.pred;
    let p_loop = regs.pred + 1;
    let r_base = regs.b32;
    let f_base = regs.f32_;
    let rd_base = regs.b64;

    // Register naming helpers
    let r_t = |i: usize| format!("%r{}", r_base + i); // 0..3: input, 4..7: weight, 8..11: scratch
    let f_tmp = |i: usize| format!("%f{}", f_base + i);
    let f_inv_rms_0 = format!("%f{}", f_base);
    let f_inv_rms_1 = format!("%f{}", f_base + 1);
    let f_sum_sq_0 = format!("%f{}", f_base + 2);
    let f_sum_sq_1 = format!("%f{}", f_base + 3);
    let f_epsilon = format!("%f{}", f_base + 4);
    let f_hidden_f = format!("%f{}", f_base + 5);
    // f_tmp(6)..f_tmp(9) for unpack/compute scratch

    let rd_weight = format!("%rd{}", rd_base);
    let rd_input = format!("%rd{}", rd_base + 1);
    let rd_row_base_0 = format!("%rd{}", rd_base + 2);
    let rd_row_base_1 = format!("%rd{}", rd_base + 3);
    let rd_stride = format!("%rd{}", rd_base + 4);
    let rd_cursor = format!("%rd{}", rd_base + 5);
    let rd_weight_addr = format!("%rd{}", rd_base + 6);

    let new_pred_count = regs.pred + 2;
    let new_b32_count = regs.b32 + 12;
    let new_f32_count = regs.f32_ + 10;
    let new_b64_count = regs.b64 + 7;

    let p_m = format!("%p{}", p_mask);
    let p_l = format!("%p{}", p_loop);

    // Find the original entry and param names
    let orig_entry = find_entry_name(&lines)?;
    let orig_param = find_struct_param_name(&lines)?;

    let mut result = Vec::new();
    let mut prologue_emitted = false;
    let mut a_load_index = 0usize;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Replace entry declaration: add rms_norm params
        if trimmed.contains(".entry") && trimmed.contains(&orig_entry) {
            result.push(format!(".visible .entry {entry_name}("));
            result.push("\t.param .u64 _ferrite_rms_weight,".into());
            result.push("\t.param .f32 _ferrite_rms_epsilon,".into());
            result.push("\t.param .u32 _ferrite_rms_hidden,".into());
            // The original CUTLASS struct param follows
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
            result.push("\t.reg .b16 \t%h_rms<16>;".to_string());
            continue;
        }
        if trimmed.starts_with(".reg .b64") && trimmed.contains(&format!("%rd<{}>", regs.b64)) {
            result.push(line.replace(
                &format!("%rd<{}>", regs.b64),
                &format!("%rd<{new_b64_count}>"),
            ));
            continue;
        }

        // Replace A-matrix cp.async with inline normalized load
        if trimmed.contains("cp.async.cg.shared.global")
            && classifications.get(&i) == Some(&CpAsyncClass::AMatrix)
        {
            if let Some((smem_dst, gmem_src, mask)) = parse_cp_async(trimmed) {
                // Emit prologue once before the first A-load
                if !prologue_emitted {
                    emit_rms_prologue(
                        &mut result,
                        &rd_input,
                        &rd_weight,
                        &rd_stride,
                        &rd_row_base_0,
                        &rd_row_base_1,
                        &rd_cursor,
                        &f_inv_rms_0,
                        &f_inv_rms_1,
                        &f_sum_sq_0,
                        &f_sum_sq_1,
                        &f_epsilon,
                        &f_hidden_f,
                        &f_tmp,
                        &r_t,
                        &p_l,
                        &orig_param,
                    );
                    prologue_emitted = true;
                }

                // Determine which inv_rms to use: even loads use inv_rms_0,
                // alternating pattern (A loads alternate between row m0 and m0+8)
                let inv_rms = if a_load_index % 2 == 0 {
                    &f_inv_rms_0
                } else {
                    &f_inv_rms_1
                };
                let row_base = if a_load_index % 2 == 0 {
                    &rd_row_base_0
                } else {
                    &rd_row_base_1
                };

                emit_normalized_load(
                    &mut result,
                    &smem_dst,
                    &gmem_src,
                    &mask,
                    inv_rms,
                    row_base,
                    &rd_weight,
                    &rd_weight_addr,
                    &r_t,
                    &f_tmp,
                    &p_m,
                );

                a_load_index += 1;
                continue;
            }
        }

        result.push(line.to_string());
    }

    Ok(result.join("\n"))
}

// ── Prologue: compute inv_rms for 2 tile rows ──

#[allow(clippy::too_many_arguments)]
fn emit_rms_prologue(
    out: &mut Vec<String>,
    rd_input: &str,
    rd_weight: &str,
    rd_stride: &str,
    rd_row_base_0: &str,
    rd_row_base_1: &str,
    rd_cursor: &str,
    f_inv_rms_0: &str,
    f_inv_rms_1: &str,
    f_sum_sq_0: &str,
    f_sum_sq_1: &str,
    f_epsilon: &str,
    f_hidden_f: &str,
    f_tmp: &dyn Fn(usize) -> String,
    r_t: &dyn Fn(usize) -> String,
    p_loop: &str,
    orig_param: &str,
) {
    out.push("\t// -- FERRITE: rms_norm prologue (intrinsic) --".into());

    // Load rms params
    out.push(format!(
        "\tld.param.u64 \t{rd_weight}, [_ferrite_rms_weight];"
    ));
    out.push(format!(
        "\tld.param.f32 \t{f_epsilon}, [_ferrite_rms_epsilon];"
    ));
    let r_hidden = r_t(8);
    out.push(format!(
        "\tld.param.u32 \t{r_hidden}, [_ferrite_rms_hidden];"
    ));
    out.push(format!("\tcvt.rn.f32.u32 \t{f_hidden_f}, {r_hidden};"));

    // Load A_ptr and stride from params.
    // After perimeter replacement, these are in ferrite_params at known offsets.
    // Before perimeter replacement, they're in the CUTLASS struct via %rd1.
    // We emit both patterns — perimeter replacement will rewrite the struct version.
    // The flat-param version: A_ptr at offset 0, lda at offset 32.
    // But we're running BEFORE perimeter replacement, so use CUTLASS struct offsets.
    // %rd1 = param base, A_ptr at [%rd1+40], stride at [%rd1+8].
    out.push(format!("\tld.param.u64 \t{rd_input}, [%rd1+40];"));
    out.push(format!("\tld.param.u64 \t{rd_stride}, [%rd1+8];"));

    // Thread-to-row mapping (CUTLASS PitchLinearWarpRakedThreadMap):
    // m_row = (tid.x % 32) / 4 + (tid.x / 32) * 16
    // Each thread handles row m_row and m_row + 8.
    // The ABSOLUTE m is already in %r5 (set by CUTLASS code before we get here):
    //   No — %r5 = tid.x. The absolute M row is computed by CUTLASS as a complex chain.
    // We recompute from tid.x (%r264 in the PTX, but we use %r5 which is lane_id? No.)
    // Let me use the known CUTLASS register: %r280 = absolute M row for this thread.
    // %r280 is set at line 102 of the PTX.
    //
    // Actually, %r280 may not be set yet when the prologue runs (prologue runs at line ~290
    // right before the first cp.async, but %r280 is set at line 102). So it IS available.
    //
    // row_base_0 = input_ptr + %r280 * stride * 2 (bf16 = 2 bytes per element)
    out.push(format!("\tcvt.s64.s32 \t{rd_row_base_0}, %r280;"));
    out.push(format!(
        "\tmul.lo.s64 \t{rd_row_base_0}, {rd_stride}, {rd_row_base_0};"
    ));
    out.push(format!("\tshl.b64 \t{rd_row_base_0}, {rd_row_base_0}, 1;"));
    out.push(format!(
        "\tadd.s64 \t{rd_row_base_0}, {rd_input}, {rd_row_base_0};"
    ));

    // row_base_1 = row_base_0 + 8 * stride * 2
    out.push(format!("\tshl.b64 \t{rd_row_base_1}, {rd_stride}, 4;")); // stride * 16 = stride * 8 * 2
    out.push(format!(
        "\tadd.s64 \t{rd_row_base_1}, {rd_row_base_0}, {rd_row_base_1};"
    ));

    // ── Reduction: sum of squares for row 0 ──
    let r_k_start = r_t(9);
    let r_k_step = r_t(10);
    let r_k_end = r_t(11);

    out.push(format!("\tmov.f32 \t{f_sum_sq_0}, 0f00000000;")); // sum0 = 0
    out.push(format!("\tmov.f32 \t{f_sum_sq_1}, 0f00000000;")); // sum1 = 0

    // Each of 4 threads handles every 4th group of 8 bf16 elements
    // k_start = (tid.x % 4) * 16 bytes, k_step = 64 bytes
    out.push(format!("\tand.b32 \t{r_k_start}, %r264, 3;")); // tid.x % 4 (use raw tid.x)
    out.push(format!("\tshl.b32 \t{r_k_start}, {r_k_start}, 4;")); // * 16 bytes
    out.push(format!("\tmov.u32 \t{r_k_step}, 64;")); // 4 threads * 16 bytes
    out.push(format!("\tshl.b32 \t{r_k_end}, {r_hidden}, 1;")); // hidden * 2 bytes

    // Row 0 loop
    out.push(format!("\tcvt.u64.u32 \t{rd_cursor}, {r_k_start};"));
    out.push(format!(
        "\tadd.s64 \t{rd_cursor}, {rd_row_base_0}, {rd_cursor};"
    ));

    out.push("$L_ferrite_rms_r0:".into());
    out.push(format!("\tsetp.lt.u32 \t{p_loop}, {r_k_start}, {r_k_end};"));
    out.push(format!("\t@!{p_loop} bra $L_ferrite_rms_r0d;"));

    out.push(format!(
        "\tld.global.v4.b32 \t{{{}, {}, {}, {}}}, [{rd_cursor}];",
        r_t(0),
        r_t(1),
        r_t(2),
        r_t(3)
    ));

    for j in 0..4 {
        let h0 = format!("%h_rms{}", j * 2);
        let h1 = format!("%h_rms{}", j * 2 + 1);
        out.push(format!("\tmov.b32 \t{{{h0}, {h1}}}, {};", r_t(j)));
        out.push(format!("\tcvt.f32.bf16 \t{}, {h0};", f_tmp(6)));
        out.push(format!("\tcvt.f32.bf16 \t{}, {h1};", f_tmp(7)));
        out.push(format!(
            "\tfma.rn.f32 \t{f_sum_sq_0}, {}, {}, {f_sum_sq_0};",
            f_tmp(6),
            f_tmp(6)
        ));
        out.push(format!(
            "\tfma.rn.f32 \t{f_sum_sq_0}, {}, {}, {f_sum_sq_0};",
            f_tmp(7),
            f_tmp(7)
        ));
    }

    out.push(format!("\tcvt.u64.u32 \t{rd_input}, {r_k_step};")); // temp
    out.push(format!("\tadd.s64 \t{rd_cursor}, {rd_cursor}, {rd_input};"));
    out.push(format!("\tadd.u32 \t{r_k_start}, {r_k_start}, {r_k_step};"));
    out.push("\tbra $L_ferrite_rms_r0;".into());
    out.push("$L_ferrite_rms_r0d:".into());

    // Row 1 loop
    out.push(format!("\tand.b32 \t{r_k_start}, %r264, 3;"));
    out.push(format!("\tshl.b32 \t{r_k_start}, {r_k_start}, 4;"));
    out.push(format!("\tcvt.u64.u32 \t{rd_cursor}, {r_k_start};"));
    out.push(format!(
        "\tadd.s64 \t{rd_cursor}, {rd_row_base_1}, {rd_cursor};"
    ));

    out.push("$L_ferrite_rms_r1:".into());
    out.push(format!("\tsetp.lt.u32 \t{p_loop}, {r_k_start}, {r_k_end};"));
    out.push(format!("\t@!{p_loop} bra $L_ferrite_rms_r1d;"));

    out.push(format!(
        "\tld.global.v4.b32 \t{{{}, {}, {}, {}}}, [{rd_cursor}];",
        r_t(0),
        r_t(1),
        r_t(2),
        r_t(3)
    ));

    for j in 0..4 {
        let h0 = format!("%h_rms{}", j * 2);
        let h1 = format!("%h_rms{}", j * 2 + 1);
        out.push(format!("\tmov.b32 \t{{{h0}, {h1}}}, {};", r_t(j)));
        out.push(format!("\tcvt.f32.bf16 \t{}, {h0};", f_tmp(6)));
        out.push(format!("\tcvt.f32.bf16 \t{}, {h1};", f_tmp(7)));
        out.push(format!(
            "\tfma.rn.f32 \t{f_sum_sq_1}, {}, {}, {f_sum_sq_1};",
            f_tmp(6),
            f_tmp(6)
        ));
        out.push(format!(
            "\tfma.rn.f32 \t{f_sum_sq_1}, {}, {}, {f_sum_sq_1};",
            f_tmp(7),
            f_tmp(7)
        ));
    }

    out.push(format!("\tcvt.u64.u32 \t{rd_input}, {r_k_step};"));
    out.push(format!("\tadd.s64 \t{rd_cursor}, {rd_cursor}, {rd_input};"));
    out.push(format!("\tadd.u32 \t{r_k_start}, {r_k_start}, {r_k_step};"));
    out.push("\tbra $L_ferrite_rms_r1;".into());
    out.push("$L_ferrite_rms_r1d:".into());

    // Butterfly reduction across 4 threads (shfl.sync.bfly xor 1, then xor 2)
    for xor_mask in [1, 2] {
        out.push(format!(
            "\tshfl.sync.bfly.b32 \t{}, {f_sum_sq_0}, {xor_mask}, 31, -1;",
            f_tmp(6)
        ));
        out.push(format!(
            "\tadd.f32 \t{f_sum_sq_0}, {f_sum_sq_0}, {};",
            f_tmp(6)
        ));
        out.push(format!(
            "\tshfl.sync.bfly.b32 \t{}, {f_sum_sq_1}, {xor_mask}, 31, -1;",
            f_tmp(6)
        ));
        out.push(format!(
            "\tadd.f32 \t{f_sum_sq_1}, {f_sum_sq_1}, {};",
            f_tmp(6)
        ));
    }

    // inv_rms = rsqrt(sum_sq / hidden + epsilon)
    out.push(format!(
        "\tdiv.rn.f32 \t{f_sum_sq_0}, {f_sum_sq_0}, {f_hidden_f};"
    ));
    out.push(format!(
        "\tadd.f32 \t{f_sum_sq_0}, {f_sum_sq_0}, {f_epsilon};"
    ));
    out.push(format!("\trsqrt.approx.f32 \t{f_inv_rms_0}, {f_sum_sq_0};"));

    out.push(format!(
        "\tdiv.rn.f32 \t{f_sum_sq_1}, {f_sum_sq_1}, {f_hidden_f};"
    ));
    out.push(format!(
        "\tadd.f32 \t{f_sum_sq_1}, {f_sum_sq_1}, {f_epsilon};"
    ));
    out.push(format!("\trsqrt.approx.f32 \t{f_inv_rms_1}, {f_sum_sq_1};"));

    // Reload input_ptr (clobbered as temp for stride conversion)
    out.push(format!("\tld.param.u64 \t{rd_input}, [%rd1+40];"));

    out.push("\tbar.sync \t15;".into()); // avoid conflict with CUTLASS barrier 0
    out.push("\t// -- FERRITE: rms_norm prologue done --".into());
}

// ── Per-element: normalized load at each A-matrix cp.async site ──

#[allow(clippy::too_many_arguments)]
fn emit_normalized_load(
    out: &mut Vec<String>,
    smem_dst: &str,
    gmem_src: &str,
    mask: &str,
    inv_rms: &str,
    row_base: &str,
    rd_weight: &str,
    rd_weight_addr: &str,
    r_t: &dyn Fn(usize) -> String,
    f_tmp: &dyn Fn(usize) -> String,
    p_mask: &str,
) {
    out.push("\t// FERRITE: normalized A-load (rms_norm intrinsic)".into());
    out.push(format!("\tsetp.ne.b32 \t{p_mask}, {mask}, 0;"));

    // Load input[m, k:k+8]
    out.push(format!(
        "\t@{p_mask} ld.global.v4.b32 \t{{{}, {}, {}, {}}}, [{gmem_src}];",
        r_t(0),
        r_t(1),
        r_t(2),
        r_t(3)
    ));

    // Weight address: weight_ptr + (gmem_src - row_base)
    out.push(format!(
        "\tsub.s64 \t{rd_weight_addr}, {gmem_src}, {row_base};"
    ));
    out.push(format!(
        "\tadd.s64 \t{rd_weight_addr}, {rd_weight}, {rd_weight_addr};"
    ));

    // Load weight[k:k+8]
    out.push(format!(
        "\t@{p_mask} ld.global.v4.b32 \t{{{}, {}, {}, {}}}, [{rd_weight_addr}];",
        r_t(4),
        r_t(5),
        r_t(6),
        r_t(7)
    ));

    // For each b32 pair: unpack, normalize (input * weight * inv_rms), repack
    for j in 0..4 {
        let h_in0 = format!("%h_rms{}", 8 + j * 2);
        let h_in1 = format!("%h_rms{}", 8 + j * 2 + 1);
        let h_wt0 = format!("%h_rms{}", j * 2);
        let h_wt1 = format!("%h_rms{}", j * 2 + 1);

        // Unpack input
        out.push(format!(
            "\t@{p_mask} mov.b32 \t{{{h_in0}, {h_in1}}}, {};",
            r_t(j)
        ));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 \t{}, {h_in0};", f_tmp(6)));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 \t{}, {h_in1};", f_tmp(7)));

        // Unpack weight
        out.push(format!(
            "\t@{p_mask} mov.b32 \t{{{h_wt0}, {h_wt1}}}, {};",
            r_t(4 + j)
        ));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 \t{}, {h_wt0};", f_tmp(8)));
        out.push(format!("\t@{p_mask} cvt.f32.bf16 \t{}, {h_wt1};", f_tmp(9)));

        // Normalize: result = input * weight * inv_rms
        out.push(format!(
            "\t@{p_mask} mul.f32 \t{}, {}, {};",
            f_tmp(6),
            f_tmp(6),
            f_tmp(8)
        ));
        out.push(format!(
            "\t@{p_mask} mul.f32 \t{}, {}, {inv_rms};",
            f_tmp(6),
            f_tmp(6)
        ));
        out.push(format!(
            "\t@{p_mask} mul.f32 \t{}, {}, {};",
            f_tmp(7),
            f_tmp(7),
            f_tmp(9)
        ));
        out.push(format!(
            "\t@{p_mask} mul.f32 \t{}, {}, {inv_rms};",
            f_tmp(7),
            f_tmp(7)
        ));

        // Pack back: cvt.rn.bf16x2.f32 packs (high, low) into one b32
        out.push(format!(
            "\t@{p_mask} cvt.rn.bf16x2.f32 \t{}, {}, {};",
            r_t(j),
            f_tmp(7),
            f_tmp(6)
        ));
    }

    // Zero-fill for boundary tiles
    for j in 0..4 {
        out.push(format!("\t@!{p_mask} mov.b32 \t{}, 0;", r_t(j)));
    }

    // Write to CUTLASS SMEM
    out.push(format!(
        "\tst.shared.v4.b32 \t[{smem_dst}], {{{}, {}, {}, {}}};",
        r_t(0),
        r_t(1),
        r_t(2),
        r_t(3)
    ));
}

// ── Helpers ──

fn find_entry_name(lines: &[&str]) -> Result<String, String> {
    for line in lines {
        let t = line.trim();
        if t.contains(".entry") && t.contains('(') {
            let after = t.split(".entry").nth(1).ok_or("malformed .entry")?;
            let after = after.trim();
            let end = after.find('(').unwrap_or(after.len());
            return Ok(after[..end].trim().to_string());
        }
    }
    Err("no .entry found".into())
}

fn find_struct_param_name(lines: &[&str]) -> Result<String, String> {
    for line in lines {
        let t = line.trim();
        if t.contains(".param") && t.contains(".b8") && t.contains('[') {
            let parts: Vec<&str> = t.split_whitespace().collect();
            for part in &parts {
                if part.contains('[') {
                    let bracket = part.find('[').unwrap();
                    return Ok(part[..bracket].to_string());
                }
            }
        }
    }
    Err("no struct param found".into())
}
