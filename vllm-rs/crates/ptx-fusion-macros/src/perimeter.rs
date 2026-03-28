//! Perimeter replacement: rewrite a CUTLASS kernel's param interface from the
//! opaque struct layout to a flat layout of raw params (pointers, strides, dims, scalars).
//!
//! Derived fields are computed inline in PTX from the raw params using formulas
//! extracted by the build.rs probe.

use crate::parser;
use std::collections::BTreeMap;

/// A single field derivation from the probe JSON.
#[derive(Debug, Clone)]
pub struct ProbedField {
    pub offset: i64,
    pub base_value: i64,
    pub check_value: i64,
    /// Which raw param this field depends on, and the raw slope (per-perturbation-delta).
    pub depends_on: Option<(String, i64)>,
    /// Whether this is an alpha field.
    pub is_alpha: bool,
    /// Whether this is a beta field.
    pub is_beta: bool,
}

/// Parsed probe results for a CUTLASS config.
#[derive(Debug)]
pub struct ProbeResult {
    pub param_size: usize,
    pub tile: (u32, u32, u32),
    /// All 4-byte slots in the struct.
    pub fields: Vec<ProbedField>,
}

/// Formula for computing a struct field from flat params.
#[derive(Debug, Clone)]
pub enum FieldFormula {
    /// Load directly from flat params at given offset.
    Direct { flat_offset: usize },
    /// Multiply a flat param by a constant: value = param * slope.
    MulConst { flat_offset: usize, slope: i64 },
    /// Multiply + add: value = param * slope + intercept.
    MulAdd {
        flat_offset: usize,
        slope: i64,
        intercept: i64,
    },
    /// Constant value (not dependent on any raw param).
    Constant { value: i64 },
    /// Ceiling division: value = (param + divisor - 1) / divisor.
    CeilDiv { flat_offset: usize, divisor: u32 },
    /// Float field (alpha or beta). Load f32 from flat offset.
    Float { flat_offset: usize },
    /// Swizzle log: computed from ceil(N/tile_N) using threshold logic.
    /// GemmIdentityThreadblockSwizzle<4>: if grid_n >= 3 → 2, elif >= 2 → 1, else 0.
    SwizzleLog { n_flat_offset: usize, tile_n: u32 },
    /// Always zero (nullptr fields).
    Zero,
}

/// A raw param in the flat layout.
#[derive(Debug, Clone)]
pub struct FlatParam {
    pub name: &'static str,
    pub ptx_type: &'static str,
    pub offset: usize,
    pub size: usize,
}

/// The flat param layout for a CUTLASS bf16 GEMM.
pub const FLAT_PARAMS: &[FlatParam] = &[
    FlatParam {
        name: "ptr_A",
        ptx_type: ".u64",
        offset: 0,
        size: 8,
    },
    FlatParam {
        name: "ptr_B",
        ptx_type: ".u64",
        offset: 8,
        size: 8,
    },
    FlatParam {
        name: "ptr_C",
        ptx_type: ".u64",
        offset: 16,
        size: 8,
    },
    FlatParam {
        name: "ptr_D",
        ptx_type: ".u64",
        offset: 24,
        size: 8,
    },
    FlatParam {
        name: "lda",
        ptx_type: ".u64",
        offset: 32,
        size: 8,
    },
    FlatParam {
        name: "ldb",
        ptx_type: ".u64",
        offset: 40,
        size: 8,
    },
    FlatParam {
        name: "ldc",
        ptx_type: ".u64",
        offset: 48,
        size: 8,
    },
    FlatParam {
        name: "ldd",
        ptx_type: ".u64",
        offset: 56,
        size: 8,
    },
    FlatParam {
        name: "M",
        ptx_type: ".s32",
        offset: 64,
        size: 4,
    },
    FlatParam {
        name: "N",
        ptx_type: ".s32",
        offset: 68,
        size: 4,
    },
    FlatParam {
        name: "K",
        ptx_type: ".s32",
        offset: 72,
        size: 4,
    },
    FlatParam {
        name: "alpha",
        ptx_type: ".f32",
        offset: 76,
        size: 4,
    },
    FlatParam {
        name: "beta",
        ptx_type: ".f32",
        offset: 80,
        size: 4,
    },
    // Pad to 8-byte alignment
];

pub const FLAT_PARAM_SIZE: usize = 88; // 84 rounded up to 8

/// Parse a probe derivations JSON string.
pub fn parse_derivations(json: &str) -> Result<ProbeResult, String> {
    // Minimal JSON parser for our specific format (no serde dependency in proc macro)
    let param_size = extract_json_int(json, "param_size").ok_or("missing param_size")? as usize;

    let tile_arr = extract_json_array_ints(json, "tile").ok_or("missing tile")?;
    if tile_arr.len() != 3 {
        return Err("tile must have 3 elements".into());
    }
    let tile = (tile_arr[0] as u32, tile_arr[1] as u32, tile_arr[2] as u32);

    let fields = parse_fields_array(json)?;

    Ok(ProbeResult {
        param_size,
        tile,
        fields,
    })
}

/// Build the formula map: for each struct offset that has an ld.param,
/// determine how to compute it from flat params.
///
/// `param_offsets` is the set of struct offsets loaded by ld.param in the kernel
/// (from Phase 1's extract_param_fields).
pub fn build_formula_map(
    probe: &ProbeResult,
    param_offsets: &[i64],
) -> BTreeMap<i64, FieldFormula> {
    let (tile_m, tile_n, tile_k) = probe.tile;
    let deltas: BTreeMap<&str, i64> = [
        ("M", tile_m as i64),
        ("N", tile_n as i64),
        ("K", tile_k as i64),
        ("lda", 1),
        ("ldb", 1),
        ("ldc", 1),
        ("ldd", 1),
    ]
    .into();

    // Build lookup: struct offset → probed field (for low 32 bits)
    let probe_map: BTreeMap<i64, &ProbedField> =
        probe.fields.iter().map(|f| (f.offset, f)).collect();

    // Map raw param names to flat layout offsets
    let flat_offset_of: BTreeMap<&str, usize> =
        FLAT_PARAMS.iter().map(|p| (p.name, p.offset)).collect();

    // Pointer offsets from the probe (constant fields = 0x1000, 0x2000, 0x3000, 0x4000)
    let ptr_values: [(i64, &str); 4] = [
        (0x1000, "ptr_A"),
        (0x2000, "ptr_B"),
        (0x3000, "ptr_C"),
        (0x4000, "ptr_D"),
    ];

    let mut formulas = BTreeMap::new();

    for &struct_off in param_offsets {
        let probed = probe_map.get(&struct_off);

        // Check if this is a pointer (constant matching a dummy ptr)
        if let Some(pf) = probed
            && let Some((_, flat_name)) = ptr_values
                .iter()
                .find(|(val, _)| pf.base_value == *val && pf.check_value == *val)
        {
            formulas.insert(
                struct_off,
                FieldFormula::Direct {
                    flat_offset: flat_offset_of[flat_name],
                },
            );
            continue;
        }

        // Check the high-word partner (for u64 fields loaded at this offset,
        // the probe has entries at struct_off and struct_off+4)
        let probed_hi = probe_map.get(&(struct_off + 4));

        // Alpha/beta
        if let Some(pf) = probed {
            if pf.is_alpha {
                formulas.insert(
                    struct_off,
                    FieldFormula::Float {
                        flat_offset: flat_offset_of["alpha"],
                    },
                );
                continue;
            }
            if pf.is_beta {
                formulas.insert(
                    struct_off,
                    FieldFormula::Float {
                        flat_offset: flat_offset_of["beta"],
                    },
                );
                continue;
            }
        }

        // Dependency-based formula
        if let Some(pf) = probed
            && let Some((ref param, raw_slope)) = pf.depends_on
        {
            let delta = deltas.get(param.as_str()).copied().unwrap_or(1);
            let normalized_slope = raw_slope as f64 / delta as f64;

            // Compute intercept
            let base_raw: BTreeMap<&str, i64> = [
                ("M", 256),
                ("N", 512),
                ("K", 128),
                ("lda", 128),
                ("ldb", 128),
                ("ldc", 512),
                ("ldd", 512),
            ]
            .into();
            let base_raw_val = base_raw.get(param.as_str()).copied().unwrap_or(0);
            let intercept = pf.base_value - (normalized_slope * base_raw_val as f64) as i64;

            let flat_off = flat_offset_of
                .get(param.as_str())
                .copied()
                .unwrap_or(flat_offset_of["M"]); // fallback

            let slope_i = normalized_slope as i64;

            // Check for ceiling division pattern (M/tile_M, N/tile_N, K/tile_K)
            if (param == "M" || param == "N" || param == "K")
                && normalized_slope.fract().abs() > 0.0001
                && intercept == 0
            {
                // This is ceil(param / tile_dim)
                let divisor = (1.0 / normalized_slope).round() as u32;
                formulas.insert(
                    struct_off,
                    FieldFormula::CeilDiv {
                        flat_offset: flat_off,
                        divisor,
                    },
                );
                continue;
            }

            if slope_i == 1 && intercept == 0 {
                formulas.insert(
                    struct_off,
                    FieldFormula::Direct {
                        flat_offset: flat_off,
                    },
                );
            } else if intercept == 0 {
                formulas.insert(
                    struct_off,
                    FieldFormula::MulConst {
                        flat_offset: flat_off,
                        slope: slope_i,
                    },
                );
            } else {
                formulas.insert(
                    struct_off,
                    FieldFormula::MulAdd {
                        flat_offset: flat_off,
                        slope: slope_i,
                        intercept,
                    },
                );
            }
            continue;
        }

        // Null pointers (Phase 1 classified as Pointer but probe shows 0)
        if let Some(pf) = probed
            && pf.base_value == 0
            && pf.check_value == 0
        {
            // Check the high word too
            if let Some(hi) = probed_hi
                && hi.base_value == 0
                && hi.check_value == 0
            {
                formulas.insert(struct_off, FieldFormula::Zero);
                continue;
            }
            formulas.insert(struct_off, FieldFormula::Zero);
            continue;
        }

        // Swizzle log: offset 24 in the CUTLASS params struct.
        // This is NOT truly constant — it depends on ceil(N/tile_N).
        // Compute it inline instead of baking the probed value.
        if struct_off == 24 {
            formulas.insert(
                struct_off,
                FieldFormula::SwizzleLog {
                    n_flat_offset: flat_offset_of["N"],
                    tile_n: probe.tile.1,
                },
            );
            continue;
        }

        // Constant (non-zero, doesn't change between base and check)
        if let Some(pf) = probed
            && pf.base_value == pf.check_value
            && pf.depends_on.is_none()
        {
            formulas.insert(
                struct_off,
                FieldFormula::Constant {
                    value: pf.base_value,
                },
            );
            continue;
        }

        // Fallback: constant from base value
        if let Some(pf) = probed {
            formulas.insert(
                struct_off,
                FieldFormula::Constant {
                    value: pf.base_value,
                },
            );
        } else {
            formulas.insert(struct_off, FieldFormula::Zero);
        }
    }

    formulas
}

/// Generate PTX instructions that compute a derived value and store it into the
/// destination register, replacing a single `ld.param` instruction.
///
/// `dest_reg` is the register the original ld.param wrote to (e.g., "%rd2").
/// `ptx_type` is the load type (e.g., "u64", "u32", "f32").
/// `formula` is how to compute the value from flat params.
/// `flat_param_name` is the PTX name of the flat param (e.g., "ferrite_params").
/// `tmp_counter` is used to generate unique temp register names.
///   Temp 64-bit: %rd{rd_base + counter}, temp 32-bit: %r{r_base + counter}
pub fn emit_replacement(
    dest_reg: &str,
    ptx_type: &str,
    formula: &FieldFormula,
    flat_param_name: &str,
    tmp_counter: &mut usize,
) -> String {
    // Helper to get a temp 64-bit register name
    let rd_tmp = |ctr: &mut usize| -> String {
        let name = format!("%rd_ptmp{}", *ctr);
        *ctr += 1;
        name
    };
    // Helper to get a temp 32-bit register name
    let r_tmp = |ctr: &mut usize| -> String {
        let name = format!("%r_ptmp{}", *ctr);
        *ctr += 1;
        name
    };

    match formula {
        FieldFormula::Direct { flat_offset } => {
            format!("\tld.param.{ptx_type} \t{dest_reg}, [{flat_param_name}+{flat_offset}];",)
        }

        FieldFormula::Float { flat_offset } => {
            format!("\tld.param.f32 \t{dest_reg}, [{flat_param_name}+{flat_offset}];",)
        }

        FieldFormula::Constant { value } => {
            if ptx_type.contains("64") {
                format!("\tmov.u64 \t{dest_reg}, {value};")
            } else if ptx_type == "f32" {
                let tmp = r_tmp(tmp_counter);
                format!("\tmov.b32 \t{tmp}, {value};\n\tmov.b32 \t{dest_reg}, {tmp};")
            } else {
                format!("\tmov.u32 \t{dest_reg}, {value};")
            }
        }

        FieldFormula::Zero => {
            if ptx_type.contains("64") {
                format!("\tmov.u64 \t{dest_reg}, 0;")
            } else {
                format!("\tmov.u32 \t{dest_reg}, 0;")
            }
        }

        FieldFormula::MulConst { flat_offset, slope } => {
            if ptx_type.contains("64") {
                let tmp = rd_tmp(tmp_counter);
                if *slope > 0 && slope.count_ones() == 1 {
                    let shift = slope.trailing_zeros();
                    format!(
                        "\tld.param.u64 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                         \tshl.b64 \t{dest_reg}, {tmp}, {shift};"
                    )
                } else if *slope < 0 && (-slope).count_ones() == 1 {
                    let shift = (-slope).trailing_zeros();
                    let tmp2 = rd_tmp(tmp_counter);
                    format!(
                        "\tld.param.u64 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                         \tshl.b64 \t{tmp2}, {tmp}, {shift};\n\
                         \tneg.s64 \t{dest_reg}, {tmp2};"
                    )
                } else {
                    format!(
                        "\tld.param.u64 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                         \tmul.lo.s64 \t{dest_reg}, {tmp}, {slope};"
                    )
                }
            } else {
                let tmp = r_tmp(tmp_counter);
                format!(
                    "\tld.param.u32 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                     \tmul.lo.s32 \t{dest_reg}, {tmp}, {slope};"
                )
            }
        }

        FieldFormula::MulAdd {
            flat_offset,
            slope,
            intercept,
        } => {
            if ptx_type.contains("64") {
                let tmp = rd_tmp(tmp_counter);
                let tmp2 = rd_tmp(tmp_counter);
                format!(
                    "\tld.param.u64 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                     \tmul.lo.s64 \t{tmp2}, {tmp}, {slope};\n\
                     \tadd.s64 \t{dest_reg}, {tmp2}, {intercept};"
                )
            } else {
                let tmp = r_tmp(tmp_counter);
                let tmp2 = r_tmp(tmp_counter);
                format!(
                    "\tld.param.u32 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                     \tmul.lo.s32 \t{tmp2}, {tmp}, {slope};\n\
                     \tadd.s32 \t{dest_reg}, {tmp2}, {intercept};"
                )
            }
        }

        FieldFormula::CeilDiv {
            flat_offset,
            divisor,
        } => {
            let tmp = r_tmp(tmp_counter);
            let tmp2 = r_tmp(tmp_counter);
            let d_minus_1 = *divisor as i32 - 1;

            if divisor.count_ones() == 1 {
                let shift = divisor.trailing_zeros();
                format!(
                    "\tld.param.s32 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                     \tadd.s32 \t{tmp2}, {tmp}, {d_minus_1};\n\
                     \tshr.s32 \t{dest_reg}, {tmp2}, {shift};"
                )
            } else {
                format!(
                    "\tld.param.s32 \t{tmp}, [{flat_param_name}+{flat_offset}];\n\
                     \tadd.s32 \t{tmp2}, {tmp}, {d_minus_1};\n\
                     \tdiv.s32 \t{dest_reg}, {tmp2}, {divisor};"
                )
            }
        }

        FieldFormula::SwizzleLog {
            n_flat_offset,
            tile_n,
        } => {
            // Compute: grid_n = ceil(N / tile_n)
            // Then: if grid_n >= 3 → 2, elif grid_n >= 2 → 1, else 0
            // (GemmIdentityThreadblockSwizzle<4>, SWIZZLE_N=4)
            let t_n = r_tmp(tmp_counter);
            let t_gn = r_tmp(tmp_counter);
            let t_p1 = format!("%p_ptmp{}", *tmp_counter);
            *tmp_counter += 1;
            let t_p2 = format!("%p_ptmp{}", *tmp_counter);
            *tmp_counter += 1;

            let d_minus_1 = tile_n - 1;
            let shift = tile_n.trailing_zeros();

            // grid_n = ceil(N / tile_n)
            let ceil_div = if tile_n.count_ones() == 1 {
                format!(
                    "\tld.param.s32 \t{t_n}, [{flat_param_name}+{n_flat_offset}];\n\
                     \tadd.s32 \t{t_gn}, {t_n}, {d_minus_1};\n\
                     \tshr.s32 \t{t_gn}, {t_gn}, {shift};"
                )
            } else {
                format!(
                    "\tld.param.s32 \t{t_n}, [{flat_param_name}+{n_flat_offset}];\n\
                     \tadd.s32 \t{t_gn}, {t_n}, {d_minus_1};\n\
                     \tdiv.s32 \t{t_gn}, {t_gn}, {tile_n};"
                )
            };
            // if grid_n >= 3 → 2, elif grid_n >= 2 → 1, else 0
            format!(
                "{ceil_div}\n\
                 \tsetp.ge.s32 \t{t_p1}, {t_gn}, 3;\n\
                 \tsetp.ge.s32 \t{t_p2}, {t_gn}, 2;\n\
                 \tselp.b32 \t{dest_reg}, 1, 0, {t_p2};\n\
                 \tselp.b32 \t{dest_reg}, 2, {dest_reg}, {t_p1};"
            )
        }
    }
}

// ── The PTX rewriter ──

const FLAT_PARAM_NAME: &str = "ferrite_params";

/// Rewrite a CUTLASS kernel's PTX to use the flat param layout.
///
/// This is the main entry point for Phase 2 perimeter replacement.
/// It parses the derivations, builds formulas for each ld.param site,
/// and rewrites the PTX text.
///
/// Returns the rewritten PTX string and the new entry name.
pub fn replace_perimeter(
    ptx_source: &str,
    derivations_json: &str,
    entry_name: &str,
) -> Result<(String, String), String> {
    let probe = parse_derivations(derivations_json)?;

    // Phase 1: extract param fields (offsets and registers)
    let lines: Vec<&str> = ptx_source.lines().collect();
    let (param_fields, base_offsets) = parser::extract_param_fields(&lines);
    let param_offsets: Vec<i64> = param_fields.iter().map(|f| f.offset).collect();

    // Build the formula map
    let formulas = build_formula_map(&probe, &param_offsets);

    // Find the mangled param name (the .param .b8 NAME[SIZE])
    let mangled_param =
        find_mangled_param_name(ptx_source).ok_or("cannot find .param .b8 declaration")?;

    // Dry run to count temp registers needed
    let mut dry_tmp_counter = 0usize;
    for line in &lines {
        let trimmed = line.trim();
        let work = if trimmed.starts_with('@') {
            trimmed
                .find(|c: char| c.is_whitespace())
                .map_or(trimmed, |i| trimmed[i..].trim())
        } else {
            trimmed
        };
        if work.starts_with("ld.param")
            && let Some(abs_off) = extract_ld_param_offset(work, &mangled_param, &base_offsets)
            && let Some(formula) = formulas.get(&abs_off)
        {
            let _ = emit_replacement("%dummy", "u64", formula, "x", &mut dry_tmp_counter);
        }
    }
    let extra_regs = dry_tmp_counter; // upper bound for both rd and r temps

    // Rewrite the PTX line by line
    let mut output = Vec::new();
    let mut tmp_counter = 0usize;
    let mut in_entry_header = false;
    let mut skip_until_body = false;

    for (_line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Replace the .entry line (contains ".entry" and "(")
        if trimmed.contains(".entry") && trimmed.contains('(') {
            output.push(format!(".visible .entry {entry_name}("));
            in_entry_header = true;
            continue;
        }

        // Replace the struct .param declaration (inside entry header)
        // Only replace the .b8 NAME[SIZE] struct param, not scalar params
        if in_entry_header && trimmed.contains(".param") {
            if trimmed.contains(".b8") && trimmed.contains('[') {
                // This is the struct param — replace with flat layout
                output.push(format!(
                    "\t.param .align 8 .b8 {FLAT_PARAM_NAME}[{FLAT_PARAM_SIZE}]"
                ));
            } else {
                // This is a scalar param (e.g., from rms_norm intrinsic) — pass through
                output.push(line.to_string());
            }
            continue;
        }

        // Close the param block
        if in_entry_header && (trimmed == ")" || trimmed == "){") {
            output.push(")".to_string());
            in_entry_header = false;
            skip_until_body = true;
            continue;
        }

        if in_entry_header {
            // Skip any other lines in the entry header
            continue;
        }

        // Skip the opening brace
        if skip_until_body && trimmed == "{" {
            output.push("{".to_string());
            skip_until_body = false;
            continue;
        }

        // After the last .reg declaration, add temp register declarations
        if trimmed.starts_with(".reg") && trimmed.contains('%') {
            output.push(line.to_string());
            // Check if the NEXT non-empty line is NOT a .reg → we've seen the last .reg
            let remaining: Vec<&&str> = lines[_line_idx + 1..]
                .iter()
                .filter(|l| !l.trim().is_empty())
                .take(1)
                .collect();
            if let Some(next) = remaining.first()
                && !next.trim().starts_with(".reg")
                && extra_regs > 0
            {
                // Add temp register declarations
                output.push(format!("\t.reg .b64 \t%rd_ptmp<{extra_regs}>;"));
                output.push(format!("\t.reg .b32 \t%r_ptmp<{extra_regs}>;"));
                output.push(format!("\t.reg .pred \t%p_ptmp<{extra_regs}>;"));
            }
            continue;
        }

        // Skip the `mov.b64 %rdN, MANGLED_PARAM;` lines (get param base address)
        if (trimmed.starts_with("mov.b64") || trimmed.starts_with("mov.u64"))
            && trimmed.contains(&mangled_param)
        {
            output.push(format!("\t// [ferrite] removed: {}", trimmed));
            continue;
        }

        // Skip the `add.s64 %rdN, %rdM, 24;` that compute base offsets from param address.
        // These are no longer needed since we address flat params directly.
        // We detect these by checking if the dest register is a known base register
        // and the immediate is a known base offset.
        if trimmed.starts_with("add.s64") {
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 4 {
                let dst = parts[1].trim_end_matches(',');
                let imm = parts[3].trim_end_matches(';');
                if base_offsets.contains_key(dst)
                    && imm.parse::<i64>().ok() == Some(base_offsets[dst])
                {
                    // Check that the source is a known base-0 register
                    let src = parts[2].trim_end_matches(',');
                    if base_offsets.get(src) == Some(&0) {
                        output.push(format!("\t// [ferrite] removed: {}", trimmed));
                        continue;
                    }
                }
            }
        }

        // Replace ld.param instructions
        if (trimmed.starts_with("ld.param")
            || (trimmed.starts_with("@") && trimmed.contains("ld.param")))
            && let Some(replacement) = try_replace_ld_param(
                trimmed,
                &mangled_param,
                &base_offsets,
                &formulas,
                &param_fields,
                &mut tmp_counter,
            )
        {
            output.push(replacement);
            continue;
        }

        // Also check for `.pragma "used_bytes_mask` which relates to param loads
        // — keep as-is (it's a hint, not executable)

        // Pass through all other lines unchanged
        output.push(line.to_string());
    }

    let new_entry = entry_name.to_string();
    Ok((output.join("\n"), new_entry))
}

/// Try to replace a single ld.param instruction.
/// Returns Some(replacement_lines) if this is a param struct load, None otherwise.
fn try_replace_ld_param(
    line: &str,
    mangled_param: &str,
    base_offsets: &BTreeMap<String, i64>,
    formulas: &BTreeMap<i64, FieldFormula>,
    _param_fields: &[parser::ParamField],
    tmp_counter: &mut usize,
) -> Option<String> {
    // Extract: predicate, opcode, dest register(s), bracket content
    let work = line.trim();

    // Handle predication
    let (pred_prefix, rest) = if work.starts_with('@') {
        let space = work.find(|c: char| c.is_whitespace())?;
        (Some(&work[..space]), work[space..].trim())
    } else {
        (None, work)
    };

    if !rest.starts_with("ld.param") {
        return None;
    }

    // Extract PTX type from opcode
    let opcode = rest.split_whitespace().next()?;
    let dot_parts: Vec<&str> = opcode.split('.').collect();
    let ptx_type = *dot_parts.last()?;
    let is_vector = opcode.contains(".v2.") || opcode.contains(".v4.");

    // Extract dest register(s)
    let after_opcode = rest[opcode.len()..].trim();
    let comma_pos = after_opcode.find(',')?;
    let dest_part = after_opcode[..comma_pos].trim();

    // Extract bracket content
    let bracket_start = after_opcode.find('[')?;
    let bracket_end = after_opcode.find(']')?;
    let bracket_content = &after_opcode[bracket_start + 1..bracket_end];

    // Resolve to absolute struct offset
    let abs_offset = resolve_bracket_offset(bracket_content, mangled_param, base_offsets)?;

    // Look up formula
    let formula = formulas.get(&abs_offset)?;

    // For vector loads ({%r164, %r165}), we need to emit separate replacements
    // for each element
    if is_vector && dest_part.contains('{') {
        let regs: Vec<&str> = dest_part
            .trim_matches(|c| c == '{' || c == '}')
            .split(',')
            .map(|s| s.trim())
            .collect();

        let elem_size: i64 = match ptx_type {
            "u64" | "s64" | "b64" | "f64" => 8,
            "u32" | "s32" | "b32" | "f32" => 4,
            "u16" | "s16" | "b16" | "f16" => 2,
            _ => 4,
        };

        let mut parts = Vec::new();
        for (i, reg) in regs.iter().enumerate() {
            let elem_offset = abs_offset + i as i64 * elem_size;
            if let Some(f) = formulas.get(&elem_offset) {
                let code = emit_replacement(reg, ptx_type, f, FLAT_PARAM_NAME, tmp_counter);
                parts.push(code);
            } else {
                // Fallback: try the formula for the base offset
                let code = emit_replacement(reg, ptx_type, formula, FLAT_PARAM_NAME, tmp_counter);
                parts.push(code);
            }
        }

        let replacement = parts.join("\n");
        return Some(if let Some(_pred) = pred_prefix {
            format!(
                "\t// [ferrite] replaced vector ld.param at offset {abs_offset}\n\t// TODO: predicated vector replacement\n{replacement}"
            )
        } else {
            format!("\t// [ferrite] replaced vector ld.param at offset {abs_offset}\n{replacement}")
        });
    }

    // Single register load
    let dest_reg = dest_part.trim();
    let code = emit_replacement(dest_reg, ptx_type, formula, FLAT_PARAM_NAME, tmp_counter);

    Some(if let Some(pred) = pred_prefix {
        // For predicated loads, wrap the replacement in the same predicate
        // This is tricky — for multi-instruction replacements, each instruction needs the predicate
        let predicated = code
            .lines()
            .map(|l| {
                let l = l.trim();
                if l.starts_with("//") || l.is_empty() {
                    l.to_string()
                } else {
                    format!("\t{pred} {l}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "\t// [ferrite] replaced ld.param at offset {abs_offset} (predicated)\n{predicated}"
        )
    } else {
        format!("\t// [ferrite] replaced ld.param at offset {abs_offset}\n{code}")
    })
}

/// Resolve a bracket expression like "[%rd1+16]" or "[MANGLED_NAME+24]" to an
/// absolute byte offset in the param struct.
fn resolve_bracket_offset(
    bracket_content: &str,
    mangled_param: &str,
    base_offsets: &BTreeMap<String, i64>,
) -> Option<i64> {
    if bracket_content.contains('+') || bracket_content.contains('-') {
        let (base_part, offset_str) = if let Some(plus_pos) = bracket_content.rfind('+') {
            (
                bracket_content[..plus_pos].trim(),
                bracket_content[plus_pos + 1..].trim(),
            )
        } else if let Some(minus_pos) = bracket_content.rfind('-') {
            if minus_pos > 0 {
                (
                    bracket_content[..minus_pos].trim(),
                    bracket_content[minus_pos..].trim(),
                )
            } else {
                return None;
            }
        } else {
            return None;
        };

        let field_offset: i64 = offset_str.parse().ok()?;

        if base_part.starts_with('%') {
            // Indirect: [%rdBase+OFFSET]
            let base_off = base_offsets.get(base_part).copied().unwrap_or(0);
            Some(base_off + field_offset)
        } else if base_part.contains(mangled_param) || base_part.starts_with('_') {
            // Direct: [param_name+OFFSET]
            Some(field_offset)
        } else {
            None
        }
    } else if bracket_content.starts_with('%') {
        base_offsets.get(bracket_content).copied()
    } else if bracket_content.contains(mangled_param) || bracket_content == mangled_param {
        // Bare struct param name → offset 0
        Some(0)
    } else {
        // Unknown param name (e.g., extra named params from fusion) — don't touch
        None
    }
}

/// Parse a .reg declaration line to extract the register prefix and count.
/// e.g., ".reg .b64 \t%rd<144>;" → ("rd", 144)
fn parse_reg_decl(line: &str) -> Option<(&str, usize)> {
    let trimmed = line.trim();
    if !trimmed.starts_with(".reg") {
        return None;
    }
    // Find the %prefix<count> pattern
    let pct = trimmed.find('%')?;
    let angle_start = trimmed[pct..].find('<')?;
    let angle_end = trimmed[pct..].find('>')?;
    let prefix = &trimmed[pct + 1..pct + angle_start];
    let count_str = &trimmed[pct + angle_start + 1..pct + angle_end];
    let count = count_str.parse().ok()?;
    Some((prefix, count))
}

/// Extract the absolute struct offset from an ld.param instruction line.
fn extract_ld_param_offset(
    line: &str,
    mangled_param: &str,
    base_offsets: &BTreeMap<String, i64>,
) -> Option<i64> {
    let bracket_start = line.find('[')?;
    let bracket_end = line.find(']')?;
    let bracket_content = &line[bracket_start + 1..bracket_end];
    resolve_bracket_offset(bracket_content, mangled_param, base_offsets)
}

/// Find the mangled param name from the .param declaration.
/// Returns the shortest unique substring that identifies it.
fn find_mangled_param_name(ptx: &str) -> Option<String> {
    for line in ptx.lines() {
        let trimmed = line.trim();
        if trimmed.contains(".param") && trimmed.contains(".b8") && trimmed.contains('[') {
            // Extract the name: .param .align 8 .b8 NAME[SIZE]
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            for part in &parts {
                if part.contains('[') {
                    let bracket = part.find('[')?;
                    return Some(part[..bracket].to_string());
                }
            }
        }
    }
    None
}

// ── Minimal JSON parsing (no serde in proc macro crate) ──

fn extract_json_int(json: &str, key: &str) -> Option<i64> {
    let pattern = format!("\"{}\":", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    after[..end].parse().ok()
}

fn extract_json_array_ints(json: &str, key: &str) -> Option<Vec<i64>> {
    let pattern = format!("\"{}\":", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let bracket_start = after.find('[')?;
    let bracket_end = after.find(']')?;
    let inner = &after[bracket_start + 1..bracket_end];
    Some(
        inner
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
    )
}

fn parse_fields_array(json: &str) -> Result<Vec<ProbedField>, String> {
    let fields_start = json.find("\"fields\":").ok_or("no fields key")?;
    let after = &json[fields_start..];
    let arr_start = after.find('[').ok_or("no fields array")? + fields_start;
    let arr_end = json.rfind(']').ok_or("no fields array end")?;
    let arr_content = &json[arr_start + 1..arr_end];

    let mut fields = Vec::new();
    let mut depth = 0;
    let mut obj_start = None;

    for (i, c) in arr_content.char_indices() {
        match c {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start {
                        let obj_str = &arr_content[start..=i];
                        if let Some(field) = parse_single_field(obj_str) {
                            fields.push(field);
                        }
                    }
                    obj_start = None;
                }
            }
            _ => {}
        }
    }

    Ok(fields)
}

/// Count how many temporary registers the formula map will need,
/// so we can declare them in the PTX register block.
pub fn count_temp_registers(formulas: &BTreeMap<i64, FieldFormula>) -> (usize, usize) {
    let mut rd_count = 0usize; // 64-bit temps
    let mut r_count = 0usize; // 32-bit temps
    for formula in formulas.values() {
        match formula {
            FieldFormula::MulConst { slope, .. } => {
                rd_count += 1; // load tmp
                if *slope < 0 && (-slope).count_ones() == 1 {
                    rd_count += 1; // intermediate for neg
                }
            }
            FieldFormula::MulAdd { .. } => {
                rd_count += 2; // load tmp + mul result
            }
            FieldFormula::CeilDiv { .. } => {
                r_count += 2; // load tmp + add result
            }
            FieldFormula::Constant { .. } => {
                // May need a u32 tmp for f32 bit-cast, but rare
            }
            _ => {}
        }
    }
    (rd_count, r_count)
}

fn parse_single_field(obj: &str) -> Option<ProbedField> {
    let offset = extract_json_int(obj, "offset")?;
    let base_value = extract_json_int(obj, "base_value").unwrap_or(0);
    let check_value = extract_json_int(obj, "check_value").unwrap_or(0);

    let is_alpha = obj.contains("\"alpha_f32\"");
    let is_beta = obj.contains("\"beta_f32\"");

    // Parse depends_on: {"param": slope} — we only handle single-dependency
    let depends_on = if let Some(dep_start) = obj.find("\"depends_on\":") {
        let after = &obj[dep_start..];
        if let Some(brace_start) = after.find('{') {
            let brace_end = after[brace_start..].find('}')?;
            let inner = &after[brace_start + 1..brace_start + brace_end];
            // Parse first key-value pair: "param": slope
            let colon = inner.find(':')?;
            let key = inner[..colon].trim().trim_matches('"').to_string();
            let val_str = inner[colon + 1..].trim().trim_matches(',');
            // If multiple deps, take just the first
            let val_end = val_str
                .find(|c: char| !c.is_ascii_digit() && c != '-')
                .unwrap_or(val_str.len());
            let val: i64 = val_str[..val_end].parse().ok()?;
            Some((key, val))
        } else {
            None
        }
    } else {
        None
    };

    Some(ProbedField {
        offset,
        base_value,
        check_value,
        depends_on,
        is_alpha,
        is_beta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_64x64x32_derivations() -> ProbeResult {
        let json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");
        parse_derivations(json).unwrap()
    }

    #[test]
    fn parse_derivations_basic() {
        let probe = load_64x64x32_derivations();
        assert_eq!(probe.param_size, 368);
        assert_eq!(probe.tile, (64, 64, 32));
        assert_eq!(probe.fields.len(), 92); // 368 / 4
    }

    #[test]
    fn parse_field_dependencies() {
        let probe = load_64x64x32_derivations();

        // offset 0 = M (depends on M with slope 64 per perturbation of +64)
        let f0 = probe.fields.iter().find(|f| f.offset == 0).unwrap();
        assert_eq!(f0.base_value, 256); // base M
        let (param, slope) = f0.depends_on.as_ref().unwrap();
        assert_eq!(param, "M");
        assert_eq!(*slope, 64); // raw slope per M perturbation of +64

        // offset 40 = lda * 16 (depends on lda with slope 16 per perturbation of +1)
        let f40 = probe.fields.iter().find(|f| f.offset == 40).unwrap();
        assert_eq!(f40.base_value, 2048); // 128 * 16
        let (param, slope) = f40.depends_on.as_ref().unwrap();
        assert_eq!(param, "lda");
        assert_eq!(*slope, 16);
    }

    #[test]
    fn parse_alpha_beta() {
        let probe = load_64x64x32_derivations();

        let f288 = probe.fields.iter().find(|f| f.offset == 288).unwrap();
        assert!(f288.is_alpha);
        assert!(!f288.is_beta);

        let f292 = probe.fields.iter().find(|f| f.offset == 292).unwrap();
        assert!(f292.is_beta);
        assert!(!f292.is_alpha);
    }

    #[test]
    fn build_formulas_pointers() {
        let probe = load_64x64x32_derivations();
        // The actual ld.param offsets from the CUTLASS kernel
        let offsets: Vec<i64> = vec![
            0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 88, 96, 104, 112, 128, 136, 160, 192, 208,
            216, 240, 272, 288, 292, 336,
        ];
        let formulas = build_formula_map(&probe, &offsets);

        // Pointer A at struct offset 64 → Direct load from flat ptr_A
        match &formulas[&64] {
            FieldFormula::Direct { flat_offset } => {
                assert_eq!(*flat_offset, 0, "ptr_A should be at flat offset 0");
            }
            other => panic!("offset 64 should be Direct, got {:?}", other),
        }

        // Pointer B at struct offset 112 → Direct load from flat ptr_B
        match &formulas[&112] {
            FieldFormula::Direct { flat_offset } => {
                assert_eq!(*flat_offset, 8, "ptr_B should be at flat offset 8");
            }
            other => panic!("offset 112 should be Direct, got {:?}", other),
        }
    }

    #[test]
    fn build_formulas_strides() {
        let probe = load_64x64x32_derivations();
        let offsets: Vec<i64> = vec![32, 40, 48, 80, 88, 96, 128, 208];
        let formulas = build_formula_map(&probe, &offsets);

        // lda at struct offset 32 → Direct (slope=1)
        match &formulas[&32] {
            FieldFormula::Direct { flat_offset } => {
                assert_eq!(*flat_offset, 32, "lda at flat offset 32");
            }
            other => panic!("offset 32 should be Direct, got {:?}", other),
        }

        // lda*16 at struct offset 40 → MulConst
        match &formulas[&40] {
            FieldFormula::MulConst { flat_offset, slope } => {
                assert_eq!(*flat_offset, 32); // lda
                assert_eq!(*slope, 16);
            }
            other => panic!("offset 40 should be MulConst, got {:?}", other),
        }
    }

    #[test]
    fn build_formulas_dimensions() {
        let probe = load_64x64x32_derivations();
        let offsets: Vec<i64> = vec![0, 4, 8, 12, 16];
        let formulas = build_formula_map(&probe, &offsets);

        // M at offset 0 → Direct (slope=1 per unit)
        match &formulas[&0] {
            FieldFormula::Direct { flat_offset } => {
                assert_eq!(*flat_offset, 64, "M at flat offset 64");
            }
            other => panic!("offset 0 should be Direct (M), got {:?}", other),
        }

        // grid_tiled_shape.m at offset 12 → CeilDiv(M, 64)
        match &formulas[&12] {
            FieldFormula::CeilDiv {
                flat_offset,
                divisor,
            } => {
                assert_eq!(*flat_offset, 64); // M
                assert_eq!(*divisor, 64);
            }
            other => panic!("offset 12 should be CeilDiv, got {:?}", other),
        }
    }

    #[test]
    fn build_formulas_scalars() {
        let probe = load_64x64x32_derivations();
        let offsets: Vec<i64> = vec![288, 292];
        let formulas = build_formula_map(&probe, &offsets);

        match &formulas[&288] {
            FieldFormula::Float { flat_offset } => {
                assert_eq!(*flat_offset, 76, "alpha at flat offset 76");
            }
            other => panic!("offset 288 should be Float (alpha), got {:?}", other),
        }

        match &formulas[&292] {
            FieldFormula::Float { flat_offset } => {
                assert_eq!(*flat_offset, 80, "beta at flat offset 80");
            }
            other => panic!("offset 292 should be Float (beta), got {:?}", other),
        }
    }

    #[test]
    fn emit_direct_load() {
        let mut ctr = 0;
        let code = emit_replacement(
            "%rd7",
            "u64",
            &FieldFormula::Direct { flat_offset: 0 },
            "fp",
            &mut ctr,
        );
        assert!(code.contains("ld.param.u64"), "should emit ld.param.u64");
        assert!(code.contains("%rd7"), "should write to dest reg");
        assert!(code.contains("[fp+0]"), "should load from flat param");
    }

    #[test]
    fn emit_mul_const_power_of_2() {
        let mut ctr = 0;
        let code = emit_replacement(
            "%rd2",
            "u64",
            &FieldFormula::MulConst {
                flat_offset: 32,
                slope: 16,
            },
            "fp",
            &mut ctr,
        );
        // slope=16 is power of 2 → should use shl
        assert!(
            code.contains("shl.b64"),
            "should use shift for power-of-2 slope: {}",
            code
        );
        assert!(code.contains("4"), "shift by 4 for *16: {}", code);
    }

    #[test]
    fn emit_mul_const_non_power() {
        let mut ctr = 0;
        let code = emit_replacement(
            "%rd2",
            "u64",
            &FieldFormula::MulConst {
                flat_offset: 48,
                slope: 3,
            },
            "fp",
            &mut ctr,
        );
        assert!(
            code.contains("mul.lo.s64"),
            "should use mul for non-power slope: {}",
            code
        );
    }

    #[test]
    fn emit_ceil_div() {
        let mut ctr = 0;
        let code = emit_replacement(
            "%r1",
            "u32",
            &FieldFormula::CeilDiv {
                flat_offset: 64,
                divisor: 64,
            },
            "fp",
            &mut ctr,
        );
        assert!(code.contains("add.s32"), "should add divisor-1: {}", code);
        assert!(
            code.contains("shr.s32"),
            "should shift for power-of-2 div: {}",
            code
        );
        assert!(code.contains("63"), "divisor-1 = 63: {}", code);
    }

    #[test]
    fn emit_mul_add() {
        let mut ctr = 0;
        let code = emit_replacement(
            "%rd2",
            "u64",
            &FieldFormula::MulAdd {
                flat_offset: 32,
                slope: -16,
                intercept: 64,
            },
            "fp",
            &mut ctr,
        );
        assert!(code.contains("mul.lo.s64"), "should multiply: {}", code);
        assert!(code.contains("add.s64"), "should add intercept: {}", code);
        assert!(code.contains("64"), "intercept=64: {}", code);
    }

    // ── Full PTX rewrite tests ──

    #[test]
    fn replace_perimeter_on_cutlass_64x64x32() {
        let ptx = include_str!("../../ptx-fusion/kernels/cutlass_gemm_bf16_sm89.ptx");
        let json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let (rewritten, entry) = replace_perimeter(ptx, json, "ferrite_gemm_64x64x32").unwrap();

        // Basic structural checks
        assert!(
            rewritten.contains(".entry ferrite_gemm_64x64x32("),
            "should have new entry name"
        );
        assert!(
            rewritten.contains("ferrite_params[88]"),
            "should have flat param declaration: {}",
            rewritten
                .lines()
                .find(|l| l.contains("ferrite_params"))
                .unwrap_or("NOT FOUND"),
        );

        // The old mangled param name should not appear in ld.param instructions
        let old_ld_params: Vec<&str> = rewritten
            .lines()
            .filter(|l| l.contains("ld.param") && l.contains("_param_0"))
            .collect();
        assert!(
            old_ld_params.is_empty(),
            "should have no ld.param referencing old param name, found: {:?}",
            old_ld_params,
        );

        // All ld.param should reference ferrite_params
        let new_ld_params: Vec<&str> = rewritten
            .lines()
            .filter(|l| {
                let t = l.trim();
                t.contains("ld.param") && !t.starts_with("//")
            })
            .collect();
        for lp in &new_ld_params {
            assert!(
                lp.contains("ferrite_params"),
                "ld.param should reference ferrite_params: {}",
                lp
            );
        }

        // Should have replacement comments
        let replaced_count = rewritten
            .lines()
            .filter(|l| l.contains("[ferrite] replaced"))
            .count();
        assert!(
            replaced_count > 10,
            "should have many replacement comments, got {}",
            replaced_count
        );

        // Should have removed the mov.b64 param address lines
        let removed_count = rewritten
            .lines()
            .filter(|l| l.contains("[ferrite] removed"))
            .count();
        assert!(
            removed_count >= 2,
            "should have removed mov.b64 lines, got {}",
            removed_count
        );

        // The kernel body (non-param instructions) should be unchanged
        // Count mma.sync instructions — should be identical
        let orig_mma = ptx.lines().filter(|l| l.contains("mma.sync")).count();
        let new_mma = rewritten.lines().filter(|l| l.contains("mma.sync")).count();
        assert_eq!(orig_mma, new_mma, "MMA instructions should be preserved");

        // Count cp.async instructions — should be identical
        let orig_cp = ptx.lines().filter(|l| l.contains("cp.async")).count();
        let new_cp = rewritten.lines().filter(|l| l.contains("cp.async")).count();
        assert_eq!(orig_cp, new_cp, "cp.async instructions should be preserved");

        println!(
            "Rewritten PTX: {} lines ({} bytes)",
            rewritten.lines().count(),
            rewritten.len()
        );
        println!("Replaced {} ld.param sites", replaced_count);
        println!("Entry: {entry}");
    }

    #[test]
    fn rewritten_ptx_has_no_stale_base_register_refs() {
        let ptx = include_str!("../../ptx-fusion/kernels/cutlass_gemm_bf16_sm89.ptx");
        let json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let (rewritten, _) = replace_perimeter(ptx, json, "ferrite_gemm_test").unwrap();

        // After rewriting, there should be no ld.param instructions that
        // reference the old base registers (%rd1, %rd138) with offsets
        let stale: Vec<&str> = rewritten
            .lines()
            .filter(|l| {
                let t = l.trim();
                t.starts_with("ld.param") && (t.contains("[%rd1+") || t.contains("[%rd138+"))
            })
            .collect();
        assert!(
            stale.is_empty(),
            "should have no ld.param with old base registers, found {} stale:\n{}",
            stale.len(),
            stale.iter().take(5).cloned().collect::<Vec<_>>().join("\n"),
        );
    }
}
