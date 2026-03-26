use std::collections::{BTreeMap, BTreeSet};

/// The extracted protocol of a single PTX kernel.
#[derive(Debug, Clone)]
pub struct KernelProtocol {
    pub name: String,

    /// Register budget: maps type (e.g. ".f32") to count.
    pub registers: Vec<(String, usize)>,

    /// Shared memory regions declared with `.shared`.
    pub smem_regions: Vec<SmemRegion>,
    pub total_smem_bytes: usize,

    /// Kernel parameters from `.param` directives.
    pub params: Vec<KernelParam>,

    /// Global memory load sites — which param is read, what type.
    pub global_loads: Vec<DataPort>,

    /// Global memory store sites — which param is written, what type.
    pub global_stores: Vec<DataPort>,

    /// Count of shared memory loads (`ld.shared.*`).
    pub smem_loads: usize,

    /// Count of shared memory stores (`st.shared.*`).
    pub smem_stores: usize,

    /// Barrier IDs from `bar.sync N`.
    pub barriers: Vec<usize>,

    /// Whether this kernel uses `wmma` or `mma` instructions.
    pub has_mma: bool,
}

#[derive(Debug, Clone)]
pub struct SmemRegion {
    pub name: String,
    pub align: usize,
    pub elem_type: String,
    pub count: usize,
    pub size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct KernelParam {
    pub name: String,
    pub ptx_type: String,
    pub is_pointer: bool,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct DataPort {
    pub param_name: String,
    pub data_type: String,
    pub line: usize,
}

pub struct PtxParser;

impl PtxParser {
    pub fn parse(source: &str) -> Result<KernelProtocol, String> {
        let lines: Vec<&str> = source.lines().collect();

        let name = Self::extract_kernel_name(&lines)?;
        let params = Self::extract_params(&lines);
        let registers = Self::extract_registers(&lines);
        let smem_regions = Self::extract_smem(&lines);
        let total_smem_bytes = smem_regions.iter().map(|r| r.size_bytes).sum();

        // Build a map from register names to the param they were loaded from.
        // This lets us trace ld.global back to a specific parameter.
        let reg_to_param = Self::trace_param_registers(&lines, &params);

        let global_loads = Self::extract_global_loads(&lines, &reg_to_param);
        let global_stores = Self::extract_global_stores(&lines, &reg_to_param);
        let smem_loads = Self::count_pattern(&lines, "ld.shared");
        let smem_stores = Self::count_pattern(&lines, "st.shared");
        let barriers = Self::extract_barriers(&lines);
        let has_mma = lines.iter().any(|l| {
            let t = l.trim();
            t.contains("wmma.") || t.contains("mma.sync") || t.contains("mma.sp")
        });

        Ok(KernelProtocol {
            name,
            registers,
            smem_regions,
            total_smem_bytes,
            params,
            global_loads,
            global_stores,
            smem_loads,
            smem_stores,
            barriers,
            has_mma,
        })
    }

    fn extract_kernel_name(lines: &[&str]) -> Result<String, String> {
        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with(".visible") && trimmed.contains(".entry") {
                // .visible .entry rms_norm(
                let after_entry = trimmed
                    .split(".entry")
                    .nth(1)
                    .ok_or("malformed .entry line")?
                    .trim();
                let name_end = after_entry.find('(').unwrap_or(after_entry.len());
                return Ok(after_entry[..name_end].trim().to_string());
            }
        }
        Err("no .visible .entry found".to_string())
    }

    fn extract_params(lines: &[&str]) -> Vec<KernelParam> {
        let mut params = Vec::new();
        let mut in_params = false;
        let mut index = 0usize;

        for line in lines {
            let trimmed = line.trim();

            // Start tracking after .entry name(
            if trimmed.contains(".entry") {
                in_params = true;
                continue;
            }

            if in_params {
                if trimmed == ")" || trimmed == "){" {
                    break;
                }

                if trimmed.starts_with(".param") {
                    let is_pointer = trimmed.contains(".ptr")
                        || trimmed.contains(".u64")  // u64 params are typically pointers
                            && !trimmed.contains(".f32")
                            && !trimmed.contains(".u32");

                    // Extract the param name (last word before comma or end)
                    let clean = trimmed.trim_end_matches(',').trim();
                    let name = clean
                        .split_whitespace()
                        .last()
                        .unwrap_or("unknown")
                        .to_string();

                    // Extract the type
                    let ptx_type = if trimmed.contains(".f32") {
                        ".f32"
                    } else if trimmed.contains(".f16") {
                        ".f16"
                    } else if trimmed.contains(".u64") {
                        ".u64"
                    } else if trimmed.contains(".u32") {
                        ".u32"
                    } else if trimmed.contains(".s32") {
                        ".s32"
                    } else {
                        ".unknown"
                    };

                    params.push(KernelParam {
                        name,
                        ptx_type: ptx_type.to_string(),
                        is_pointer,
                        index,
                    });
                    index += 1;
                }
            }
        }

        params
    }

    fn extract_registers(lines: &[&str]) -> Vec<(String, usize)> {
        let mut regs: BTreeMap<String, usize> = BTreeMap::new();

        for line in lines {
            let trimmed = line.trim();
            if !trimmed.starts_with(".reg") {
                continue;
            }

            // .reg .f32 %f<16>;
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }

            let reg_type = parts[1].to_string(); // .f32, .u32, .u64, .pred
            let reg_decl = parts[2].trim_end_matches(';');

            // Extract count from %name<count>
            if let Some(angle_start) = reg_decl.find('<')
                && let Some(angle_end) = reg_decl.find('>')
            {
                let count_str = &reg_decl[angle_start + 1..angle_end];
                if let Ok(count) = count_str.parse::<usize>() {
                    *regs.entry(reg_type).or_insert(0) += count;
                }
            }
        }

        regs.into_iter().collect()
    }

    fn extract_smem(lines: &[&str]) -> Vec<SmemRegion> {
        let mut regions = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            if !trimmed.contains(".shared") {
                continue;
            }

            // .shared .align 16 .f32 smem_vec[4096];
            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            let mut align = 1usize;
            let mut elem_type = String::new();
            let mut name = String::new();
            let mut count = 0usize;

            let mut i = 0;
            while i < parts.len() {
                if parts[i] == ".align" && i + 1 < parts.len() {
                    align = parts[i + 1].parse().unwrap_or(1);
                    i += 2;
                    continue;
                }
                if parts[i].starts_with('.') && parts[i] != ".shared" {
                    elem_type = parts[i].to_string();
                }
                if parts[i].contains('[') {
                    // smem_vec[4096];
                    let clean = parts[i].trim_end_matches(';');
                    let bracket = clean.find('[').unwrap();
                    name = clean[..bracket].to_string();
                    let end_bracket = clean.find(']').unwrap_or(clean.len());
                    count = clean[bracket + 1..end_bracket].parse().unwrap_or(0);
                }
                i += 1;
            }

            let elem_size = type_size_bytes(&elem_type);
            let size_bytes = count * elem_size;

            if !name.is_empty() {
                regions.push(SmemRegion {
                    name,
                    align,
                    elem_type,
                    count,
                    size_bytes,
                });
            }
        }

        regions
    }

    /// Public wrapper for trace_param_registers (used by fuse module).
    pub fn trace_param_registers_pub(
        lines: &[&str],
        params: &[KernelParam],
    ) -> BTreeMap<String, String> {
        Self::trace_param_registers(lines, params)
    }

    /// Trace which registers hold pointers loaded from params.
    /// e.g., `ld.param.u64 %rd0, [input];` → %rd0 maps to "input"
    fn trace_param_registers(lines: &[&str], _params: &[KernelParam]) -> BTreeMap<String, String> {
        let mut reg_to_param: BTreeMap<String, String> = BTreeMap::new();

        for line in lines {
            let trimmed = line.trim();
            if !trimmed.starts_with("ld.param") {
                continue;
            }

            // ld.param.u64 %rd0, [input];
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();

            if parts.len() >= 3 {
                let reg = parts[1].trim_end_matches(',').to_string();
                // The param name is in brackets: [input] or [epsilon]
                let param_ref = parts[2].trim_matches(|c| c == '[' || c == ']' || c == ';');
                reg_to_param.insert(reg, param_ref.to_string());
            }
        }

        // Trace pointer-propagating instructions to follow address chains.
        // Handles: add.u64, add.s64 (nvcc uses signed), cvta.to.global.u64 (address space cast).
        // Do multiple passes to propagate through chains.
        for _pass in 0..4 {
            for line in lines {
                let trimmed = line.trim();
                let opcode = trimmed.split_whitespace().next().unwrap_or("");

                // cvta.to.global.u64 %rd4, %rd1; — propagates pointer identity (1:1 rename)
                if opcode == "cvta.to.global.u64" {
                    let parts: Vec<&str> = trimmed
                        .split([',', ' ', '\t'])
                        .filter(|s| !s.is_empty())
                        .collect();
                    if parts.len() >= 3 {
                        let dst = parts[1].trim_end_matches(',').to_string();
                        let src = parts[2].trim_end_matches(';');
                        if let Some(param) = reg_to_param.get(src).cloned() {
                            reg_to_param.entry(dst).or_insert(param);
                        }
                    }
                    continue;
                }

                // mov.u64 / mov.b64 — register copy (1:1 rename)
                if opcode == "mov.u64" || opcode == "mov.b64" {
                    let parts: Vec<&str> = trimmed
                        .split([',', ' ', '\t'])
                        .filter(|s| !s.is_empty())
                        .collect();
                    if parts.len() >= 3 {
                        let dst = parts[1].trim_end_matches(',').to_string();
                        let src = parts[2].trim_end_matches(';');
                        if let Some(param) = reg_to_param.get(src).cloned() {
                            reg_to_param.entry(dst).or_insert(param);
                        }
                    }
                    continue;
                }

                // add.u64 / add.s64 — pointer + offset propagation
                if opcode != "add.u64" && opcode != "add.s64" {
                    continue;
                }

                let parts: Vec<&str> = trimmed
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                    .collect();

                if parts.len() >= 4 {
                    let dst = parts[1].trim_end_matches(',').to_string();
                    let src1 = parts[2].trim_end_matches(',');
                    let src2 = parts[3].trim_end_matches(';');

                    // If either source is a known param pointer, propagate
                    let traced = reg_to_param
                        .get(src1)
                        .or_else(|| reg_to_param.get(src2))
                        .cloned();
                    if let Some(param) = traced {
                        reg_to_param.entry(dst).or_insert(param);
                    }
                }
            }
        }

        reg_to_param
    }

    fn extract_global_loads(
        lines: &[&str],
        reg_to_param: &BTreeMap<String, String>,
    ) -> Vec<DataPort> {
        let mut loads = Vec::new();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            // Skip predicated-away or comment lines, match ld.global.*
            if !trimmed.contains("ld.global") {
                continue;
            }

            // ld.global.f32 %f1, [%rd4];
            let data_type = extract_load_store_type(trimmed);

            // Find the address register (in brackets)
            let param_name = if let Some(bracket_start) = trimmed.find('[') {
                let bracket_end = trimmed.find(']').unwrap_or(trimmed.len());
                let addr_reg = &trimmed[bracket_start + 1..bracket_end];
                // Might have an offset: [%rd4+8], just take the register part
                let base_reg = addr_reg
                    .split('+')
                    .next()
                    .unwrap_or(addr_reg)
                    .trim()
                    .to_string();
                reg_to_param
                    .get(&base_reg)
                    .cloned()
                    .unwrap_or_else(|| format!("?{base_reg}"))
            } else {
                "?unknown".to_string()
            };

            loads.push(DataPort {
                param_name,
                data_type,
                line: line_num + 1,
            });
        }

        loads
    }

    fn extract_global_stores(
        lines: &[&str],
        reg_to_param: &BTreeMap<String, String>,
    ) -> Vec<DataPort> {
        let mut stores = Vec::new();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if !trimmed.contains("st.global") {
                continue;
            }

            // st.global.f32 [%rd6], %f7;
            let data_type = extract_load_store_type(trimmed);

            let param_name = if let Some(bracket_start) = trimmed.find('[') {
                let bracket_end = trimmed.find(']').unwrap_or(trimmed.len());
                let addr_reg = &trimmed[bracket_start + 1..bracket_end];
                let base_reg = addr_reg
                    .split('+')
                    .next()
                    .unwrap_or(addr_reg)
                    .trim()
                    .to_string();
                reg_to_param
                    .get(&base_reg)
                    .cloned()
                    .unwrap_or_else(|| format!("?{base_reg}"))
            } else {
                "?unknown".to_string()
            };

            stores.push(DataPort {
                param_name,
                data_type,
                line: line_num + 1,
            });
        }

        stores
    }

    fn extract_barriers(lines: &[&str]) -> Vec<usize> {
        let mut barriers = BTreeSet::new();

        for line in lines {
            let trimmed = line.trim();
            if !trimmed.starts_with("bar.sync") {
                continue;
            }

            // bar.sync 0;
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                let id_str = parts[1].trim_end_matches(';');
                if let Ok(id) = id_str.parse::<usize>() {
                    barriers.insert(id);
                }
            }
        }

        barriers.into_iter().collect()
    }

    fn count_pattern(lines: &[&str], pattern: &str) -> usize {
        lines.iter().filter(|l| l.trim().contains(pattern)).count()
    }
}

fn extract_load_store_type(instruction: &str) -> String {
    // ld.global.f32 or st.global.v4.f16 → extract the data type
    // The type is the last dot-separated segment before the first space
    let first_word = instruction.split_whitespace().next().unwrap_or("");
    let dot_parts: Vec<&str> = first_word.split('.').collect();
    dot_parts.last().unwrap_or(&"unknown").to_string()
}

fn type_size_bytes(ty: &str) -> usize {
    match ty {
        ".f64" | ".u64" | ".s64" | ".b64" => 8,
        ".f32" | ".u32" | ".s32" | ".b32" => 4,
        ".f16" | ".u16" | ".s16" | ".b16" => 2,
        ".u8" | ".s8" | ".b8" => 1,
        _ => 4, // default assumption
    }
}
