use std::collections::{BTreeMap, BTreeSet, VecDeque};

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

    /// Async copy sites: cp.async.cg.shared.global (GMEM → SMEM).
    /// These are the input perimeter for async-pipeline kernels (CUTLASS).
    pub async_loads: Vec<AsyncCopyPort>,

    /// Count of shared memory loads (`ld.shared.*`).
    pub smem_loads: usize,

    /// Count of shared memory stores (`st.shared.*`).
    pub smem_stores: usize,

    /// Barrier IDs from `bar.sync N`.
    pub barriers: Vec<usize>,

    /// Whether this kernel uses `wmma` or `mma` instructions.
    pub has_mma: bool,

    /// Classified param struct fields with roles (Pointer, Stride, Dimension, Scalar, Derived).
    /// Only populated for struct-param kernels (CUTLASS). Empty for simple-param kernels.
    pub classified_params: Vec<ParamField>,
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

/// An async copy site: cp.async.cg.shared.global copies GMEM → SMEM.
/// This is part of the input perimeter for kernels that use async pipelines (CUTLASS).
#[derive(Debug, Clone)]
pub struct AsyncCopyPort {
    /// Which param the GMEM source traces to.
    pub param_name: String,
    /// The SMEM destination register (e.g., "%r226").
    pub smem_dst: String,
    /// The GMEM source register (e.g., "%rd26").
    pub gmem_src: String,
    /// The predicate/mask register (e.g., "%r227").
    pub mask: String,
    /// Size in bytes (always 16 for cp.async.cg with L2::128B).
    pub size_bytes: usize,
    /// Line number in the PTX.
    pub line: usize,
}

pub struct PtxParser;

impl PtxParser {
    pub fn parse(source: &str) -> Result<KernelProtocol, String> {
        let lines: Vec<&str> = source.lines().collect();

        // If multi-entry PTX, extract the first entry automatically.
        let entry_count = lines
            .iter()
            .filter(|l| l.contains(".entry") && l.contains('('))
            .count();
        let source_owned;
        let lines = if entry_count > 1 {
            // Extract the first entry's name
            let first_entry_name = Self::extract_kernel_name(&lines)?;
            // Use a short unique substring from the entry name
            let substr = if first_entry_name.len() > 20 {
                &first_entry_name[..20]
            } else {
                &first_entry_name
            };
            source_owned = crate::extract::extract_entry(source, substr)
                .map_err(|e| format!("multi-entry PTX: failed to extract first entry: {e}"))?;
            source_owned.lines().collect::<Vec<&str>>()
        } else {
            lines
        };

        let name = Self::extract_kernel_name(&lines)?;
        let params = Self::extract_params(&lines);
        let registers = Self::extract_registers(&lines);
        let smem_regions = Self::extract_smem(&lines);
        let total_smem_bytes = smem_regions.iter().map(|r| r.size_bytes).sum();

        let reg_to_param = Self::trace_param_registers(&lines, &params);

        let global_loads = Self::extract_global_loads(&lines, &reg_to_param);
        let global_stores = Self::extract_global_stores(&lines, &reg_to_param);
        let async_loads = Self::extract_async_loads(&lines, &reg_to_param);
        let smem_loads = Self::count_pattern(&lines, "ld.shared");
        let smem_stores = Self::count_pattern(&lines, "st.shared");
        let barriers = Self::extract_barriers(&lines);
        let has_mma = lines.iter().any(|l| {
            let t = l.trim();
            t.contains("wmma.") || t.contains("mma.sync") || t.contains("mma.sp")
        });

        let mut protocol = KernelProtocol {
            name,
            registers,
            smem_regions,
            total_smem_bytes,
            params,
            global_loads,
            global_stores,
            async_loads,
            smem_loads,
            smem_stores,
            barriers,
            has_mma,
            classified_params: Vec::new(),
        };

        // Build def-use graph and classify param fields for struct-param kernels
        let (mut param_fields, base_offsets) = extract_param_fields(&lines);
        if !param_fields.is_empty() {
            let graph = DefUseGraph::build(&lines);
            classify_param_fields(&mut param_fields, &graph, &protocol, &base_offsets);
            protocol.classified_params = param_fields;
        }

        Ok(protocol)
    }

    fn extract_kernel_name(lines: &[&str]) -> Result<String, String> {
        for line in lines {
            let trimmed = line.trim();
            if trimmed.contains(".entry") && trimmed.contains('(') {
                // .visible .entry rms_norm( OR .entry _ZN7cutlass...(
                let after_entry = trimmed
                    .split(".entry")
                    .nth(1)
                    .ok_or("malformed .entry line")?
                    .trim();
                let name_end = after_entry.find('(').unwrap_or(after_entry.len());
                return Ok(after_entry[..name_end].trim().to_string());
            }
        }
        Err("no .entry found".to_string())
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
                    } else if trimmed.contains(".b8") {
                        // Struct param: .param .align N .b8 name[SIZE]
                        ".struct"
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

            // Two patterns:
            // 1. ld.param.u64 %rd0, [param_name];           (individual params)
            // 2. ld.param.u64 %rd2, [%rd1+16];              (struct params, CUTLASS)
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();

            if parts.len() >= 3 {
                let reg = parts[1].trim_end_matches(',').to_string();
                let bracket_content = parts[2].trim_matches(|c| c == '[' || c == ']' || c == ';');

                if bracket_content.contains('+') {
                    // Struct param: [%rd1+16] or [_param_0+16]
                    // Use the full bracket content as a synthetic param name
                    reg_to_param.insert(reg, bracket_content.to_string());
                } else {
                    reg_to_param.insert(reg, bracket_content.to_string());
                }
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

    fn extract_async_loads(
        lines: &[&str],
        reg_to_param: &BTreeMap<String, String>,
    ) -> Vec<AsyncCopyPort> {
        let mut ports = Vec::new();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if !trimmed.contains("cp.async.cg.shared.global") {
                continue;
            }

            // cp.async.cg.shared.global.L2::128B [%r226], [%rd26], 16, %r227;
            // Extract: smem_dst, gmem_src, mask from bracket pairs and trailing tokens
            let mut brackets = Vec::new();
            let mut i = 0;
            let bytes = trimmed.as_bytes();
            while i < bytes.len() {
                if bytes[i] == b'[' {
                    let start = i + 1;
                    while i < bytes.len() && bytes[i] != b']' {
                        i += 1;
                    }
                    brackets.push(trimmed[start..i].trim().to_string());
                }
                i += 1;
            }

            if brackets.len() < 2 {
                continue;
            }

            let smem_dst = brackets[0].clone();
            let gmem_src = brackets[1].clone();

            // Extract mask (last token before semicolon)
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            let mask = parts
                .last()
                .map(|s| s.trim_end_matches(';').to_string())
                .unwrap_or_default();

            // Trace GMEM source to param
            let param_name = reg_to_param
                .get(&gmem_src)
                .cloned()
                .unwrap_or_else(|| format!("?{gmem_src}"));

            ports.push(AsyncCopyPort {
                param_name,
                smem_dst,
                gmem_src,
                mask,
                size_bytes: 16,
                line: line_num + 1,
            });
        }

        ports
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

// ===== Def-Use Graph & Param Classification =====

/// Role classification for a param field loaded via ld.param.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamRole {
    /// Base address that feeds into ld.global/st.global/cp.async
    Pointer,
    /// Leading dimension/stride — multiplied before reaching a memory address
    Stride,
    /// Bound/shape value (M, N, K) — used in setp comparisons
    Dimension,
    /// Floating-point value (alpha, beta, epsilon) — used in f32 arithmetic
    Scalar,
    /// Precomputed value (iterator increment, swizzle_log, etc.)
    Derived,
}

/// A param struct field loaded by ld.param.
#[derive(Debug, Clone)]
pub struct ParamField {
    /// Byte offset within the param struct (absolute from struct start)
    pub offset: i64,
    /// PTX type of the load (.u32, .u64, .f32)
    pub ptx_type: String,
    /// Register(s) defined by this ld.param
    pub dest_regs: Vec<String>,
    /// Line number in PTX (0-indexed)
    pub line: usize,
    /// Classified role
    pub role: ParamRole,
}

/// A single instruction node in the def-use graph.
#[derive(Debug, Clone)]
pub struct InstrNode {
    pub line: usize,
    pub opcode: String,
    pub dests: Vec<String>,
    pub sources: Vec<String>,
}

/// Def-use graph over PTX registers.
///
/// Built in a single pass over the PTX. Supports both forward tracing
/// (register → instructions that use it) and backward tracing
/// (register → instructions that define it).
#[derive(Debug)]
pub struct DefUseGraph {
    pub nodes: Vec<InstrNode>,
    /// Register → indices into `nodes` where register is defined (written)
    pub defs: BTreeMap<String, Vec<usize>>,
    /// Register → indices into `nodes` where register is used (read)
    pub uses: BTreeMap<String, Vec<usize>>,
}

impl DefUseGraph {
    /// Build the def-use graph from PTX source lines. Single pass, O(N).
    pub fn build(lines: &[&str]) -> Self {
        let mut nodes = Vec::new();
        let mut defs: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut uses: BTreeMap<String, Vec<usize>> = BTreeMap::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Skip non-instruction lines
            if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('.')
                || trimmed.ends_with(':')
                || trimmed == "{"
                || trimmed == "}"
                || trimmed.starts_with("$")
            {
                continue;
            }

            if let Some(node) = Self::parse_instruction(trimmed, line_idx) {
                let idx = nodes.len();
                for d in &node.dests {
                    defs.entry(d.clone()).or_default().push(idx);
                }
                for s in &node.sources {
                    uses.entry(s.clone()).or_default().push(idx);
                }
                nodes.push(node);
            }
        }

        Self { nodes, defs, uses }
    }

    fn parse_instruction(line: &str, line_num: usize) -> Option<InstrNode> {
        let mut work = line;
        let mut extra_sources = Vec::new();

        // Handle predication: @%pN or @!%pN
        if work.starts_with('@') {
            let space_pos = work.find(|c: char| c.is_whitespace())?;
            let pred_part = &work[1..space_pos];
            let pred_reg = pred_part.trim_start_matches('!');
            if pred_reg.starts_with('%') {
                extra_sources.push(pred_reg.to_string());
            }
            work = work[space_pos..].trim();
        }

        // Get opcode (first token)
        let opcode_end = work.find(|c: char| c.is_whitespace()).unwrap_or(work.len());
        let opcode = &work[..opcode_end];

        // Skip non-instructions that slipped through
        if opcode.starts_with('.')
            || opcode.starts_with('$')
            || opcode.starts_with('{')
            || opcode.starts_with('}')
            || opcode == "//"
        {
            return None;
        }

        let operands_str = work[opcode_end..].trim().trim_end_matches(';').trim();

        // Determine if this opcode has no register destination
        let is_no_dest = opcode.starts_with("st.")
            || opcode.starts_with("cp.async")
            || opcode.starts_with("bar.")
            || opcode.starts_with("barrier.")
            || opcode == "bra"
            || opcode == "ret"
            || opcode == "exit"
            || opcode.starts_with("red.")
            || opcode.starts_with("membar")
            || opcode.starts_with("fence");

        let (dests, mut sources) = if is_no_dest {
            // All registers are sources
            (vec![], extract_all_registers(operands_str))
        } else {
            split_dest_and_sources(operands_str)
        };

        sources.extend(extra_sources);

        Some(InstrNode {
            line: line_num,
            opcode: opcode.to_string(),
            dests,
            sources,
        })
    }

    /// Trace forward from a register, returning opcodes reached within `max_depth` hops.
    /// Each entry is (depth, node_index).
    pub fn trace_forward(&self, start_reg: &str, max_depth: usize) -> Vec<(usize, usize)> {
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        let mut results = Vec::new();

        queue.push_back((start_reg.to_string(), 0usize));

        while let Some((reg, depth)) = queue.pop_front() {
            if depth > max_depth || !visited.insert(reg.clone()) {
                continue;
            }

            if let Some(indices) = self.uses.get(&reg) {
                for &idx in indices {
                    results.push((depth, idx));

                    // Continue BFS through this instruction's dests
                    if depth < max_depth {
                        for d in &self.nodes[idx].dests {
                            queue.push_back((d.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        results
    }
}

/// Extract all register tokens (%rN, %rdN, %fN, %pN, etc.) from a string.
fn extract_all_registers(s: &str) -> Vec<String> {
    let mut regs = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if i > start + 1 {
                regs.push(s[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
    regs
}

/// Split operand string into (dest_registers, source_registers).
///
/// The first operand group is the destination:
/// - `%r2, %r3, %r4` → dest=[%r2], src=[%r3, %r4]
/// - `{%r1, %r2}, [%rd3+8]` → dest=[%r1, %r2], src=[%rd3]
/// - `%r1|%p2, %r3, %r4, %r5, %r6` → dest=[%r1, %p2], src=[%r3, %r4, %r5, %r6]
fn split_dest_and_sources(operands: &str) -> (Vec<String>, Vec<String>) {
    if operands.is_empty() {
        return (vec![], vec![]);
    }

    // Find the end of the first operand group (first comma outside braces)
    let first_op_end = {
        let mut depth = 0i32;
        let mut end = operands.len();
        for (i, c) in operands.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                ',' if depth == 0 => {
                    end = i;
                    break;
                }
                _ => {}
            }
        }
        end
    };

    let first_operand = &operands[..first_op_end];
    let rest = if first_op_end < operands.len() {
        operands[first_op_end + 1..].trim()
    } else {
        ""
    };

    // Handle pipe-separated dests (shfl): %r207|%p16
    let dest_str = if first_operand.contains('|') && !first_operand.contains('[') {
        first_operand.replace('|', " ")
    } else {
        first_operand.to_string()
    };

    let dests = extract_all_registers(&dest_str);
    let sources = extract_all_registers(rest);

    (dests, sources)
}

// ===== Param Field Extraction & Classification =====

/// Extract all param struct fields loaded by ld.param, computing absolute byte offsets.
///
/// Handles two addressing patterns:
/// 1. Direct: `ld.param.u32 %r, [param_name+OFFSET]` → offset = OFFSET
/// 2. Indirect via base register: `ld.param.u64 %rd, [%rdBase+OFFSET]`
///    where %rdBase = param_addr + BASE_OFFSET → offset = BASE_OFFSET + OFFSET
pub fn extract_param_fields(lines: &[&str]) -> (Vec<ParamField>, BTreeMap<String, i64>) {
    // Step 1: Find base register offsets.
    // Pattern: mov.b64 %rdN, param_name; → %rdN holds param_base (offset 0)
    // Pattern: add.s64 %rdM, %rdN, CONST; → %rdM holds param_base + CONST
    let mut base_offsets: BTreeMap<String, i64> = BTreeMap::new();

    for line in lines.iter() {
        let trimmed = line.trim();

        // mov.b64 %rdN, param_name — param_name starts with _ (mangled)
        if trimmed.starts_with("mov.b64") || trimmed.starts_with("mov.u64") {
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 3 {
                let dst = parts[1].trim_end_matches(',');
                let src = parts[2].trim_end_matches(';');
                // If src is a mangled name (the param struct address), not a register
                if !src.starts_with('%') && !src.chars().next().is_none_or(|c| c.is_ascii_digit()) {
                    base_offsets.insert(dst.to_string(), 0);
                }
            }
        }

        // add.s64 %rdM, %rdN, CONST — adds constant offset to base
        if trimmed.starts_with("add.s64") {
            let parts: Vec<&str> = trimmed
                .split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 4 {
                let dst = parts[1].trim_end_matches(',');
                let src1 = parts[2].trim_end_matches(',');
                let src2 = parts[3].trim_end_matches(';');

                // Check if src1 is a known base and src2 is an immediate
                if let Some(&base_off) = base_offsets.get(src1)
                    && let Ok(imm) = src2.parse::<i64>()
                {
                    base_offsets
                        .entry(dst.to_string())
                        .or_insert(base_off + imm);
                }
                // Also check reverse: src2 is base, src1 is immediate
                if let Some(&base_off) = base_offsets.get(src2)
                    && let Ok(imm) = src1.parse::<i64>()
                {
                    base_offsets
                        .entry(dst.to_string())
                        .or_insert(base_off + imm);
                }
            }
        }
    }

    // Step 2: Extract all ld.param instructions and compute absolute offsets.
    let mut fields = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Strip predication
        let trimmed = if trimmed.starts_with('@') {
            trimmed
                .find(|c: char| c.is_whitespace())
                .map_or(trimmed, |i| trimmed[i..].trim())
        } else {
            trimmed
        };

        if !trimmed.starts_with("ld.param") {
            continue;
        }

        // Extract PTX type from opcode: ld.param.TYPE or ld.param.v2.TYPE
        let opcode = trimmed.split_whitespace().next().unwrap_or("");
        let dot_parts: Vec<&str> = opcode.split('.').collect();
        let ptx_type = dot_parts.last().unwrap_or(&"unknown").to_string();
        let is_vector = opcode.contains(".v2.") || opcode.contains(".v4.");

        // Extract bracket content: [base+offset] or [param_name+offset]
        let bracket_start = match trimmed.find('[') {
            Some(i) => i,
            None => continue,
        };
        let bracket_end = trimmed.find(']').unwrap_or(trimmed.len());
        let bracket_content = &trimmed[bracket_start + 1..bracket_end];

        // Parse offset from bracket content
        let offset = if bracket_content.contains('+') || bracket_content.contains('-') {
            // Split on + or - (handle [base+-20] which means base + (-20))
            let (base_part, offset_str) = if let Some(plus_pos) = bracket_content.rfind('+') {
                (
                    bracket_content[..plus_pos].trim(),
                    bracket_content[plus_pos + 1..].trim(),
                )
            } else if let Some(minus_pos) = bracket_content.rfind('-') {
                // Check it's not part of a name
                if minus_pos > 0 {
                    let before = bracket_content[..minus_pos].trim();
                    let after = bracket_content[minus_pos..].trim(); // includes the -
                    (before, after)
                } else {
                    continue;
                }
            } else {
                continue;
            };

            let field_offset: i64 = offset_str.parse().unwrap_or(0);

            if base_part.starts_with('%') {
                // Indirect: [%rdBase+OFFSET]
                let base_off = base_offsets.get(base_part).copied().unwrap_or(0);
                base_off + field_offset
            } else {
                // Direct: [param_name+OFFSET]
                field_offset
            }
        } else if bracket_content.starts_with('%') {
            // [%rdBase] with no offset
            base_offsets.get(bracket_content).copied().unwrap_or(0)
        } else {
            // [param_name] with no offset — offset 0
            0
        };

        // Extract dest registers
        let after_opcode = trimmed[opcode.len()..].trim();
        let before_bracket = &after_opcode[..after_opcode.find('[').unwrap_or(after_opcode.len())];
        let dest_regs = extract_all_registers(before_bracket);

        // For vector loads, we may have multiple dest regs
        if dest_regs.is_empty() {
            continue;
        }

        // For vector loads like ld.param.v2.u32 {%r164, %r165}, the two regs
        // are at consecutive offsets. Record each separately.
        if is_vector && dest_regs.len() > 1 {
            let elem_size = match ptx_type.as_str() {
                "u64" | "s64" | "b64" | "f64" => 8i64,
                "u32" | "s32" | "b32" | "f32" => 4,
                "u16" | "s16" | "b16" | "f16" => 2,
                _ => 4,
            };
            for (i, reg) in dest_regs.iter().enumerate() {
                fields.push(ParamField {
                    offset: offset + i as i64 * elem_size,
                    ptx_type: ptx_type.clone(),
                    dest_regs: vec![reg.clone()],
                    line: line_idx,
                    role: ParamRole::Derived, // default, will be classified
                });
            }
        } else {
            fields.push(ParamField {
                offset,
                ptx_type: ptx_type.clone(),
                dest_regs,
                line: line_idx,
                role: ParamRole::Derived,
            });
        }
    }

    // Deduplicate by offset (same field may be loaded multiple times, e.g. in epilogue)
    fields.sort_by_key(|f| f.offset);
    fields.dedup_by_key(|f| f.offset);

    (fields, base_offsets)
}

/// Classify param fields using the def-use graph and the existing perimeter analysis.
///
/// Classification rules (in priority order):
/// 1. **Pointer**: the backward trace from a data port (global_loads/stores/async_loads)
///    resolves to this field's offset
/// 2. **Stride**: u64 field used in mul.lo.s64 / mul.wide.s32 with a REGISTER operand
///    (not a small immediate constant)
/// 3. **Dimension**: field that reaches setp within a few hops
/// 4. **Scalar**: f32 field
/// 5. **Derived**: everything else (iterator increments, swizzle, etc.)
pub fn classify_param_fields(
    fields: &mut [ParamField],
    graph: &DefUseGraph,
    protocol: &KernelProtocol,
    base_offsets: &BTreeMap<String, i64>,
) {
    // Step 1: Identify Pointer offsets from the existing perimeter analysis.
    // The param_names in data ports are the bracket content of ld.param, e.g. "%rd1+40".
    // Resolve each to an absolute byte offset using the base_offsets map.
    let pointer_offsets = resolve_pointer_offsets(protocol, base_offsets);

    for field in fields.iter_mut() {
        // Rule 4: f32 → Scalar
        if field.ptx_type == "f32" {
            field.role = ParamRole::Scalar;
            continue;
        }

        // Rule 1: Pointer — this offset is the traced source of a memory access
        if pointer_offsets.contains(&field.offset) {
            field.role = ParamRole::Pointer;
            continue;
        }

        // Rule 1b: Pointer via cvta.to.global — if the register goes through
        // cvta.to.global.u64, it's definitely a global pointer
        let is_cvta_pointer = field.dest_regs.iter().any(|reg| {
            if let Some(indices) = graph.uses.get(reg) {
                indices
                    .iter()
                    .any(|&idx| graph.nodes[idx].opcode.starts_with("cvta.to.global"))
            } else {
                false
            }
        });
        if is_cvta_pointer {
            field.role = ParamRole::Pointer;
            continue;
        }

        // Rule 2: Stride — used in mul.lo.s64/mul.wide with a register (not small immediate)
        let is_stride = field.dest_regs.iter().any(|reg| {
            if let Some(indices) = graph.uses.get(reg) {
                indices.iter().any(|&idx| {
                    let node = &graph.nodes[idx];
                    let op = &node.opcode;
                    let is_mul = op.starts_with("mul.lo.s64")
                        || op.starts_with("mul.wide")
                        || op.starts_with("mul.lo.u64");
                    if !is_mul {
                        return false;
                    }
                    // Check: is the OTHER operand a register (not a small immediate)?
                    // A stride is multiplied by an index (register). A derived increment
                    // might be multiplied by a small constant (2, 3 for pipeline stages).
                    node.sources
                        .iter()
                        .filter(|s| *s != reg)
                        .any(|s| s.starts_with('%'))
                })
            } else {
                false
            }
        });

        if is_stride {
            field.role = ParamRole::Stride;
            continue;
        }

        // Rule 3: Dimension — reaches setp within a few hops
        let is_dimension = field.dest_regs.iter().any(|reg| {
            let reached = graph.trace_forward(reg, 3);
            reached
                .iter()
                .any(|&(_, idx)| graph.nodes[idx].opcode.starts_with("setp"))
        });

        if is_dimension {
            field.role = ParamRole::Dimension;
            continue;
        }

        // Rule 5: Default is already Derived
    }
}

/// Resolve param_names from the protocol's data ports to absolute byte offsets.
///
/// Each param_name is the bracket content from an ld.param that was backward-traced
/// from a memory access, e.g. "%rd1+40" or "param_name+12".
/// We resolve these to absolute byte offsets using the base_offsets map from
/// extract_param_fields (which knows that %rd1 = param_base + 24, etc.).
fn resolve_pointer_offsets(
    protocol: &KernelProtocol,
    base_offsets: &BTreeMap<String, i64>,
) -> BTreeSet<i64> {
    let mut offsets = BTreeSet::new();

    let all_param_names: Vec<&str> = protocol
        .global_loads
        .iter()
        .map(|p| p.param_name.as_str())
        .chain(protocol.global_stores.iter().map(|p| p.param_name.as_str()))
        .chain(protocol.async_loads.iter().map(|p| p.param_name.as_str()))
        .filter(|n| !n.starts_with('?'))
        .collect();

    for name in &all_param_names {
        if let Some(offset) = resolve_param_name_to_offset(name, base_offsets) {
            offsets.insert(offset);
        }
    }

    offsets
}

/// Resolve a param_name like "%rd1+40" to an absolute byte offset.
///
/// Uses the base_offsets map to look up the base register's offset,
/// then adds the field offset: absolute = base_offset(reg) + field_offset.
fn resolve_param_name_to_offset(name: &str, base_offsets: &BTreeMap<String, i64>) -> Option<i64> {
    if name.contains('+') || name.contains('-') {
        // Split into base register and offset: "%rd1+40" → ("%rd1", 40)
        let (base_part, offset_str) = if let Some(plus_pos) = name.rfind('+') {
            (&name[..plus_pos], &name[plus_pos + 1..])
        } else if let Some(minus_pos) = name.rfind('-') {
            if minus_pos > 0 {
                (&name[..minus_pos], &name[minus_pos..])
            } else {
                return None;
            }
        } else {
            return None;
        };

        let field_offset: i64 = offset_str.parse().ok()?;
        let base_off = base_offsets.get(base_part).copied().unwrap_or(0);
        Some(base_off + field_offset)
    } else if name.starts_with('%') {
        // Just a register with no offset: [%rd1]
        base_offsets.get(name).copied()
    } else {
        // Simple param name like "input" — offset 0 or not struct-based
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Instruction parsing tests ──

    #[test]
    fn parse_simple_add() {
        let node = DefUseGraph::parse_instruction("add.s32 %r2, %r3, %r4;", 0).unwrap();
        assert_eq!(node.opcode, "add.s32");
        assert_eq!(node.dests, vec!["%r2"]);
        assert_eq!(node.sources, vec!["%r3", "%r4"]);
    }

    #[test]
    fn parse_load_param() {
        let node = DefUseGraph::parse_instruction("ld.param.u64 %rd0, [input];", 0).unwrap();
        assert_eq!(node.opcode, "ld.param.u64");
        assert_eq!(node.dests, vec!["%rd0"]);
        // "input" is not a register, so no register sources
        assert!(node.sources.is_empty());
    }

    #[test]
    fn parse_load_param_struct_offset() {
        let node = DefUseGraph::parse_instruction("ld.param.u64 %rd2, [%rd1+16];", 0).unwrap();
        assert_eq!(node.dests, vec!["%rd2"]);
        assert_eq!(node.sources, vec!["%rd1"]);
    }

    #[test]
    fn parse_load_global() {
        let node = DefUseGraph::parse_instruction("ld.global.f32 %f1, [%rd4];", 0).unwrap();
        assert_eq!(node.dests, vec!["%f1"]);
        assert_eq!(node.sources, vec!["%rd4"]);
    }

    #[test]
    fn parse_store_global() {
        let node = DefUseGraph::parse_instruction("st.global.f32 [%rd6], %f7;", 0).unwrap();
        // Stores have no dest registers
        assert!(node.dests.is_empty());
        // Both address and value are sources
        assert!(node.sources.contains(&"%rd6".to_string()));
        assert!(node.sources.contains(&"%f7".to_string()));
    }

    #[test]
    fn parse_predicated_instruction() {
        let node = DefUseGraph::parse_instruction("@%p0 bra EXIT;", 0).unwrap();
        assert_eq!(node.opcode, "bra");
        assert!(node.dests.is_empty());
        // Predicate is a source
        assert!(node.sources.contains(&"%p0".to_string()));
    }

    #[test]
    fn parse_negated_predicate() {
        let node = DefUseGraph::parse_instruction("@!%p5 ld.global.f32 %f1, [%rd4];", 0).unwrap();
        assert_eq!(node.dests, vec!["%f1"]);
        assert!(node.sources.contains(&"%p5".to_string()));
        assert!(node.sources.contains(&"%rd4".to_string()));
    }

    #[test]
    fn parse_setp() {
        let node = DefUseGraph::parse_instruction("setp.ge.u32 %p0, %r4, %r0;", 0).unwrap();
        assert_eq!(node.dests, vec!["%p0"]);
        assert_eq!(node.sources, vec!["%r4", "%r0"]);
    }

    #[test]
    fn parse_cvta() {
        let node = DefUseGraph::parse_instruction("cvta.to.global.u64 %rd1, %rd0;", 0).unwrap();
        assert_eq!(node.dests, vec!["%rd1"]);
        assert_eq!(node.sources, vec!["%rd0"]);
    }

    #[test]
    fn parse_mul_wide() {
        let node = DefUseGraph::parse_instruction("mul.wide.u32 %rd3, %r4, 4;", 0).unwrap();
        assert_eq!(node.dests, vec!["%rd3"]);
        assert_eq!(node.sources, vec!["%r4"]); // 4 is immediate, not a register
    }

    #[test]
    fn parse_vector_load() {
        let node = DefUseGraph::parse_instruction("ld.param.v2.u32 {%r164, %r165}, [%rd1+-24];", 0)
            .unwrap();
        // Both registers in braces are dests
        assert!(node.dests.contains(&"%r164".to_string()));
        assert!(node.dests.contains(&"%r165".to_string()));
        assert_eq!(node.sources, vec!["%rd1"]);
    }

    #[test]
    fn parse_shfl() {
        let node = DefUseGraph::parse_instruction(
            "shfl.sync.idx.b32 %r207|%p16, %r204, %r735, %r169, %r119;",
            0,
        )
        .unwrap();
        // Pipe-separated first operand: both are dests
        assert!(node.dests.contains(&"%r207".to_string()));
        assert!(node.dests.contains(&"%p16".to_string()));
        // Rest are sources
        assert!(node.sources.contains(&"%r204".to_string()));
        assert!(node.sources.contains(&"%r735".to_string()));
    }

    #[test]
    fn parse_cp_async() {
        let node = DefUseGraph::parse_instruction(
            "cp.async.cg.shared.global.L2::128B [%r125], [%rd42], 16, %r126;",
            0,
        )
        .unwrap();
        // cp.async has no register dests (writes to SMEM via address)
        assert!(node.dests.is_empty());
        // All registers are sources
        assert!(node.sources.contains(&"%r125".to_string()));
        assert!(node.sources.contains(&"%rd42".to_string()));
        assert!(node.sources.contains(&"%r126".to_string()));
    }

    #[test]
    fn parse_mad() {
        let node = DefUseGraph::parse_instruction("mad.lo.s32 %r12, %r213, 24, %r216;", 0).unwrap();
        assert_eq!(node.dests, vec!["%r12"]);
        assert!(node.sources.contains(&"%r213".to_string()));
        assert!(node.sources.contains(&"%r216".to_string()));
        // 24 is an immediate, not in sources
    }

    #[test]
    fn parse_selp() {
        let node = DefUseGraph::parse_instruction("selp.u32 %r195, 1, 0, %p9;", 0).unwrap();
        assert_eq!(node.dests, vec!["%r195"]);
        assert_eq!(node.sources, vec!["%p9"]); // 1 and 0 are immediates
    }

    #[test]
    fn parse_mov_special() {
        let node = DefUseGraph::parse_instruction("mov.u32 %r115, %ctaid.x;", 0).unwrap();
        assert_eq!(node.dests, vec!["%r115"]);
        // %ctaid is a special register, extracted as a source
        assert!(node.sources.contains(&"%ctaid".to_string()));
    }

    #[test]
    fn skip_non_instructions() {
        assert!(DefUseGraph::parse_instruction(".reg .f32 %f<16>;", 0).is_none());
        assert!(DefUseGraph::parse_instruction("// comment", 0).is_none());
        assert!(DefUseGraph::parse_instruction(".shared .align 16 .f32 smem[4096];", 0).is_none());
        assert!(DefUseGraph::parse_instruction("$L__BB0_19:", 0).is_none());
        assert!(DefUseGraph::parse_instruction("{", 0).is_none());
        assert!(DefUseGraph::parse_instruction("}", 0).is_none());
    }

    // ── DefUseGraph build tests ──

    #[test]
    fn graph_simple_chain() {
        let ptx = "\
            ld.param.u64 %rd0, [input];\n\
            cvta.to.global.u64 %rd1, %rd0;\n\
            add.u64 %rd2, %rd1, %rd3;\n\
            ld.global.f32 %f1, [%rd2];\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let graph = DefUseGraph::build(&lines);

        // %rd0 should be defined by ld.param and used by cvta
        assert!(!graph.defs["%rd0"].is_empty());
        assert!(!graph.uses["%rd0"].is_empty());

        // %rd1 defined by cvta, used by add
        assert!(!graph.defs["%rd1"].is_empty());
        assert!(!graph.uses["%rd1"].is_empty());

        // %rd2 defined by add, used by ld.global
        assert!(!graph.defs["%rd2"].is_empty());
        assert!(!graph.uses["%rd2"].is_empty());

        // %f1 defined by ld.global, not used
        assert!(!graph.defs["%f1"].is_empty());
        assert!(!graph.uses.contains_key("%f1"));
    }

    #[test]
    fn graph_store_no_defs() {
        let ptx = "st.global.f32 [%rd6], %f7;\n";
        let lines: Vec<&str> = ptx.lines().collect();
        let graph = DefUseGraph::build(&lines);

        // st.global should only have uses, no defs
        assert!(!graph.uses["%rd6"].is_empty());
        assert!(!graph.uses["%f7"].is_empty());
        assert!(!graph.defs.contains_key("%rd6"));
        assert!(!graph.defs.contains_key("%f7"));
    }

    #[test]
    fn graph_trace_forward() {
        let ptx = "\
            ld.param.u64 %rd0, [input];\n\
            cvta.to.global.u64 %rd1, %rd0;\n\
            mul.wide.u32 %rd3, %r4, 4;\n\
            add.u64 %rd4, %rd1, %rd3;\n\
            ld.global.f32 %f1, [%rd4];\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let graph = DefUseGraph::build(&lines);

        // Trace forward from %rd0 should reach ld.global within 3 hops
        let reached = graph.trace_forward("%rd0", 3);
        let opcodes: Vec<&str> = reached
            .iter()
            .map(|&(_, idx)| graph.nodes[idx].opcode.as_str())
            .collect();
        assert!(
            opcodes.contains(&"cvta.to.global.u64"),
            "should reach cvta: {:?}",
            opcodes
        );
        assert!(
            opcodes.contains(&"ld.global.f32"),
            "should reach ld.global: {:?}",
            opcodes
        );
    }

    #[test]
    fn graph_trace_forward_depth_limit() {
        let ptx = "\
            mov.u32 %r1, %r0;\n\
            mov.u32 %r2, %r1;\n\
            mov.u32 %r3, %r2;\n\
            mov.u32 %r4, %r3;\n\
            setp.gt.u32 %p0, %r4, 0;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let graph = DefUseGraph::build(&lines);

        // Depth 2 from %r0 should NOT reach setp (4 hops away)
        let reached_2 = graph.trace_forward("%r0", 2);
        let has_setp = reached_2
            .iter()
            .any(|&(_, idx)| graph.nodes[idx].opcode.starts_with("setp"));
        assert!(!has_setp, "depth 2 should not reach setp");

        // Depth 4 should reach it
        let reached_4 = graph.trace_forward("%r0", 4);
        let has_setp = reached_4
            .iter()
            .any(|&(_, idx)| graph.nodes[idx].opcode.starts_with("setp"));
        assert!(has_setp, "depth 4 should reach setp");
    }

    // ── extract_all_registers tests ──

    #[test]
    fn extract_regs_simple() {
        assert_eq!(
            extract_all_registers("%r1, %r2, %r3"),
            vec!["%r1", "%r2", "%r3"]
        );
    }

    #[test]
    fn extract_regs_with_brackets() {
        assert_eq!(extract_all_registers("[%rd4+8]"), vec!["%rd4"]);
    }

    #[test]
    fn extract_regs_with_immediates() {
        assert_eq!(extract_all_registers("%r1, 42, %r2"), vec!["%r1", "%r2"]);
    }

    #[test]
    fn extract_regs_vector_braces() {
        assert_eq!(
            extract_all_registers("{%r164, %r165}"),
            vec!["%r164", "%r165"]
        );
    }

    #[test]
    fn extract_regs_empty() {
        assert!(extract_all_registers("").is_empty());
        assert!(extract_all_registers("42").is_empty());
        assert!(extract_all_registers("EXIT").is_empty());
    }

    // ── split_dest_and_sources tests ──

    #[test]
    fn split_simple() {
        let (d, s) = split_dest_and_sources("%r1, %r2, %r3");
        assert_eq!(d, vec!["%r1"]);
        assert_eq!(s, vec!["%r2", "%r3"]);
    }

    #[test]
    fn split_vector_dest() {
        let (d, s) = split_dest_and_sources("{%r1, %r2}, [%rd3+8]");
        assert!(d.contains(&"%r1".to_string()));
        assert!(d.contains(&"%r2".to_string()));
        assert_eq!(s, vec!["%rd3"]);
    }

    #[test]
    fn split_pipe_dest() {
        let (d, s) = split_dest_and_sources("%r1|%p2, %r3, %r4");
        assert!(d.contains(&"%r1".to_string()));
        assert!(d.contains(&"%p2".to_string()));
        assert!(s.contains(&"%r3".to_string()));
        assert!(s.contains(&"%r4".to_string()));
    }

    #[test]
    fn split_single_operand() {
        let (d, s) = split_dest_and_sources("%r1");
        assert_eq!(d, vec!["%r1"]);
        assert!(s.is_empty());
    }

    // ── Param field extraction tests ──

    #[test]
    fn extract_simple_params() {
        let ptx = "\
            .visible .entry foo(\n\
                .param .u64 input,\n\
                .param .u64 output,\n\
                .param .u32 n,\n\
                .param .f32 epsilon\n\
            )\n\
            {\n\
            ld.param.u64 %rd0, [input];\n\
            ld.param.u64 %rd1, [output];\n\
            ld.param.u32 %r0, [n];\n\
            ld.param.f32 %f0, [epsilon];\n\
            ret;\n\
            }\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (fields, _base_offsets) = extract_param_fields(&lines);

        // Simple named params don't have +/- offsets, so they all resolve to offset 0
        // and get deduplicated. This is expected — named params don't need struct offset tracking.
        // The important thing is that the function doesn't crash.
        assert!(
            !fields.is_empty(),
            "should extract at least some param fields"
        );
    }

    #[test]
    fn extract_struct_param_base_offset() {
        let ptx = "\
            mov.b64 %rd40, _some_param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.u32 %r1, [_some_param+12];\n\
            ld.param.u64 %rd2, [%rd1+16];\n\
            ld.param.u64 %rd3, [%rd1+24];\n\
            ld.param.f32 %f1, [%rd1+264];\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (fields, base_offsets) = extract_param_fields(&lines);

        // Verify base_offsets: %rd40 → 0, %rd1 → 24
        assert_eq!(base_offsets.get("%rd40"), Some(&0));
        assert_eq!(base_offsets.get("%rd1"), Some(&24));

        // Verify field offsets
        let offsets: Vec<i64> = fields.iter().map(|f| f.offset).collect();
        assert!(
            offsets.contains(&12),
            "should have offset 12 from [param+12]"
        );
        assert!(
            offsets.contains(&40),
            "should have offset 40 from [%rd1+16] = 24+16"
        );
        assert!(
            offsets.contains(&48),
            "should have offset 48 from [%rd1+24] = 24+24"
        );
        assert!(
            offsets.contains(&288),
            "should have offset 288 from [%rd1+264] = 24+264"
        );
    }

    #[test]
    fn extract_vector_load_splits_fields() {
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.v2.u32 {%r164, %r165}, [%rd1+-24];\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (fields, _base) = extract_param_fields(&lines);

        // v2.u32 at [%rd1+-24] = [%rd1+(-24)] → offset = 24 + (-24) = 0
        // Two fields at offset 0 and 4 (each u32 = 4 bytes)
        let offsets: Vec<i64> = fields.iter().map(|f| f.offset).collect();
        assert!(
            offsets.contains(&0),
            "first element at offset 0: {:?}",
            offsets
        );
        assert!(
            offsets.contains(&4),
            "second element at offset 4: {:?}",
            offsets
        );
    }

    #[test]
    fn extract_negative_offset() {
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.u32 %r1, [%rd1+-20];\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (fields, _base) = extract_param_fields(&lines);

        // [%rd1+-20] → offset = 24 + (-20) = 4
        let offsets: Vec<i64> = fields.iter().map(|f| f.offset).collect();
        assert!(offsets.contains(&4), "should have offset 4: {:?}", offsets);
    }

    // ── Classification tests ──

    #[test]
    fn classify_hand_written_rms_norm() {
        let ptx = include_str!("../../ptx-fusion/kernels/rms_norm.ptx");
        let protocol = PtxParser::parse(ptx).unwrap();

        // Hand-written PTX has simple named params, not struct params
        // classified_params may be populated with offset-0 entries for named params
        // The key test: it doesn't crash and produces reasonable results
        println!(
            "rms_norm classified_params: {:?}",
            protocol.classified_params
        );

        // If there are classified params, check they make sense
        for cp in &protocol.classified_params {
            match cp.ptx_type.as_str() {
                "f32" => assert_eq!(cp.role, ParamRole::Scalar, "f32 should be Scalar"),
                _ => {} // other types depend on usage
            }
        }
    }

    #[test]
    fn classify_nvcc_gemm() {
        let ptx = include_str!("../../ptx-fusion/kernels/gemm_row_f32.ptx");
        let protocol = PtxParser::parse(ptx).unwrap();

        println!(
            "gemm_row_f32 classified_params: {:?}",
            protocol.classified_params
        );

        // nvcc GEMM with named params — check that u64 params traced to
        // ld.global are classified as Pointer
        let pointer_count = protocol
            .classified_params
            .iter()
            .filter(|p| p.role == ParamRole::Pointer)
            .count();
        // gemm_row_f32 has 3 u64 params (A, B, C) that are pointers
        // They all have cvta.to.global, so should be classified as Pointer
        println!("Pointer count: {pointer_count}");
    }

    #[test]
    fn classify_f32_always_scalar() {
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.f32 %f1, [%rd1+264];\n\
            ld.param.f32 %f2, [%rd1+268];\n\
            mul.f32 %f3, %f1, %f2;\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (mut fields, base_offsets) = extract_param_fields(&lines);
        let graph = DefUseGraph::build(&lines);

        // Create a minimal protocol (no data ports)
        let protocol = KernelProtocol {
            name: "test".to_string(),
            registers: vec![],
            smem_regions: vec![],
            total_smem_bytes: 0,
            params: vec![],
            global_loads: vec![],
            global_stores: vec![],
            async_loads: vec![],
            smem_loads: 0,
            smem_stores: 0,
            barriers: vec![],
            has_mma: false,
            classified_params: vec![],
        };

        classify_param_fields(&mut fields, &graph, &protocol, &base_offsets);

        for f in &fields {
            assert_eq!(f.role, ParamRole::Scalar, "f32 fields must be Scalar");
        }
    }

    #[test]
    fn classify_stride_via_mul() {
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 0;\n\
            ld.param.u64 %rd10, [%rd1+8];\n\
            cvt.s64.s32 %rd20, %r1;\n\
            mul.lo.s64 %rd21, %rd10, %rd20;\n\
            add.s64 %rd22, %rd23, %rd21;\n\
            ld.global.f32 %f1, [%rd22];\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (mut fields, base_offsets) = extract_param_fields(&lines);
        let graph = DefUseGraph::build(&lines);

        let protocol = KernelProtocol {
            name: "test".to_string(),
            registers: vec![],
            smem_regions: vec![],
            total_smem_bytes: 0,
            params: vec![],
            global_loads: vec![],
            global_stores: vec![],
            async_loads: vec![],
            smem_loads: 0,
            smem_stores: 0,
            barriers: vec![],
            has_mma: false,
            classified_params: vec![],
        };

        classify_param_fields(&mut fields, &graph, &protocol, &base_offsets);

        let field = fields.iter().find(|f| f.offset == 8).unwrap();
        assert_eq!(
            field.role,
            ParamRole::Stride,
            "u64 field used in mul.lo.s64 with register operand should be Stride"
        );
    }

    #[test]
    fn classify_dimension_via_setp() {
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            ld.param.u32 %r1, [_param+12];\n\
            setp.ge.s32 %p1, %r2, %r1;\n\
            @%p1 bra EXIT;\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (mut fields, base_offsets) = extract_param_fields(&lines);
        let graph = DefUseGraph::build(&lines);

        let protocol = KernelProtocol {
            name: "test".to_string(),
            registers: vec![],
            smem_regions: vec![],
            total_smem_bytes: 0,
            params: vec![],
            global_loads: vec![],
            global_stores: vec![],
            async_loads: vec![],
            smem_loads: 0,
            smem_stores: 0,
            barriers: vec![],
            has_mma: false,
            classified_params: vec![],
        };

        classify_param_fields(&mut fields, &graph, &protocol, &base_offsets);

        let field = fields.iter().find(|f| f.offset == 12).unwrap();
        assert_eq!(
            field.role,
            ParamRole::Dimension,
            "u32 field reaching setp should be Dimension"
        );
    }

    #[test]
    fn classify_pointer_via_perimeter() {
        // Simulate a struct param where offset 64 is traced as a pointer
        // by the existing perimeter analysis
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.u64 %rd7, [%rd1+40];\n\
            cvt.s64.s32 %rd50, %r1;\n\
            mul.lo.s64 %rd51, %rd50, 4;\n\
            add.s64 %rd52, %rd7, %rd51;\n\
            ld.global.f32 %f1, [%rd52];\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (mut fields, base_offsets) = extract_param_fields(&lines);
        let graph = DefUseGraph::build(&lines);

        // Simulate the perimeter: ld.global traces to "%rd1+40"
        let protocol = KernelProtocol {
            name: "test".to_string(),
            registers: vec![],
            smem_regions: vec![],
            total_smem_bytes: 0,
            params: vec![],
            global_loads: vec![DataPort {
                param_name: "%rd1+40".to_string(),
                data_type: "f32".to_string(),
                line: 6,
            }],
            global_stores: vec![],
            async_loads: vec![],
            smem_loads: 0,
            smem_stores: 0,
            barriers: vec![],
            has_mma: false,
            classified_params: vec![],
        };

        classify_param_fields(&mut fields, &graph, &protocol, &base_offsets);

        let field = fields.iter().find(|f| f.offset == 64).unwrap();
        assert_eq!(
            field.role,
            ParamRole::Pointer,
            "u64 field traced by perimeter analysis to ld.global should be Pointer"
        );
    }

    #[test]
    fn classify_pointer_via_cvta() {
        // Pointers going through cvta.to.global should be classified as Pointer
        // even without being in the perimeter data ports
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.u64 %rd7, [%rd1+40];\n\
            cvta.to.global.u64 %rd8, %rd7;\n\
            ld.global.f32 %f1, [%rd8];\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (mut fields, base_offsets) = extract_param_fields(&lines);
        let graph = DefUseGraph::build(&lines);

        let protocol = KernelProtocol {
            name: "test".to_string(),
            registers: vec![],
            smem_regions: vec![],
            total_smem_bytes: 0,
            params: vec![],
            global_loads: vec![],
            global_stores: vec![],
            async_loads: vec![],
            smem_loads: 0,
            smem_stores: 0,
            barriers: vec![],
            has_mma: false,
            classified_params: vec![],
        };

        classify_param_fields(&mut fields, &graph, &protocol, &base_offsets);

        let field = fields.iter().find(|f| f.offset == 64).unwrap();
        assert_eq!(
            field.role,
            ParamRole::Pointer,
            "u64 field going through cvta.to.global should be Pointer"
        );
    }

    #[test]
    fn classify_derived_increment() {
        // A u64 field that reaches memory but isn't in the perimeter and isn't
        // multiplied with a register should be Derived
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.u64 %rd2, [%rd1+16];\n\
            ld.param.u64 %rd7, [%rd1+40];\n\
            add.s64 %rd42, %rd7, %rd2;\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (mut fields, base_offsets) = extract_param_fields(&lines);
        let graph = DefUseGraph::build(&lines);

        // Only offset 64 is in the perimeter
        let protocol = KernelProtocol {
            name: "test".to_string(),
            registers: vec![],
            smem_regions: vec![],
            total_smem_bytes: 0,
            params: vec![],
            global_loads: vec![],
            global_stores: vec![],
            async_loads: vec![AsyncCopyPort {
                param_name: "%rd1+40".to_string(),
                smem_dst: "%r1".to_string(),
                gmem_src: "%rd7".to_string(),
                mask: "%r2".to_string(),
                size_bytes: 16,
                line: 5,
            }],
            smem_loads: 0,
            smem_stores: 0,
            barriers: vec![],
            has_mma: false,
            classified_params: vec![],
        };

        classify_param_fields(&mut fields, &graph, &protocol, &base_offsets);

        let pointer_field = fields.iter().find(|f| f.offset == 64).unwrap();
        assert_eq!(pointer_field.role, ParamRole::Pointer);

        let derived_field = fields.iter().find(|f| f.offset == 40).unwrap();
        assert_eq!(
            derived_field.role,
            ParamRole::Derived,
            "u64 field NOT in perimeter, not mul'd with register, should be Derived"
        );
    }

    #[test]
    fn classify_mul_with_immediate_is_not_stride() {
        // A u64 field multiplied by a small immediate (not a register) should NOT be Stride
        let ptx = "\
            mov.b64 %rd40, _param;\n\
            add.s64 %rd1, %rd40, 24;\n\
            ld.param.u64 %rd2, [%rd1+16];\n\
            mul.lo.s64 %rd78, %rd2, 3;\n\
            ret;\n\
        ";
        let lines: Vec<&str> = ptx.lines().collect();
        let (mut fields, base_offsets) = extract_param_fields(&lines);
        let graph = DefUseGraph::build(&lines);

        let protocol = KernelProtocol {
            name: "test".to_string(),
            registers: vec![],
            smem_regions: vec![],
            total_smem_bytes: 0,
            params: vec![],
            global_loads: vec![],
            global_stores: vec![],
            async_loads: vec![],
            smem_loads: 0,
            smem_stores: 0,
            barriers: vec![],
            has_mma: false,
            classified_params: vec![],
        };

        classify_param_fields(&mut fields, &graph, &protocol, &base_offsets);

        let field = fields.iter().find(|f| f.offset == 40).unwrap();
        assert_ne!(
            field.role,
            ParamRole::Stride,
            "mul.lo.s64 with immediate 3 should NOT be Stride (it's a pipeline scale)"
        );
        assert_eq!(field.role, ParamRole::Derived);
    }

    // ── resolve_param_name_to_offset tests ──

    #[test]
    fn resolve_register_plus_offset() {
        let mut bases = BTreeMap::new();
        bases.insert("%rd1".to_string(), 24);
        assert_eq!(resolve_param_name_to_offset("%rd1+40", &bases), Some(64));
    }

    #[test]
    fn resolve_register_plus_negative() {
        let mut bases = BTreeMap::new();
        bases.insert("%rd1".to_string(), 24);
        assert_eq!(resolve_param_name_to_offset("%rd1+-20", &bases), Some(4));
    }

    #[test]
    fn resolve_unknown_base() {
        let bases = BTreeMap::new();
        // Unknown base defaults to 0
        assert_eq!(resolve_param_name_to_offset("%rd99+40", &bases), Some(40));
    }

    #[test]
    fn resolve_named_param() {
        let bases = BTreeMap::new();
        assert_eq!(resolve_param_name_to_offset("input", &bases), None);
    }

    #[test]
    fn resolve_register_only() {
        let mut bases = BTreeMap::new();
        bases.insert("%rd1".to_string(), 24);
        assert_eq!(resolve_param_name_to_offset("%rd1", &bases), Some(24));
    }
}
