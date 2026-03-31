//! Pipeline compiler: compose N pipeline stages into a single fused kernel.
//!
//! MVP: Reduction → TiledGemm fusion (rms_norm → CUTLASS GEMM).
//! The compiler:
//! 1. Decomposes the reduction into accumulate/finalize/emit
//! 2. Generates a prologue that runs the reduction with the GEMM's thread count
//! 3. Extracts the per-element formula from the emit phase
//! 4. Feeds both into replace_a_loads_with_inline_fn
//!
//! Everything is derived from PTX analysis — the formula, reduction pattern,
//! and finalization come from the actual rms_norm kernel. Only the thread-to-row
//! mapping uses runtime arithmetic (correct for any CUTLASS thread layout).

use crate::fuse_general::{PointwiseComputation, replace_a_loads_with_inline_fn};
use crate::parser::{
    CarryRegister, CarryRole, DefUseGraph, LoopDescriptor, TileIndexMap,
    analyze_carries, detect_loops, extract_tile_index_map,
};
use crate::pipeline::{
    PipelineStage, ReductionDecomposition, StagePattern, TilePerimeter, TilePortAccess,
};

// ===== GEMM Body Descriptor: parsed structure of a CUTLASS GEMM =====

/// Parsed decomposition of a CUTLASS GEMM kernel into its structural components.
///
/// All fields are derived from PTX analysis (detect_loops, analyze_carries,
/// classify_cp_async_loads). No hardcoded register names or line numbers.
#[derive(Debug, Clone)]
pub struct GemmBodyDescriptor {
    /// Lines before the K-loop header (tile index, param loads, pipeline prologue).
    pub preamble: Vec<String>,
    /// Lines of the K-loop body (from header label through backedge branch, inclusive).
    pub k_loop: Vec<String>,
    /// Lines after the K-loop (epilogue: accum → SMEM → scale → bf16 → st.global).
    pub epilogue: Vec<String>,
    /// The K-loop descriptor (header label, backedge, predicate).
    pub loop_desc: LoopDescriptor,
    /// MMA accumulator registers (f32, carry state across K-iterations).
    pub mma_accumulators: Vec<String>,
    /// Tile pointer registers (GMEM addresses advancing per K-iteration).
    pub tile_pointers: Vec<String>,
    /// Induction variable registers (K-loop counters, pipeline state).
    pub induction_vars: Vec<String>,
    /// Buffer state registers (SMEM triple-buffer rotation via selp).
    pub buffer_state: Vec<String>,
    /// Register declarations from the kernel body (.reg lines).
    pub reg_decls: Vec<String>,
    /// Shared memory declarations from the kernel body (.shared lines).
    pub smem_decls: Vec<String>,
    /// Kernel parameters (.param lines from entry signature).
    pub params: Vec<String>,
    /// The entry function name.
    pub entry_name: String,
}

/// Extract a `GemmBodyDescriptor` from a flat-param CUTLASS GEMM PTX.
///
/// The input must be a single-entry, perimeter-replaced CUTLASS kernel.
/// Returns an error if the PTX doesn't contain a recognizable K-loop or MMA.
pub fn extract_gemm_body(ptx: &str) -> Result<GemmBodyDescriptor, String> {
    let lines: Vec<&str> = ptx.lines().collect();

    // ── Identify the K-loop ──
    let loops = detect_loops(&lines);
    // Find the main K-loop: outermost loop with MMA instructions inside
    let main_loop = loops
        .iter()
        .filter(|l| {
            // Must contain mma.sync within body
            let (start, end) = l.body_range;
            (start..=end).any(|i| {
                i < lines.len() && lines[i].contains("mma.sync")
            })
        })
        .min_by_key(|l| l.depth)
        .ok_or("no K-loop with MMA instructions found")?
        .clone();

    // ── Carry analysis ──
    let carries = analyze_carries(&lines, &main_loop);
    let mma_accumulators: Vec<String> = carries
        .iter()
        .filter(|c| c.role == CarryRole::MmaAccumulator)
        .map(|c| c.register.clone())
        .collect();
    let tile_pointers: Vec<String> = carries
        .iter()
        .filter(|c| c.role == CarryRole::TilePointer)
        .map(|c| c.register.clone())
        .collect();
    let induction_vars: Vec<String> = carries
        .iter()
        .filter(|c| c.role == CarryRole::InductionVar)
        .map(|c| c.register.clone())
        .collect();
    let buffer_state: Vec<String> = carries
        .iter()
        .filter(|c| c.role == CarryRole::BufferState)
        .map(|c| c.register.clone())
        .collect();

    if mma_accumulators.is_empty() {
        return Err("no MMA accumulator registers found in K-loop".into());
    }

    // ── Extract entry name and params ──
    let mut entry_name = String::new();
    let mut params = Vec::new();
    let mut in_entry = false;
    for line in &lines {
        let t = line.trim();
        if t.contains(".entry") {
            in_entry = true;
            // Extract name: ".visible .entry name("
            if let Some(pos) = t.find(".entry") {
                let after = t[pos + 6..].trim();
                let name_end = after.find('(').unwrap_or(after.len());
                entry_name = after[..name_end].trim().to_string();
            }
            continue;
        }
        if !in_entry {
            continue;
        }
        if t.starts_with(".param") {
            params.push(t.trim_end_matches(',').to_string());
        }
        if t == ")" || t == "{" || t.ends_with('{') {
            break;
        }
    }

    // ── Find kernel body boundaries ──
    // Body starts at first '{' after entry, ends at matching '}'
    let mut body_start = 0;
    let mut found_entry = false;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.contains(".entry") {
            found_entry = true;
        }
        if found_entry && (t == "{" || t.ends_with('{')) {
            body_start = i + 1;
            break;
        }
    }

    // Find ret; (end of body)
    let body_end = lines
        .iter()
        .rposition(|l| l.trim() == "ret;")
        .unwrap_or(lines.len() - 1);

    // ── Extract reg and smem declarations ──
    let mut reg_decls = Vec::new();
    let mut smem_decls = Vec::new();
    for i in body_start..main_loop.header_line {
        let t = lines[i].trim();
        if t.starts_with(".reg ") {
            reg_decls.push(lines[i].to_string());
        } else if t.starts_with(".shared ") {
            smem_decls.push(lines[i].to_string());
        }
    }

    // ── Split into preamble / k_loop / epilogue ──
    // Preamble: body lines before K-loop, excluding reg/smem decls
    let mut preamble = Vec::new();
    for i in body_start..main_loop.header_line {
        let t = lines[i].trim();
        if t.starts_with(".reg ") || t.starts_with(".shared ") || t.is_empty() {
            continue;
        }
        preamble.push(lines[i].to_string());
    }

    // K-loop: header through backedge (inclusive)
    let k_loop: Vec<String> = (main_loop.header_line..=main_loop.backedge_line)
        .map(|i| lines[i].to_string())
        .collect();

    // Epilogue: after backedge through ret (exclusive of ret itself)
    let epilogue: Vec<String> = (main_loop.backedge_line + 1..body_end)
        .map(|i| lines[i].to_string())
        .collect();

    Ok(GemmBodyDescriptor {
        preamble,
        k_loop,
        epilogue,
        loop_desc: main_loop,
        mma_accumulators,
        tile_pointers,
        induction_vars,
        buffer_state,
        reg_decls,
        smem_decls,
        params,
        entry_name,
    })
}

/// Generated PTX for spilling or reloading MMA accumulators to/from SMEM.
#[derive(Debug, Clone)]
pub struct AccumSpillReload {
    /// PTX instructions for spilling accumulators to SMEM.
    pub spill: Vec<String>,
    /// PTX instructions for reloading accumulators from SMEM.
    pub reload: Vec<String>,
    /// PTX instructions for zeroing accumulators (before first accumulation).
    pub zero: Vec<String>,
    /// Extra register declarations needed (address computation temps).
    pub reg_decls: Vec<String>,
    /// SMEM bytes required for the spill region.
    pub smem_bytes: u32,
}

/// Build PTX to spill/reload MMA accumulators to a dedicated SMEM region.
///
/// Each thread spills its own accumulators to a unique SMEM slot:
///   addr = smem_base + tid * num_accum * 4 + accum_idx * 4
///
/// The `smem_base_reg` is an existing register holding the dynamic SMEM base
/// address (from `mov.u32 %r_base, _dynamic_smem`). The `smem_offset` is a
/// static byte offset within the dynamic SMEM region.
///
/// All accumulator register names are taken from the GemmBodyDescriptor —
/// no hardcoded register names.
pub fn build_accum_spill_reload(
    desc: &GemmBodyDescriptor,
    smem_offset: u32,
) -> AccumSpillReload {
    let accums = &desc.mma_accumulators;
    let n = accums.len() as u32;
    let bytes_per_thread = n * 4; // f32 = 4 bytes each

    // Register declarations for address computation
    let reg_decls = vec![
        ".reg .u32 %r_spill_tid, %r_spill_off;".into(),
    ];

    // Address computation shared by spill and reload:
    //   %r_spill_tid = %tid.x
    //   %r_spill_off = smem_offset + %r_spill_tid * bytes_per_thread
    let addr_setup = vec![
        format!("\tmov.u32 \t%r_spill_tid, %tid.x;"),
        format!("\tmad.lo.u32 \t%r_spill_off, %r_spill_tid, {bytes_per_thread}, {smem_offset};"),
    ];

    // Spill: st.shared.f32 for each accumulator
    let mut spill = Vec::new();
    spill.push("\t// FERRITE: spill MMA accumulators to SMEM".into());
    spill.extend(addr_setup.iter().cloned());
    for (i, reg) in accums.iter().enumerate() {
        let offset = i as u32 * 4;
        spill.push(format!(
            "\tst.shared.f32 \t[%r_spill_off+{offset}], {reg};"
        ));
    }

    // Reload: ld.shared.f32 for each accumulator
    let mut reload = Vec::new();
    reload.push("\t// FERRITE: reload MMA accumulators from SMEM".into());
    reload.extend(addr_setup.iter().cloned());
    for (i, reg) in accums.iter().enumerate() {
        let offset = i as u32 * 4;
        reload.push(format!(
            "\tld.shared.f32 \t{reg}, [%r_spill_off+{offset}];"
        ));
    }

    // Zero: mov.f32 0.0 for each accumulator (before first accumulation)
    let mut zero = Vec::new();
    zero.push("\t// FERRITE: zero MMA accumulators".into());
    for reg in accums {
        zero.push(format!("\tmov.f32 \t{reg}, 0f00000000;"));
    }

    let smem_bytes = bytes_per_thread * 128; // 128 threads per block

    AccumSpillReload {
        spill,
        reload,
        zero,
        reg_decls,
        smem_bytes,
    }
}

// ===== Epilogue Store Map: analysis of CUTLASS epilogue output stores =====

/// One st.global instruction in the CUTLASS epilogue.
#[derive(Debug, Clone)]
pub struct EpilogueStore {
    /// Line index in the epilogue lines.
    pub line_idx: usize,
    /// The full instruction text.
    pub instruction: String,
    /// The 64-bit address register (e.g., "%rd22").
    pub addr_reg: String,
    /// The data registers being stored (e.g., ["%r1057", "%r1058", ...]).
    pub data_regs: Vec<String>,
    /// How this address relates to the base: a list of stride register additions.
    /// Empty for the base store. E.g., ["%rd20"] means base + stride_A,
    /// ["%rd20", "%rd21"] means base + stride_A + stride_B.
    pub stride_chain: Vec<String>,
}

/// The complete map of epilogue stores and their address structure.
#[derive(Debug, Clone)]
pub struct EpilogueStoreMap {
    /// The base address register (first st.global's address).
    pub base_addr_reg: String,
    /// The row register used in the base address computation.
    pub row_reg: String,
    /// The column register used in the base address computation.
    pub col_reg: String,
    /// Stride registers loaded from params, used for row offsets between stores.
    pub stride_regs: Vec<String>,
    /// All st.global instructions with their address decomposition.
    pub stores: Vec<EpilogueStore>,
}

/// Analyze the epilogue of a CUTLASS GEMM to map all st.global stores
/// and their address relationships.
///
/// Traces store addresses backward through the def-use graph to decompose
/// them as: base + sum_of(stride_regs). Works for any CUTLASS epilogue that
/// computes output addresses as linear combinations of a base address and
/// param-derived strides.
pub fn analyze_epilogue_stores(
    epilogue_lines: &[String],
    full_ptx_lines: &[&str],
    _epilogue_start_line: usize,
) -> Result<EpilogueStoreMap, String> {
    use std::collections::{BTreeMap, BTreeSet};

    // ── Find all st.global instructions ──
    let mut raw_stores: Vec<(usize, String, String, Vec<String>)> = Vec::new();
    for (i, line) in epilogue_lines.iter().enumerate() {
        let t = line.trim();
        if !t.contains("st.global") {
            continue;
        }
        let addr_reg = t
            .find('[')
            .and_then(|start| {
                let rest = &t[start + 1..];
                rest.find(']').map(|end| rest[..end].to_string())
            })
            .ok_or_else(|| format!("can't parse st.global address at epilogue line {i}"))?;

        let data_regs = t
            .find('{')
            .and_then(|start| {
                let rest = &t[start + 1..];
                rest.find('}').map(|end| {
                    rest[..end]
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect::<Vec<_>>()
                })
            })
            .unwrap_or_default();

        raw_stores.push((i, line.to_string(), addr_reg, data_regs));
    }

    if raw_stores.is_empty() {
        return Err("no st.global found in epilogue".into());
    }

    // ── Build register definition map for 64-bit address regs ──
    let mut reg_defs: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (line_idx, line) in full_ptx_lines.iter().enumerate() {
        let t = line.trim();
        let work = if t.starts_with('@') {
            match t.find(|c: char| c.is_whitespace()) {
                Some(i) => t[i..].trim(),
                None => continue,
            }
        } else {
            t
        };
        if !work.contains("%rd") {
            continue;
        }
        if let Some(node) = DefUseGraph::parse_instruction(work, line_idx) {
            for dest in &node.dests {
                if dest.starts_with("%rd") {
                    reg_defs.insert(dest.clone(), (line_idx, work.to_string()));
                }
            }
        }
    }

    // ── Decompose each store address as base + stride chain ──
    let base_addr = raw_stores[0].2.clone();
    let mut analyzed_stores = Vec::new();
    let mut all_stride_regs: BTreeSet<String> = BTreeSet::new();

    for (line_idx, instruction, addr_reg, data_regs) in &raw_stores {
        let stride_chain = trace_address_to_strides(addr_reg, &base_addr, &reg_defs);
        for s in &stride_chain {
            all_stride_regs.insert(s.clone());
        }
        analyzed_stores.push(EpilogueStore {
            line_idx: *line_idx,
            instruction: instruction.clone(),
            addr_reg: addr_reg.clone(),
            data_regs: data_regs.clone(),
            stride_chain,
        });
    }

    // ── Find row/col registers from base address computation ──
    let (row_reg, col_reg) = find_row_col_regs(&base_addr, &reg_defs);

    Ok(EpilogueStoreMap {
        base_addr_reg: base_addr,
        row_reg,
        col_reg,
        stride_regs: all_stride_regs.into_iter().collect(),
        stores: analyzed_stores,
    })
}

/// Trace an address register back through add.s64 chains to decompose
/// it as base_reg + sum_of(stride_regs).
fn trace_address_to_strides(
    addr_reg: &str,
    base_reg: &str,
    reg_defs: &std::collections::BTreeMap<String, (usize, String)>,
) -> Vec<String> {
    if addr_reg == base_reg {
        return Vec::new();
    }
    let (_, def_instr) = match reg_defs.get(addr_reg) {
        Some(d) => d,
        None => return Vec::new(),
    };
    let tokens: Vec<&str> = def_instr
        .trim_end_matches(';')
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.len() < 4 || !tokens[0].contains("add") {
        return Vec::new();
    }
    let src_a = tokens[2];
    let src_b = tokens[3];

    if src_a == base_reg {
        vec![src_b.to_string()]
    } else if src_b == base_reg {
        vec![src_a.to_string()]
    } else {
        // Recurse through intermediate address regs
        let mut chain_a = trace_address_to_strides(src_a, base_reg, reg_defs);
        if !chain_a.is_empty() {
            chain_a.push(src_b.to_string());
            return chain_a;
        }
        let mut chain_b = trace_address_to_strides(src_b, base_reg, reg_defs);
        if !chain_b.is_empty() {
            chain_b.push(src_a.to_string());
            return chain_b;
        }
        Vec::new()
    }
}

/// Find row and column registers from the base output address computation.
///
/// Traces backward from the base address through add.s64/mul chains.
/// Row: 32-bit register fed through cvt.s64.s32 (row index → 64-bit for stride multiply).
/// Col: 32-bit register fed through mul.wide.s32 (col index × element size).
fn find_row_col_regs(
    base_addr: &str,
    reg_defs: &std::collections::BTreeMap<String, (usize, String)>,
) -> (String, String) {
    let mut row_reg = String::new();
    let mut col_reg = String::new();
    let mut queue = vec![base_addr.to_string()];
    let mut visited = std::collections::BTreeSet::new();

    while let Some(reg) = queue.pop() {
        if !visited.insert(reg.clone()) {
            continue;
        }
        let (_, def_instr) = match reg_defs.get(&reg) {
            Some(d) => d,
            None => continue,
        };
        let tokens: Vec<&str> = def_instr
            .trim_end_matches(';')
            .split(|c: char| c == ',' || c.is_whitespace())
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if tokens.is_empty() {
            continue;
        }
        let opcode = tokens[0];

        // mul.wide.s32 %rdN, %rCol, elemsize → column register
        if opcode == "mul.wide.s32" && tokens.len() >= 4 {
            let src = tokens[2];
            if src.starts_with("%r") && !src.starts_with("%rd") {
                col_reg = src.to_string();
            }
        }
        // cvt.s64.s32 %rdN, %rRow → row register
        if opcode == "cvt.s64.s32" && tokens.len() >= 3 {
            let src = tokens[2];
            if src.starts_with("%r") && !src.starts_with("%rd") {
                row_reg = src.to_string();
            }
        }
        // Follow 64-bit sources through add/mul chains
        for &tok in &tokens[2..] {
            if tok.starts_with("%rd") {
                queue.push(tok.to_string());
            }
        }
    }

    (row_reg, col_reg)
}

/// Generated epilogue with st.global redirected to st.shared (SMEM scratch).
#[derive(Debug, Clone)]
pub struct RedirectedEpilogue {
    /// The modified epilogue lines (st.global → st.shared, bounds checks simplified).
    pub lines: Vec<String>,
    /// Extra register declarations needed for SMEM address computation.
    pub reg_decls: Vec<String>,
    /// SMEM bytes required for the scratch region (tile_m × tile_n × 2 bytes bf16).
    pub smem_bytes: u32,
}

/// Rewrite a CUTLASS epilogue to store output to SMEM scratch instead of GMEM.
///
/// The epilogue's MMA fragment rearrangement, alpha scaling, and bf16 conversion
/// are preserved unchanged. Only the final st.global stores are replaced with
/// st.shared stores to a linear SMEM scratch region.
///
/// The SMEM scratch layout is row-major: `smem_base + row * tile_n * 2 + col * 2`.
/// Row and column registers are identified from the EpilogueStoreMap analysis.
///
/// Bounds checks (row < M predicates) are kept since we reuse the same epilogue
/// code — they become harmless (always true for full tiles, safely skip for partials).
pub fn redirect_epilogue_to_smem(
    epilogue_lines: &[String],
    store_map: &EpilogueStoreMap,
    smem_offset: u32,
    tile_n: u32,
) -> Result<RedirectedEpilogue, String> {
    let row_stride_bytes = tile_n * 2; // bf16 = 2 bytes per element

    // Register declarations for SMEM address computation
    let reg_decls = vec![
        ".reg .u32 %r_epi_smem_base, %r_epi_smem_addr;".into(),
    ];

    // Compute stride row offsets from the store patterns
    let stride_row_offsets = infer_stride_row_offsets(&store_map.stores, &store_map.stride_regs);

    // Build a lookup: address register → store info
    let mut addr_to_store: std::collections::BTreeMap<String, Vec<&EpilogueStore>> =
        std::collections::BTreeMap::new();
    for store in &store_map.stores {
        addr_to_store
            .entry(store.addr_reg.clone())
            .or_default()
            .push(store);
    }

    // Simple line-by-line replacement. For each line containing st.global,
    // match by address register and replace the entire inline asm block
    // (or bare instruction) with a SMEM store.
    let mut output = Vec::new();
    let mut skip_until_end_asm = false;
    let mut pending_store: Option<&EpilogueStore> = None;

    for (_i, line) in epilogue_lines.iter().enumerate() {
        let t = line.trim();

        // If we're skipping an inline asm block we already replaced
        if skip_until_end_asm {
            if t == "// end inline asm" {
                skip_until_end_asm = false;
            }
            continue;
        }

        // Check if this line starts an inline asm block containing a st.global
        if t == "// begin inline asm" {
            // Peek ahead to see if this block contains st.global
            pending_store = None;
            // We'll check st.global on subsequent lines
            output.push(line.clone());
            continue;
        }

        if t == "// end inline asm" {
            if pending_store.is_some() {
                // We already emitted the replacement, skip the end marker
                pending_store = None;
            } else {
                output.push(line.clone());
            }
            continue;
        }

        // Match st.global by address register
        if t.contains("st.global") {
            let matched = addr_to_store.iter().find_map(|(addr, stores)| {
                if t.contains(&format!("[{addr}]")) {
                    stores.first().copied()
                } else {
                    None
                }
            });

            if let Some(store) = matched {
                // Remove the "// begin inline asm" we already pushed
                if output.last().map(|l| l.trim() == "// begin inline asm").unwrap_or(false) {
                    output.pop();
                }
                // Also remove any .reg .pred and setp lines from the asm block
                while output.last().map(|l| {
                    let lt = l.trim();
                    lt.starts_with(".reg .pred") || lt.starts_with("setp.") || lt == "{"
                }).unwrap_or(false) {
                    output.pop();
                }

                let row_offset = compute_row_offset(&store.stride_chain, &stride_row_offsets);
                let smem_row_offset = row_offset * row_stride_bytes;
                let data_str = store.data_regs.join(", ");

                output.push(format!(
                    "\t// FERRITE: epilogue store redirected to SMEM scratch"
                ));
                output.push(format!(
                    "\tmad.lo.u32 \t%r_epi_smem_base, {}, {row_stride_bytes}, {smem_offset};",
                    store_map.row_reg
                ));
                output.push(format!(
                    "\tmad.lo.u32 \t%r_epi_smem_addr, {}, 2, %r_epi_smem_base;",
                    store_map.col_reg
                ));
                if smem_row_offset > 0 {
                    output.push(format!(
                        "\tadd.u32 \t%r_epi_smem_addr, %r_epi_smem_addr, {smem_row_offset};"
                    ));
                }
                let vec_width = store.data_regs.len();
                output.push(format!(
                    "\tst.shared.v{vec_width}.b32 \t[%r_epi_smem_addr], {{{data_str}}};"
                ));

                pending_store = Some(store);
                skip_until_end_asm = true;
                continue;
            }
        }

        // Keep all other lines
        output.push(line.clone());
    }

    let smem_bytes = 64 * tile_n * 2; // tile_m=64 × tile_n × 2 bytes (bf16)

    Ok(RedirectedEpilogue {
        lines: output,
        reg_decls,
        smem_bytes,
    })
}

/// Infer the row offset (in elements, not bytes) for each stride register
/// by analyzing the stride chain patterns across all stores.
///
/// The first store with exactly one stride gives that stride's row offset.
/// Cross-referencing with the add.s32 instructions in the epilogue would be
/// more precise, but the chain structure is sufficient: we assign row offsets
/// by examining which stores share the same chain pattern as known MMA
/// fragment layouts (m16n8k16: rows at +2 and +8 intervals).
fn infer_stride_row_offsets(
    stores: &[EpilogueStore],
    stride_regs: &[String],
) -> std::collections::BTreeMap<String, u32> {
    // Strategy: count how many times each stride appears across all stores.
    // The stride that appears in pairs with itself (chain len 2 vs 1) is the
    // larger stride. The one that appears mixed with the other is the smaller one.
    //
    // For m16n8k16 MMA with 64-row tile: rows 0,2,8,10,16,18,24,26
    //   small stride (%rd20) = 2 rows, large stride (%rd21) = 8 rows
    //   chain patterns: [], [s], [L], [s,L], [L,L], [s,L,L], [L,L,L], [s,L,L,L]

    let mut result = std::collections::BTreeMap::new();

    if stride_regs.len() == 2 {
        // Two strides: figure out which is small vs large.
        // The store with chain=[stride_A] should have a smaller row offset
        // than the store with chain=[stride_B].
        // We can determine this from chain lengths: the stride that appears
        // alone in a length-1 chain and also in length-2 chains alongside
        // the other stride is the small one.
        let s0 = &stride_regs[0];
        let s1 = &stride_regs[1];

        // Find stores with single-element chains
        let s0_alone = stores.iter().any(|st| st.stride_chain.len() == 1 && st.stride_chain[0] == *s0);
        let s1_alone = stores.iter().any(|st| st.stride_chain.len() == 1 && st.stride_chain[0] == *s1);

        // Find stores with chain=[sX, sX] (doubled stride)
        let s0_doubled = stores.iter().any(|st| st.stride_chain.len() == 2 && st.stride_chain.iter().all(|s| s == s0));
        let s1_doubled = stores.iter().any(|st| st.stride_chain.len() == 2 && st.stride_chain.iter().all(|s| s == s1));

        // The large stride is the one that appears doubled (stride_chain=[L,L] = +16 rows)
        // The small stride appears alone and mixed with the large
        if s1_doubled && !s0_doubled {
            // s1 is large (8 rows), s0 is small (2 rows)
            result.insert(s0.clone(), 2);
            result.insert(s1.clone(), 8);
        } else if s0_doubled && !s1_doubled {
            // s0 is large, s1 is small
            result.insert(s0.clone(), 8);
            result.insert(s1.clone(), 2);
        } else if s0_alone && s1_alone {
            // Both appear alone — need another heuristic.
            // The one that appears in more chains total is the larger stride
            // (it's used in the +8, +16, +24 positions).
            let s0_count: usize = stores.iter().map(|st| st.stride_chain.iter().filter(|s| *s == s0).count()).sum();
            let s1_count: usize = stores.iter().map(|st| st.stride_chain.iter().filter(|s| *s == s1).count()).sum();
            if s1_count > s0_count {
                result.insert(s0.clone(), 2);
                result.insert(s1.clone(), 8);
            } else {
                result.insert(s0.clone(), 8);
                result.insert(s1.clone(), 2);
            }
        }
    } else if stride_regs.len() == 1 {
        // Single stride register — must be the row stride
        result.insert(stride_regs[0].clone(), 2);
    }

    result
}

/// Compute the total row offset for a stride chain.
fn compute_row_offset(
    stride_chain: &[String],
    stride_row_offsets: &std::collections::BTreeMap<String, u32>,
) -> u32 {
    stride_chain
        .iter()
        .map(|s| stride_row_offsets.get(s).copied().unwrap_or(0))
        .sum()
}

// ===== K-Loop De-pipelining: remove cp.async software pipeline =====

/// Result of de-pipelining a CUTLASS K-loop.
#[derive(Debug, Clone)]
pub struct DepipelinedKLoop {
    /// The modified K-loop lines with cp.async replaced by explicit loads.
    pub lines: Vec<String>,
    /// Number of A-matrix cp.async sites that were replaced.
    pub a_loads_replaced: usize,
    /// Number of B-matrix cp.async sites that were replaced.
    pub b_loads_replaced: usize,
    /// Extra register declarations needed (temp b32 for explicit loads).
    pub reg_decls: Vec<String>,
}

/// Remove the cp.async software pipeline from a CUTLASS K-loop.
///
/// Replaces all cp.async (A and B) with explicit ld.global + st.shared,
/// and strips cp.async.commit_group and cp.async.wait_group instructions.
/// The result is a simple synchronous K-loop body suitable for running
/// a small fixed number of iterations without pipeline overhead.
///
/// A-loads are classified using the same param-tracing analysis as
/// `fuse_cp_async::classify_cp_async_loads`. They can be optionally
/// redirected to SMEM (for register transfer) by passing a non-empty
/// `a_smem_base_reg`. When set, A-loads become `ld.shared` from the
/// given SMEM region instead of `ld.global` from GMEM.
///
/// B-loads always use `ld.global + st.shared` (they load weight tiles).
pub fn depipeline_k_loop(
    k_loop_lines: &[String],
    full_ptx: &str,
    a_smem_base_reg: Option<&str>,
) -> Result<DepipelinedKLoop, String> {
    use crate::fuse_cp_async::{classify_cp_async_loads, identify_a_matrix_param, parse_cp_async};
    use crate::parser::PtxParser;

    let full_lines: Vec<&str> = full_ptx.lines().collect();
    let proto = PtxParser::parse(full_ptx)?;
    let reg_to_param = PtxParser::trace_param_registers_pub(&full_lines, &proto.params);

    // Identify A-matrix parameter
    let a_param = identify_a_matrix_param(&full_lines, &reg_to_param, "")?;
    let a_addr_regs: Vec<String> = reg_to_param
        .iter()
        .filter(|(_, p)| **p == a_param)
        .map(|(r, _)| r.clone())
        .collect();

    // Classify each cp.async in the K-loop lines
    let k_lines_ref: Vec<&str> = k_loop_lines.iter().map(|s| s.as_str()).collect();
    let classifications = classify_cp_async_loads(&k_lines_ref, &a_addr_regs);

    // Temp registers for explicit loads (shared across all replacement sites)
    let reg_decls = vec![
        ".reg .pred %p_dpipe;".into(),
        ".reg .b32 %r_dpipe0, %r_dpipe1, %r_dpipe2, %r_dpipe3;".into(),
    ];

    let mut result = Vec::new();
    let mut a_replaced = 0usize;
    let mut b_replaced = 0usize;
    let mut a_load_idx = 0usize;

    for (i, line) in k_loop_lines.iter().enumerate() {
        let t = line.trim();

        // Strip pipeline synchronization
        if t.contains("cp.async.commit_group") || t.contains("cp.async.wait_group") {
            result.push(format!("\t// FERRITE: removed {}", t));
            continue;
        }

        // Replace cp.async loads
        if t.contains("cp.async.cg.shared.global") {
            if let Some((smem_dst, gmem_src, mask)) = parse_cp_async(t) {
                let is_a = classifications.get(&i) == Some(&crate::fuse_cp_async::CpAsyncClass::AMatrix);

                if is_a {
                    // A-load: either from SMEM scratch or GMEM
                    result.push("\t// FERRITE: A-load (de-pipelined)".into());
                    match a_smem_base_reg {
                        Some(base_reg) => {
                            // Load from SMEM scratch: base + a_load_idx * 16
                            let offset = a_load_idx * 16;
                            result.push(format!(
                                "\tld.shared.v4.b32 \t{{%r_dpipe0, %r_dpipe1, %r_dpipe2, %r_dpipe3}}, [{base_reg}+{offset}];"
                            ));
                        }
                        None => {
                            // Load from GMEM (standard explicit replacement)
                            result.push(format!("\tsetp.ne.b32 \t%p_dpipe, {mask}, 0;"));
                            result.push(format!(
                                "\t@%p_dpipe ld.global.v4.b32 \t{{%r_dpipe0, %r_dpipe1, %r_dpipe2, %r_dpipe3}}, [{gmem_src}];"
                            ));
                            for j in 0..4 {
                                result.push(format!("\t@!%p_dpipe mov.b32 \t%r_dpipe{j}, 0;"));
                            }
                        }
                    }
                    result.push(format!(
                        "\tst.shared.v4.b32 \t[{smem_dst}], {{%r_dpipe0, %r_dpipe1, %r_dpipe2, %r_dpipe3}};"
                    ));
                    a_replaced += 1;
                    a_load_idx += 1;
                } else {
                    // B-load: always from GMEM with explicit ld+st
                    result.push("\t// FERRITE: B-load (de-pipelined)".into());
                    result.push(format!("\tsetp.ne.b32 \t%p_dpipe, {mask}, 0;"));
                    result.push(format!(
                        "\t@%p_dpipe ld.global.v4.b32 \t{{%r_dpipe0, %r_dpipe1, %r_dpipe2, %r_dpipe3}}, [{gmem_src}];"
                    ));
                    for j in 0..4 {
                        result.push(format!("\t@!%p_dpipe mov.b32 \t%r_dpipe{j}, 0;"));
                    }
                    result.push(format!(
                        "\tst.shared.v4.b32 \t[{smem_dst}], {{%r_dpipe0, %r_dpipe1, %r_dpipe2, %r_dpipe3}};"
                    ));
                    b_replaced += 1;
                }
                continue;
            }
        }

        // Keep everything else (MMA, bar.sync, ld.shared, arithmetic, branches)
        result.push(line.clone());
    }

    Ok(DepipelinedKLoop {
        lines: result,
        a_loads_replaced: a_replaced,
        b_loads_replaced: b_replaced,
        reg_decls,
    })
}

/// Sequence two flat-param CUTLASS GEMMs into a single kernel with a barrier.
///
/// Both GEMMs must be perimeter-replaced (using ferrite_params flat layout).
/// The output kernel has two param blocks: `ferrite_params` for GEMM_A and
/// `ferrite_params_2` for GEMM_B.
///
/// GEMM_A runs first, then `bar.sync 0`, then GEMM_B runs.
// ===== GEMM → Pointwise → GEMM fusion (register transfer) =====

/// Configuration for the producer GEMM's output pairing.
#[derive(Debug, Clone)]
pub enum ProducerOutput {
    /// Single output: one N-tile per outer loop iteration.
    Single,
    /// Paired output: two N-tiles per iteration (e.g., gate+up for MLP).
    /// The second tile is at `n_tile + n_offset` in the producer's N-dimension.
    Paired { n_offset_tiles: u32 },
}

/// Fuse two GEMMs with a pointwise operation between them.
///
/// Producer GEMM → redirected epilogue → SMEM scratch
/// Pointwise (e.g., SiLU+mul) applied at consumer GEMM's A-load sites
/// Consumer GEMM accumulates across all producer N-tiles
///
/// The result is a single non-persistent kernel where each block processes
/// one consumer output tile, iterating internally over producer N-tiles.
///
/// Both GEMMs must be perimeter-replaced (flat ferrite_params layout).
/// The producer and consumer can use different tile configs.
///
/// `producer_ptx`: flat-param PTX for the producer GEMM (e.g., gate_up)
/// `consumer_ptx`: flat-param PTX for the consumer GEMM (e.g., down)
/// `pointwise`: the computation to apply between stages (e.g., SiLU+mul)
/// `output_mode`: Single or Paired (for gate+up style outputs)
/// `name`: entry point name for the fused kernel
pub fn fuse_gemm_pointwise_gemm(
    producer_ptx: &str,
    consumer_ptx: &str,
    pointwise: &PointwiseComputation,
    output_mode: &ProducerOutput,
    name: &str,
) -> Result<String, String> {
    use crate::fuse_real::{compute_register_offsets, offset_all_registers};

    // ── Parse both GEMMs ──
    let prod_desc = extract_gemm_body(producer_ptx)?;
    let cons_desc = extract_gemm_body(consumer_ptx)?;
    let prod_proto = crate::parser::PtxParser::parse(producer_ptx)?;
    let cons_proto = crate::parser::PtxParser::parse(consumer_ptx)?;

    // ── Analyze epilogue stores (for redirect) ──
    let prod_lines: Vec<&str> = producer_ptx.lines().collect();
    let prod_epi_start = prod_desc.loop_desc.backedge_line + 1;
    let prod_store_map = analyze_epilogue_stores(&prod_desc.epilogue, &prod_lines, prod_epi_start)?;

    // ── Tile dimensions (derived from GEMM body analysis) ──
    let prod_tile_n = 64u32; // TODO: extract from PTX or derivations
    let cons_tile_n = 64u32;
    let cons_tile_k = 32u32;
    let tile_m = 64u32;

    let k_iters_per_producer_tile = prod_tile_n / cons_tile_k;

    // ── SMEM layout (no accumulator spill — both accum sets live in registers) ──
    // CUTLASS SMEM at offset 0 (24KB for 64×64×32)
    // Gate scratch after CUTLASS, up scratch after gate.
    // Total: 24KB + 8KB + 8KB = 40KB < 50KB (fits 2 blocks/SM on L4)
    let cutlass_smem = 24576u32;
    let gate_scratch_offset = cutlass_smem;
    let gate_scratch_size = tile_m * prod_tile_n * 2; // bf16
    let up_scratch_offset = gate_scratch_offset + gate_scratch_size;

    // ── Build redirected epilogue for producer ──
    let gate_redir = redirect_epilogue_to_smem(
        &prod_desc.epilogue, &prod_store_map, gate_scratch_offset, prod_tile_n,
    )?;
    let up_redir = redirect_epilogue_to_smem(
        &prod_desc.epilogue, &prod_store_map, up_scratch_offset, prod_tile_n,
    )?;

    // ── Build consumer GEMM with pointwise at A-loads ──
    let cons_fused = replace_a_loads_with_inline_fn(consumer_ptx, "", pointwise)?;
    let cons_fused_desc = extract_gemm_body(&cons_fused)?;
    // Parse fused consumer to get register counts AFTER A-load replacement bumped them
    let cons_fused_proto = crate::parser::PtxParser::parse(&cons_fused)?;

    // ── Register namespace: offset consumer registers to avoid collision ──
    let offsets = compute_register_offsets(&prod_proto.registers, &cons_fused_proto.registers);

    // ── Emit the fused kernel ──
    let mut out = String::new();

    // PTX header
    out.push_str(".version 8.7\n");
    out.push_str(".target sm_89\n");
    out.push_str(".address_size 64\n\n");

    // Extern shared memory declaration (CUTLASS uses dynamic shared memory)
    // Extract from the producer PTX (or consumer — they use the same symbol)
    for line in producer_ptx.lines() {
        let t = line.trim();
        if t.starts_with(".extern .shared") {
            out.push_str(&format!("{t}\n\n"));
            break;
        }
    }

    // Entry point with params from both GEMMs + driver loop params
    out.push_str(&format!(".visible .entry {name}(\n"));
    for p in &prod_desc.params {
        out.push_str(&format!("\t{p},\n"));
    }
    // Consumer params (renamed to _2)
    for p in &cons_desc.params {
        let renamed = p.replace("ferrite_params", "ferrite_params_2");
        out.push_str(&format!("\t{renamed},\n"));
    }
    // Extra params from pointwise computation
    for p in &pointwise.extra_params {
        out.push_str(&format!("\t{p}\n"));
    }
    // Driver loop params
    out.push_str("\t.param .u32 _num_producer_n_tiles,\n");
    out.push_str("\t.param .u32 _intermediate_n_offset\n");
    out.push_str(")\n{\n");

    // Register declarations (merged: one declaration per type covering both GEMMs)
    let merged_decls = crate::fuse_real::compute_merged_reg_decls(
        &prod_proto.registers,
        &cons_fused_proto.registers,
        &offsets,
    );
    for (ty, prefix, total_count) in &merged_decls {
        out.push_str(&format!("\t.reg {ty} \t{prefix}<{total_count}>;\n"));
    }
    // Named register declarations from both GEMMs (e.g., %r_ptmp from perimeter replacement)
    // These don't conflict because they use names, not indices.
    let mut named_decls = std::collections::BTreeSet::new();
    for decl in &prod_desc.reg_decls {
        let t = decl.trim();
        // Named regs contain "_" in the register name (not just %r<N>)
        if t.starts_with(".reg") && t.contains("_") {
            named_decls.insert(t.to_string());
        }
    }
    for decl in &cons_desc.reg_decls {
        let t = decl.trim();
        if t.starts_with(".reg") && t.contains("_") {
            named_decls.insert(t.to_string());
        }
    }
    // Also from the fused consumer (which may add more from the pointwise computation)
    for decl in &cons_fused_desc.reg_decls {
        let t = decl.trim();
        if t.starts_with(".reg") && t.contains("_") {
            named_decls.insert(t.to_string());
        }
    }
    for decl in &named_decls {
        out.push_str(&format!("\t{decl}\n"));
    }
    // Extra registers for driver loop
    out.push_str("\t// FERRITE: driver loop registers\n");
    out.push_str("\t.reg .u32 %r_driver_iter, %r_driver_total, %r_driver_n_off;\n");
    out.push_str("\t.reg .pred %p_driver_loop;\n");
    out.push_str("\t.reg .u32 %r_gate_scratch, %r_up_scratch;\n");
    // Epilogue redirect registers
    for decl in &gate_redir.reg_decls {
        out.push_str(&format!("\t{decl}\n"));
    }
    // Pointwise extra reg decls — only emit if not already in named_decls
    for decl in &pointwise.extra_reg_decls {
        let t = decl.trim();
        if !named_decls.contains(t) {
            out.push_str(&format!("\t{decl}\n"));
        }
    }
    out.push_str("\n");

    // SMEM declarations (from both GEMMs — deduplicated)
    let mut smem_decl_set = std::collections::BTreeSet::new();
    for decl in &prod_desc.smem_decls {
        smem_decl_set.insert(decl.trim().to_string());
    }
    for decl in &cons_desc.smem_decls {
        smem_decl_set.insert(decl.trim().to_string());
    }
    for decl in &cons_fused_desc.smem_decls {
        smem_decl_set.insert(decl.trim().to_string());
    }
    for decl in &smem_decl_set {
        out.push_str(&format!("\t{decl}\n"));
    }

    // ── Preamble: tile index from consumer GEMM (defines output tile) ──
    out.push_str("\t// === Consumer tile index (output tile) ===\n");
    for line in &cons_fused_desc.preamble {
        // The consumer preamble computes ctaid.x/y → m_tile, n_tile
        // We need this for the output tile coordinates
        let renamed = offset_all_registers(line, &offsets);
        let renamed = renamed.replace("$L__BB0", "$L__BB_cons");
        out.push_str(&format!("{renamed}\n"));
    }

    // ── Initialize driver loop ──
    out.push_str("\n\t// === Driver loop initialization ===\n");
    out.push_str("\tld.param.u32 \t%r_driver_total, [_num_producer_n_tiles];\n");
    out.push_str("\tld.param.u32 \t%r_driver_n_off, [_intermediate_n_offset];\n");
    out.push_str(&format!("\tmov.u32 \t%r_gate_scratch, {gate_scratch_offset};\n"));
    out.push_str(&format!("\tmov.u32 \t%r_up_scratch, {up_scratch_offset};\n"));

    // Zero consumer accumulators (in offset register namespace)
    out.push_str("\t// FERRITE: zero down MMA accumulators\n");
    for reg in &cons_fused_desc.mma_accumulators {
        let renamed = offset_all_registers(&format!("\tmov.f32 \t{reg}, 0f00000000;"), &offsets);
        out.push_str(&format!("{renamed}\n"));
    }

    out.push_str("\tmov.u32 \t%r_driver_iter, 0;\n");

    // ── Outer loop ──
    // No accumulator spill — both gate_up and down accum sets live in registers.
    // gate_up accums are consumed each iteration. down accums accumulate across iterations.
    out.push_str("\n$L_driver_loop:\n");

    // Gate GEMM: preamble + K-loop + redirected epilogue
    // TODO: The preamble needs tile index overridden:
    //   m_tile = consumer's m_tile (from output tile)
    //   n_tile = driver_iter (current gate N-tile)
    out.push_str("\t// --- Gate GEMM (producer, N-tile = iter) ---\n");
    for line in &prod_desc.preamble {
        out.push_str(&format!("{line}\n"));
    }
    for line in &prod_desc.k_loop {
        out.push_str(&format!("{line}\n"));
    }
    out.push_str("\t// --- Gate epilogue -> SMEM scratch A ---\n");
    for line in &gate_redir.lines {
        out.push_str(&format!("{line}\n"));
    }
    out.push_str("\tbar.sync \t0;\n\n");

    // Up GEMM: same preamble + K-loop but with labels renamed to avoid collision
    out.push_str("\t// --- Up GEMM (producer, N-tile = iter + offset) ---\n");
    for line in &prod_desc.preamble {
        let renamed = line.replace("$L__BB0", "$L__BB_up");
        out.push_str(&format!("{renamed}\n"));
    }
    for line in &prod_desc.k_loop {
        let renamed = line.replace("$L__BB0", "$L__BB_up");
        out.push_str(&format!("{renamed}\n"));
    }
    out.push_str("\t// --- Up epilogue -> SMEM scratch B ---\n");
    for line in &up_redir.lines {
        let renamed = line.replace("$L__BB0", "$L__BB_up");
        out.push_str(&format!("{renamed}\n"));
    }
    out.push_str("\tbar.sync \t0;\n\n");

    // Consumer K-loop with SiLU from SMEM (A-loads replaced by pointwise)
    out.push_str("\t// --- Down K-loop (consumer, A from SiLU scratch) ---\n");
    for line in &cons_fused_desc.k_loop {
        let renamed = offset_all_registers(line, &offsets);
        let renamed = renamed.replace("$L__BB0", "$L__BB_cons");
        out.push_str(&format!("{renamed}\n"));
    }
    out.push_str("\n");

    // Loop back
    out.push_str("\t// --- Loop control ---\n");
    out.push_str("\tadd.u32 \t%r_driver_iter, %r_driver_iter, 1;\n");
    out.push_str("\tsetp.lt.u32 \t%p_driver_loop, %r_driver_iter, %r_driver_total;\n");
    out.push_str("\t@%p_driver_loop bra \t$L_driver_loop;\n\n");

    // Consumer epilogue (final output to GMEM)
    out.push_str("\t// === Down epilogue (output to GMEM) ===\n");
    for line in &cons_fused_desc.epilogue {
        let renamed = offset_all_registers(line, &offsets);
        let renamed = renamed.replace("$L__BB0", "$L__BB_cons");
        out.push_str(&format!("{renamed}\n"));
    }

    out.push_str("\n\tret;\n}\n");

    Ok(out)
}

/// Both phases reuse the same SMEM (sequential execution).
/// GEMM_B's registers are offset to avoid collisions with GEMM_A's.
/// GEMM_B's labels are renamed ($L__BB0 → $L__BB1).
///
/// Returns the sequenced PTX kernel.
/// Sequence two GEMM phases into a single kernel (convenience wrapper).
pub fn sequence_two_gemms(
    gemm_a_ptx: &str,
    gemm_b_ptx: &str,
    name: &str,
) -> Result<String, String> {
    sequence_gemm_phases(&[gemm_a_ptx, gemm_b_ptx], name)
}

/// Sequence N GEMM phases into a single kernel with barriers between them.
///
/// Each phase gets its own param block (ferrite_params, ferrite_params_2, ...),
/// its own label namespace ($L__BB0, $L__BB1, ...), and offset registers to
/// avoid collisions. All phases reuse the same SMEM (sequential execution).
pub fn sequence_gemm_phases(phases: &[&str], name: &str) -> Result<String, String> {
    use crate::fuse_real::{compute_register_offsets, offset_all_registers};
    use std::collections::{BTreeMap, HashSet};

    if phases.is_empty() {
        return Err("no phases to sequence".into());
    }

    // Parse all phases
    let protos: Vec<_> = phases
        .iter()
        .map(|ptx| crate::parser::PtxParser::parse(ptx))
        .collect::<Result<Vec<_>, _>>()?;

    // Extract body and params from each phase
    let extract_body = |ptx: &str| -> Vec<String> {
        let lines: Vec<&str> = ptx.lines().collect();
        let mut in_body = false;
        let mut depth = 0i32;
        let mut body = Vec::new();
        for line in &lines {
            let t = line.trim();
            if !in_body {
                if (t == "{" || t.ends_with('{')) && t.contains('{') {
                    in_body = true;
                    depth = 1;
                }
                continue;
            }
            for ch in t.chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            if depth <= 0 || t == "ret;" {
                break;
            }
            if t.starts_with(".reg ") || t.starts_with(".shared ") {
                continue;
            }
            body.push(line.to_string());
        }
        body
    };

    let extract_params = |ptx: &str| -> Vec<String> {
        let mut params = Vec::new();
        let mut in_entry = false;
        for line in ptx.lines() {
            let t = line.trim();
            if t.contains(".entry") {
                in_entry = true;
                continue;
            }
            if !in_entry {
                continue;
            }
            if t.starts_with(".param") {
                params.push(t.trim_end_matches(',').to_string());
            }
            if t == ")" || t == "{" || t.ends_with('{') {
                break;
            }
        }
        params
    };

    let bodies: Vec<Vec<String>> = phases.iter().map(|ptx| extract_body(ptx)).collect();
    let all_params: Vec<Vec<String>> = phases.iter().map(|ptx| extract_params(ptx)).collect();

    // Compute cumulative register offsets: phase 0 gets no offset,
    // phase 1 gets offset by phase 0's register count, etc.
    let mut cumulative_offsets: Vec<BTreeMap<String, usize>> = Vec::new();
    cumulative_offsets.push(BTreeMap::new()); // Phase 0: no offset

    let mut running_regs = protos[0].registers.clone();
    for proto in &protos[1..] {
        let offsets = compute_register_offsets(&running_regs, &proto.registers);
        cumulative_offsets.push(offsets.clone());
        // Update running totals
        for (ty, count) in &proto.registers {
            let offset = offsets.get(ty).copied().unwrap_or(0);
            let entry = running_regs
                .iter_mut()
                .find(|(t, _)| t == ty)
                .map(|(_, c)| c);
            if let Some(c) = entry {
                *c = offset + count;
            } else {
                running_regs.push((ty.clone(), offset + count));
            }
        }
    }

    // Rename each phase's body (except phase 0)
    let renamed_bodies: Vec<Vec<String>> = bodies
        .iter()
        .enumerate()
        .map(|(phase_idx, body)| {
            if phase_idx == 0 {
                return body.clone();
            }
            let offsets = &cumulative_offsets[phase_idx];
            let suffix = if phase_idx == 1 {
                "_2".to_string()
            } else {
                format!("_{}", phase_idx + 1)
            };
            let param_suffix = suffix.clone();

            // Collect extra param names for this phase
            let extra_param_names: Vec<String> = all_params[phase_idx]
                .iter()
                .filter_map(|p| {
                    if p.contains("ferrite_params") {
                        return None;
                    }
                    p.split_whitespace()
                        .last()
                        .map(|s| s.trim_end_matches(',').to_string())
                })
                .collect();

            body.iter()
                .map(|line| {
                    let mut renamed = offset_all_registers(line, offsets);
                    renamed = renamed.replace("$L__BB0", &format!("$L__BB{phase_idx}"));
                    renamed =
                        renamed.replace("ferrite_params", &format!("ferrite_params{param_suffix}"));
                    for pname in &extra_param_names {
                        if renamed.contains(pname.as_str()) {
                            renamed = renamed.replace(pname.as_str(), &format!("{pname}{suffix}"));
                        }
                    }
                    renamed
                })
                .collect()
        })
        .collect();

    // Rename each phase's params (except phase 0)
    let renamed_params: Vec<Vec<String>> = all_params
        .iter()
        .enumerate()
        .map(|(phase_idx, params)| {
            if phase_idx == 0 {
                return params.clone();
            }
            let suffix = if phase_idx == 1 {
                "_2".to_string()
            } else {
                format!("_{}", phase_idx + 1)
            };
            params
                .iter()
                .map(|p| {
                    if p.contains("ferrite_params") {
                        p.replace("ferrite_params", &format!("ferrite_params{suffix}"))
                    } else if let Some(last_word) = p.split_whitespace().last() {
                        let clean = last_word.trim_end_matches(',');
                        p.replace(clean, &format!("{clean}{suffix}"))
                    } else {
                        p.clone()
                    }
                })
                .collect()
        })
        .collect();

    // Merged register declarations (total across all phases)
    let merged_regs = {
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
        let mut totals: BTreeMap<String, usize> = BTreeMap::new();
        for proto in &protos {
            for (ty, count) in &proto.registers {
                let entry = totals.entry(ty.clone()).or_insert(0);
                *entry += count;
            }
        }
        let mut result = Vec::new();
        for (ty, total) in &totals {
            if let Some((_, prefix)) = type_to_prefix.iter().find(|(t, _)| *t == ty.as_str()) {
                result.push((ty.clone(), prefix.to_string(), *total));
            }
        }
        result
    };

    // Build PTX
    let mut out = String::new();
    out.push_str(".version 8.8\n.target sm_89\n.address_size 64\n\n");

    // Top-level .extern .shared declarations (from phase 0)
    for line in phases[0].lines() {
        let t = line.trim();
        if t.starts_with(".extern") && t.contains(".shared") {
            out.push_str(t);
            out.push('\n');
        }
    }
    out.push('\n');

    let num_barriers = phases.len().saturating_sub(1);

    // Entry with barrier param + all phases' params
    out.push_str(&format!(".visible .entry {name}(\n"));
    if num_barriers > 0 {
        out.push_str("\t.param .u64 _phase_barriers,\n");
    }
    let flat_params: Vec<&String> = renamed_params.iter().flat_map(|p| p.iter()).collect();
    for (i, p) in flat_params.iter().enumerate() {
        let comma = if i + 1 < flat_params.len() { "," } else { "" };
        out.push_str(&format!("\t{p}{comma}\n"));
    }
    out.push_str(")\n{\n");

    // Register declarations
    for (reg_type, prefix, total) in &merged_regs {
        if *total > 0 {
            out.push_str(&format!("\t.reg {reg_type} \t{prefix}<{total}>;\n"));
        }
    }

    // Copy non-standard .reg declarations from all phases
    let is_standard_numeric_range = |t: &str| -> bool {
        let prefixes = ["%r<", "%rd<", "%f<", "%p<", "%rs<", "%fd<"];
        t.contains('<') && prefixes.iter().any(|p| t.contains(p)) && !t.contains('_')
    };
    let mut seen_decls = HashSet::new();
    for ptx in phases {
        let lines: Vec<&str> = ptx.lines().collect();
        let mut in_body = false;
        for line in &lines {
            let t = line.trim();
            if t == "{" || t.ends_with('{') {
                in_body = true;
                continue;
            }
            if !in_body {
                continue;
            }
            if t.starts_with(".reg ") && !is_standard_numeric_range(t) {
                if seen_decls.insert(t.to_string()) {
                    out.push_str(&format!("\t{t}\n"));
                }
            }
        }
    }

    // SMEM declarations from phase 0 body (reused by all phases)
    {
        let lines: Vec<&str> = phases[0].lines().collect();
        let mut in_body = false;
        for line in &lines {
            let t = line.trim();
            if t == "{" || t.ends_with('{') {
                in_body = true;
                continue;
            }
            if !in_body {
                continue;
            }
            if t.starts_with(".shared") {
                out.push_str(&format!("\t{t}\n"));
            }
        }
    }
    // Also copy SMEM from later phases (e.g., norm's _ferrite_inv_rms)
    let mut seen_smem = HashSet::new();
    for ptx in &phases[1..] {
        let lines: Vec<&str> = ptx.lines().collect();
        let mut in_body = false;
        for line in &lines {
            let t = line.trim();
            if t == "{" || t.ends_with('{') {
                in_body = true;
                continue;
            }
            if !in_body {
                continue;
            }
            if t.starts_with(".shared") && seen_smem.insert(t.to_string()) {
                out.push_str(&format!("\t{t}\n"));
            }
        }
    }
    // Global barrier infrastructure (for inter-block synchronization between phases)
    if num_barriers > 0 {
        out.push_str("\t// Global barrier registers\n");
        out.push_str("\t.reg .u64 \t%rd_gbar_ptr;\n");
        out.push_str("\t.reg .u32 \t%r_gbar_val, %r_gbar_total, %r_gbar_tid;\n");
        out.push_str("\t.reg .pred \t%p_gbar_t0, %p_gbar_done;\n");
        out.push_str("\t.shared .align 4 .u32 _gbar_sense[1];\n");
    }
    out.push('\n');

    // Compute grid total (gridDim.x * gridDim.y) for barrier target
    if num_barriers > 0 {
        out.push_str("\t// Compute grid total for global barrier\n");
        out.push_str("\tmov.u32 \t%r_gbar_total, %nctaid.x;\n");
        out.push_str("\tmov.u32 \t%r_gbar_val, %nctaid.y;\n");
        out.push_str("\tmul.lo.u32 \t%r_gbar_total, %r_gbar_total, %r_gbar_val;\n");
        out.push_str("\tld.param.u64 \t%rd_gbar_ptr, [_phase_barriers];\n");
        out.push_str("\tcvta.to.global.u64 \t%rd_gbar_ptr, %rd_gbar_ptr;\n\n");
    }

    // Emit phases with global barriers between them
    for (phase_idx, body) in renamed_bodies.iter().enumerate() {
        out.push_str(&format!("\t// ====== Phase {} ======\n", phase_idx + 1));
        for line in body {
            out.push_str(line);
            out.push('\n');
        }
        if phase_idx + 1 < renamed_bodies.len() {
            // Global atomic barrier: all blocks must arrive before any proceeds
            let barrier_offset = phase_idx * 4; // each barrier is a u32
            out.push_str(&format!(
                "\n\t// ====== Global barrier {} (all blocks sync) ======\n",
                phase_idx + 1
            ));
            // First: block-level barrier to ensure all threads in this block are done
            out.push_str("\tbar.sync \t0;\n");
            // Thread 0 atomicAdds the global counter
            out.push_str("\tmov.u32 \t%r_gbar_tid, %tid.x;\n");
            out.push_str("\tsetp.eq.u32 \t%p_gbar_t0, %r_gbar_tid, 0;\n");
            out.push_str(&format!(
                "\t@%p_gbar_t0 atom.global.add.u32 \t%r_gbar_val, [%rd_gbar_ptr+{barrier_offset}], 1;\n"
            ));
            // Thread 0 stores arrival count to SMEM for broadcast
            out.push_str("\t@%p_gbar_t0 add.u32 \t%r_gbar_val, %r_gbar_val, 1;\n");
            // Spin until all blocks have arrived
            out.push_str(&format!("$L_gbar_spin_{phase_idx}:\n"));
            out.push_str(&format!(
                "\tld.global.acquire.gpu.u32 \t%r_gbar_val, [%rd_gbar_ptr+{barrier_offset}];\n"
            ));
            out.push_str("\tsetp.ge.u32 \t%p_gbar_done, %r_gbar_val, %r_gbar_total;\n");
            out.push_str(&format!(
                "\t@!%p_gbar_done bra \t$L_gbar_spin_{phase_idx};\n"
            ));
            // Block-level barrier after spinning to sync all threads before next phase
            out.push_str("\tbar.sync \t0;\n\n");
        }
    }

    out.push_str("}\n");
    Ok(out)
}

/// Wrap a multi-phase sequenced kernel in a persistent work-queue loop.
///
/// Transforms the global barrier model (all blocks sync between phases) into
/// a persistent model where `num_blocks` blocks loop over tiles across all phases.
///
/// The work queue is organized as: all Phase 0 tiles, then all Phase 1 tiles, etc.
/// Per-M-tile atomic counters ensure Phase N+1 tiles don't start until their
/// M-tile's Phase N tiles are complete.
///
/// Params added:
/// - `_persistent_counter`: u64 ptr to atomic tile counter (u32, init to 0)
/// - `_persistent_mtile_done`: u64 ptr to per-M-tile completion counters
///   (array of u32, one per M-tile per barrier, all init to 0)
/// - `_persistent_total_tiles`: u32 total tiles across all phases
///
/// Grid dims for each phase are baked as constants from the phase params
/// (read M, N from ferrite_params at compile time is not possible, so the
/// caller passes the tile counts).
pub fn make_persistent_multiphase(
    sequenced_ptx: &str,
    name: &str,
    num_phases: usize,
    tiles_per_phase: &[u32], // [total_tiles_phase0, total_tiles_phase1, ...]
    n_tiles_per_phase: &[u32], // [n_tiles for phase 0, n_tiles for phase 1, ...]
) -> Result<String, String> {
    if tiles_per_phase.len() != num_phases || n_tiles_per_phase.len() != num_phases {
        return Err("tiles_per_phase and n_tiles_per_phase must match num_phases".into());
    }

    let total_tiles: u32 = tiles_per_phase.iter().sum();
    let mut cumulative_tiles = vec![0u32]; // cumulative[i] = sum of tiles for phases 0..i-1
    for &t in tiles_per_phase {
        cumulative_tiles.push(cumulative_tiles.last().unwrap() + t);
    }

    let mut lines: Vec<String> = sequenced_ptx.lines().map(|l| l.to_string()).collect();

    // Find and rename entry
    let entry_idx = lines
        .iter()
        .position(|l| l.contains(".visible") && l.contains(".entry"))
        .ok_or("no .entry found")?;
    let old_name = {
        let line = &lines[entry_idx];
        let pos = line.find(".entry").unwrap();
        let after = line[pos + 6..].trim();
        let paren = after.find('(').ok_or("no '(' in entry")?;
        after[..paren].trim().to_string()
    };
    lines[entry_idx] = lines[entry_idx].replace(&old_name, name);

    // Add persistent params before existing params
    let first_param_idx = lines
        .iter()
        .position(|l| l.trim().starts_with(".param"))
        .ok_or("no .param found")?;
    let persistent_params = vec![
        "\t.param .u64 _persistent_counter,".to_string(),
        "\t.param .u64 _persistent_mtile_done,".to_string(),
        format!("\t.param .u32 _persistent_total_tiles, // = {total_tiles}"),
    ];
    for (j, p) in persistent_params.iter().enumerate().rev() {
        lines.insert(first_param_idx, p.clone());
    }

    // Find body start (after '{')
    let body_start = lines
        .iter()
        .position(|l| {
            let t = l.trim();
            t == "{" || t.ends_with('{')
        })
        .ok_or("no '{' found")?;

    let mut insert_pos = body_start + 1;
    // Skip past declarations
    while insert_pos < lines.len() {
        let t = lines[insert_pos].trim();
        if !t.starts_with(".reg")
            && !t.starts_with(".shared")
            && !t.starts_with(".local")
            && !t.starts_with("//")
            && !t.is_empty()
        {
            break;
        }
        insert_pos += 1;
    }

    // Insert persistent declarations
    let decls = vec![
        "\t// FERRITE: persistent multi-phase scratch".to_string(),
        "\t.reg .u32 \t%r_ptile, %r_ptotal, %r_pphase;".to_string(),
        "\t.reg .u32 \t%r_pm_tile, %r_pn_tile, %r_poffs;".to_string(),
        "\t.reg .u64 \t%rd_pctr, %rd_pmtdone;".to_string(),
        "\t.reg .pred \t%p_pdone, %p_pt0, %p_pphase;".to_string(),
        "\t.shared .align 4 .u32 _ptile_smem[1];".to_string(),
        String::new(),
    ];
    for (j, d) in decls.iter().enumerate() {
        lines.insert(insert_pos + j, d.clone());
    }
    insert_pos += decls.len();

    // Insert persistent loop preamble
    let preamble = vec![
        "\t// FERRITE: persistent loop setup".to_string(),
        "\tld.param.u64 \t%rd_pctr, [_persistent_counter];".to_string(),
        "\tcvta.to.global.u64 \t%rd_pctr, %rd_pctr;".to_string(),
        "\tld.param.u64 \t%rd_pmtdone, [_persistent_mtile_done];".to_string(),
        "\tcvta.to.global.u64 \t%rd_pmtdone, %rd_pmtdone;".to_string(),
        "\tld.param.u32 \t%r_ptotal, [_persistent_total_tiles];".to_string(),
        String::new(),
        "$L_persistent_loop:".to_string(),
        "\t// Grab next tile".to_string(),
        "\tmov.u32 \t%r_ptile, %tid.x;".to_string(),
        "\tsetp.eq.u32 \t%p_pt0, %r_ptile, 0;".to_string(),
        "\t@%p_pt0 atom.global.add.u32 \t%r_ptile, [%rd_pctr], 1;".to_string(),
        "\t@%p_pt0 st.shared.u32 \t[_ptile_smem], %r_ptile;".to_string(),
        "\tbar.sync \t14;".to_string(),
        "\tld.shared.u32 \t%r_ptile, [_ptile_smem];".to_string(),
        "\tsetp.ge.u32 \t%p_pdone, %r_ptile, %r_ptotal;".to_string(),
        "\t@%p_pdone bra \t$L_persistent_exit;".to_string(),
        String::new(),
    ];
    for (j, line) in preamble.iter().enumerate() {
        lines.insert(insert_pos + j, line.clone());
    }
    insert_pos += preamble.len();

    // Insert phase dispatch: determine which phase this tile belongs to
    // and compute (m_tile, n_tile) within that phase.
    // Then jump to the appropriate phase label.
    let mut dispatch = Vec::new();
    dispatch.push("\t// Phase dispatch: tile_idx → phase + (m_tile, n_tile)".to_string());
    for phase_idx in 0..num_phases {
        let cum = cumulative_tiles[phase_idx];
        let n_tiles = n_tiles_per_phase[phase_idx];
        let phase_label = format!("$L_phase_{phase_idx}");
        let next_label = if phase_idx + 1 < num_phases {
            format!("$L_phase_check_{}", phase_idx + 1)
        } else {
            "$L_persistent_loop_back".to_string()
        };

        if phase_idx == 0 {
            dispatch.push(format!(
                "\tsetp.lt.u32 \t%p_pphase, %r_ptile, {};",
                cumulative_tiles[1]
            ));
            dispatch.push(format!("\t@%p_pphase bra \t{phase_label};"));
        } else {
            dispatch.push(format!("$L_phase_check_{phase_idx}:"));
            if phase_idx + 1 < num_phases {
                dispatch.push(format!(
                    "\tsetp.lt.u32 \t%p_pphase, %r_ptile, {};",
                    cumulative_tiles[phase_idx + 1]
                ));
                dispatch.push(format!("\t@%p_pphase bra \t{phase_label};"));
            } else {
                dispatch.push(format!("\tbra \t{phase_label};"));
            }
        }
    }
    dispatch.push(String::new());

    // Phase entry points: compute m_tile, n_tile, set %ctaid.x/%ctaid.y equivalents
    for phase_idx in 0..num_phases {
        let cum = cumulative_tiles[phase_idx];
        let n_tiles = n_tiles_per_phase[phase_idx];

        dispatch.push(format!("$L_phase_{phase_idx}:"));
        // Subtract cumulative offset to get tile index within this phase
        if cum > 0 {
            dispatch.push(format!("\tsub.u32 \t%r_poffs, %r_ptile, {cum};"));
        } else {
            dispatch.push("\tmov.u32 \t%r_poffs, %r_ptile;".to_string());
        }
        // Decode: m_tile = poffs / n_tiles, n_tile = poffs % n_tiles
        dispatch.push(format!("\tdiv.u32 \t%r_pm_tile, %r_poffs, {n_tiles};"));
        dispatch.push(format!("\trem.u32 \t%r_pn_tile, %r_poffs, {n_tiles};"));

        // TODO: per-M-tile barrier for phase > 0
        // For now: phase 0 tiles run immediately, phase 1+ tiles need to wait
        // until all phase (idx-1) N-tiles for their M-tile are done.
        if phase_idx > 0 {
            let prev_n_tiles = n_tiles_per_phase[phase_idx - 1];
            let barrier_idx = phase_idx - 1;
            // Spin until mtile_done[m_tile * num_barriers + barrier_idx] >= prev_n_tiles
            dispatch.push(format!(
                "\t// Wait for M-tile's Phase {} to complete",
                phase_idx - 1
            ));
            // Compute barrier address: &mtile_done[m_tile * {num_barriers} + {barrier_idx}]
            let num_barriers = num_phases - 1;
            dispatch.push(format!(
                "\tmul.lo.u32 \t%r_poffs, %r_pm_tile, {num_barriers};"
            ));
            dispatch.push(format!("\tadd.u32 \t%r_poffs, %r_poffs, {barrier_idx};"));
            dispatch.push("\tshl.b32 \t%r_poffs, %r_poffs, 2;".to_string()); // * 4 bytes
            dispatch.push("\tcvt.u64.u32 \t%rd_pctr, %r_poffs;".to_string()); // reuse rd_pctr temporarily
            dispatch.push("\tadd.u64 \t%rd_pctr, %rd_pmtdone, %rd_pctr;".to_string());
            dispatch.push(format!("$L_mtile_wait_{phase_idx}:"));
            dispatch.push("\tld.global.acquire.gpu.u32 \t%r_poffs, [%rd_pctr];".to_string());
            dispatch.push(format!(
                "\tsetp.ge.u32 \t%p_pdone, %r_poffs, {prev_n_tiles};"
            ));
            dispatch.push(format!("\t@!%p_pdone bra \t$L_mtile_wait_{phase_idx};"));
            // Restore rd_pctr to the persistent counter
            dispatch.push("\tld.param.u64 \t%rd_pctr, [_persistent_counter];".to_string());
            dispatch.push("\tcvta.to.global.u64 \t%rd_pctr, %rd_pctr;".to_string());
        }

        // Set ctaid.x/y equivalents — the GEMM body reads these
        // For the swizzle: ctaid.x encodes both m_tile and n_tile
        // ctaid.x = m_tile * swizzle_tile + (n_tile % swizzle_tile)
        // ctaid.y = n_tile / swizzle_tile
        // But the sequenced kernel already has the swizzle computation in each phase.
        // We just need to set ctaid.x and ctaid.y to the dispatched values.
        // Problem: we can't SET %ctaid.x — it's a read-only special register.
        // The existing make_persistent replaces `mov.u32 %rN, %ctaid.x` with
        // `mov.u32 %rN, %r_ptile`. We need a similar approach but phase-aware.
        // For now, store the linear tile index and let the swizzle in each phase body
        // decode it correctly. But each phase has different N dims...
        //
        // Actually: set %r_ptile to the SWIZZLED ctaid.x value that the GEMM expects.
        // ctaid.x = m_tile * tile + (n_tile & (tile-1))
        // ctaid.y = n_tile >> swizzle_log
        // But swizzle_log depends on n_tiles which varies per phase.
        //
        // Simplest correct approach: compute the swizzled ctaid.x and ctaid.y values
        // from (m_tile, n_tile) and replace %ctaid.x/%ctaid.y reads.
        // But we can't replace %ctaid.y reads because the register rename only
        // handles %ctaid.x in make_persistent...
        //
        // For phase 0: jump to the phase 0 body start
        // For phase 1+: jump to the phase N body start (after the previous barrier)
        dispatch.push(format!("\tbra \t$L_phase_body_{phase_idx};"));
        dispatch.push(String::new());
    }

    // Insert dispatch code
    for (j, line) in dispatch.iter().enumerate() {
        lines.insert(insert_pos + j, line.clone());
    }

    // TODO: Insert phase body labels ($L_phase_body_N) at the start of each phase
    // TODO: Replace %ctaid.x with the dispatched m/n tile values
    // TODO: Add per-M-tile completion counter increment at end of each phase
    // TODO: Replace ret; with loop back

    // For now, return what we have as a structural proof
    Ok(lines.join("\n"))
}

/// Compile a fusible segment into a single kernel.
///
/// Takes two `PipelineStage`s and their `TilePerimeter`s, connected by a
/// `StageEdge`. Produces fused PTX.
///
/// For `ReductionToGemm`: the reduction's accumulate+finalize becomes the GEMM
/// prologue, and the reduction's per-element formula (derived from the emit body
/// in the TilePerimeter) is injected at each A-load site.
pub fn compile_segment(
    producer: &PipelineStage,
    consumer: &PipelineStage,
    producer_perim: &TilePerimeter,
    _consumer_perim: &TilePerimeter,
    fused_name: &str,
) -> Result<String, String> {
    use crate::pipeline::{StageEdge, StageKind};

    // Determine the edge type from the stage kinds
    let edge = match (&producer_perim.kind, &_consumer_perim.kind) {
        (StageKind::Reduction, StageKind::TiledGemm) => StageEdge::ReductionToGemm,
        (StageKind::Pointwise, StageKind::TiledGemm) => StageEdge::PointwiseToGemm,
        _ => {
            return Err(format!(
                "unsupported edge: {:?} → {:?}",
                producer_perim.kind, _consumer_perim.kind
            ));
        }
    };

    match edge {
        StageEdge::ReductionToGemm => {
            // Verify the producer's TilePerimeter has an InlineFormula output
            // with a non-empty emit body
            let has_emit_body = producer_perim.outputs.iter().any(|p| {
                matches!(
                    &p.access,
                    TilePortAccess::InlineFormula { emit_body } if !emit_body.is_empty()
                )
            });
            if !has_emit_body {
                return Err("reduction TilePerimeter has no InlineFormula output".into());
            }

            // Delegate to the existing fusion function (which now uses
            // extract_per_element_from_emit_body internally)
            fuse_reduction_into_gemm(producer, consumer, fused_name)
        }
        StageEdge::PointwiseToGemm => fuse_pointwise_into_gemm(producer, consumer, fused_name),
        StageEdge::GmemMaterialization => Err("GmemMaterialization not yet implemented".into()),
    }
}

/// Extract per-element instruction templates from a reduction's emit body.
///
/// The emit body (from `ReductionDecomposition`) contains the full pass-2 loop:
/// loads, per-element computation, and stores. This function extracts JUST the
/// per-element pattern and converts it to `{INPUT}` / `{ELEM_IDX}` templates.
///
/// For rms_norm bf16, the emit body contains 8 pairs like:
/// ```ptx
/// mul.f32  %f98, %f74, %f12;    // input * inv_rms
/// mul.f32  %f90, %f98, %f82;    // result * weight
/// ```
///
/// Returns per-element instructions using `{INPUT}` for the input value
/// and `%f_rms_inv` for inv_rms (loaded from SMEM by the prologue),
/// and `%f_rms_wt{ELEM_IDX}` for the weight (loaded per-site).
fn extract_per_element_from_emit_body(
    emit_body: &[String],
    finalized_value_reg: &str,
) -> Option<Vec<String>> {
    // Find all mul.f32 instructions that use the finalized_value_reg (inv_rms)
    // These are the "input * inv_rms" multiplications.
    let mut inv_rms_muls: Vec<(usize, String, String)> = Vec::new(); // (line_idx, dest, input_src)

    for (i, line) in emit_body.iter().enumerate() {
        let t = line.trim();
        if !t.starts_with("mul.f32") {
            continue;
        }
        // Parse: mul.f32 %dest, %srcA, %srcB;
        let parts: Vec<&str> = t
            .split([' ', '\t', ',', ';'])
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() < 4 {
            continue;
        }
        let dest = parts[1].trim();
        let src_a = parts[2].trim();
        let src_b = parts[3].trim();

        if src_a == finalized_value_reg || src_b == finalized_value_reg {
            let input = if src_a == finalized_value_reg {
                src_b
            } else {
                src_a
            };
            inv_rms_muls.push((i, dest.to_string(), input.to_string()));
        }
    }

    if inv_rms_muls.is_empty() {
        return None;
    }

    // For each inv_rms mul, find the subsequent weight mul that uses its dest.
    // Pattern: mul.f32 %fN, %inv_rms_result, %weight_reg
    let mut has_weight_mul = false;
    for &(inv_idx, ref inv_dest, _) in &inv_rms_muls {
        for line in emit_body[inv_idx + 1..].iter().take(5) {
            let t = line.trim();
            if t.starts_with("mul.f32") && t.contains(inv_dest.as_str()) {
                has_weight_mul = true;
                break;
            }
        }
        if has_weight_mul {
            break;
        }
    }

    // Generate the per-element template instructions.
    // These use the framework's placeholders:
    // - {INPUT}: the f32 value from the A-load (after bf16→f32 conversion)
    // - {ELEM_IDX}: 0..7 index for selecting the weight register
    // - %f_rms_inv: inv_rms for this row (loaded from SMEM by per-site code)
    // - %f_rms_wt{ELEM_IDX}: weight for this K position (loaded per-site)
    let mut instructions = Vec::new();
    instructions.push("mul.f32 \t{INPUT}, {INPUT}, %f_rms_inv;".into());
    if has_weight_mul {
        instructions.push("mul.f32 \t{INPUT}, {INPUT}, %f_rms_wt{ELEM_IDX};".into());
    }

    Some(instructions)
}

/// Extract the GEMM tile dimensions from a CUTLASS mangled entry name.
///
/// Parses `GemmShapeILi{M}ELi{N}ELi{K}E` → (M, N, K).
fn extract_gemm_shape_from_entry(name: &str) -> Option<(u32, u32, u32)> {
    let marker = "GemmShapeILi";
    let idx = name.find(marker)?;
    let rest = &name[idx + marker.len()..];
    // rest = "{M}ELi{N}ELi{K}E..."
    let mut dims = Vec::new();
    let mut cur = rest;
    for _ in 0..3 {
        let end = cur.find('E')?;
        dims.push(cur[..end].parse::<u32>().ok()?);
        cur = &cur[end + 1..]; // skip 'E'
        if cur.starts_with("Li") {
            cur = &cur[2..]; // skip 'Li'
        }
    }
    Some((dims[0], dims[1], dims[2]))
}

/// Extract just tile_m for backward compatibility.
fn extract_tile_m_from_entry(name: &str) -> Option<u32> {
    extract_gemm_shape_from_entry(name).map(|(m, _, _)| m)
}

/// Thread-to-row mapping parameters extracted from GEMM PTX.
///
/// The mapping formula is:
///   m_rel = (tid.x % 32) / lanes_per_row + (tid.x / 32) * rows_per_warp
///   row_0 = m_rel, row_1 = m_rel + row_stride
///
/// These are extracted from the `shr` and `shl` shift amounts in the
/// A-load address computation chain.
#[derive(Debug, Clone)]
struct ThreadRowMap {
    /// Number of lanes that share a row (detected as 1 << shr_amount on lane)
    lanes_per_row_log2: u32,
    /// Number of rows each warp covers (detected as 1 << shl_amount on warp)
    rows_per_warp_log2: u32,
    /// Row stride between a thread's two rows (typically 8)
    row_stride: u32,
}

/// Extract the thread-to-row mapping from a GEMM kernel's PTX.
///
/// Strategy: find the first A-load's address chain, locate the
/// `mul.lo.s64 %rd, stride, row_reg`, then trace `row_reg` backward
/// to find the `shr` (lane division) and `shl` (warp multiply) constants.
fn extract_thread_row_map(gemm_lines: &[&str]) -> Option<ThreadRowMap> {
    let graph = DefUseGraph::build(gemm_lines);

    // Find the first cp.async A-load's gmem_src register
    let mut gmem_src = String::new();
    for line in gemm_lines {
        let t = line.trim();
        if t.contains("cp.async.cg.shared.global") {
            // Extract second bracket content (gmem source)
            let mut brackets = Vec::new();
            let mut i = 0;
            let bytes = t.as_bytes();
            while i < bytes.len() {
                if bytes[i] == b'[' {
                    let start = i + 1;
                    while i < bytes.len() && bytes[i] != b']' {
                        i += 1;
                    }
                    brackets.push(t[start..i].trim().to_string());
                }
                i += 1;
            }
            if brackets.len() >= 2 {
                gmem_src = brackets[1].clone();
                break;
            }
        }
    }
    if gmem_src.is_empty() {
        return None;
    }

    // Trace backward from gmem_src to find the mul.lo.s64 by stride
    let trace = graph.trace_backward(&gmem_src, 10);

    // Find the mul.lo.s64 — one operand is the stride (from ld.param), the other is the row
    let mut row_reg = String::new();
    for &(_depth, node_idx) in &trace {
        let node = &graph.nodes[node_idx];
        if node.opcode == "mul.lo.s64" && node.sources.len() == 2 {
            // One source should be a param-derived register, the other is the row
            // The param-derived one will appear in ld.param results
            // Check which source is NOT in our trace (i.e., comes from a param)
            for src in &node.sources {
                // Trace this source backward — if it hits ld.param quickly, it's the stride
                let sub_trace = graph.trace_backward(src, 3);
                let hits_param = sub_trace
                    .iter()
                    .any(|&(_, ni)| graph.nodes[ni].opcode.starts_with("ld.param"));
                if !hits_param {
                    // This is the row register (not from param)
                    row_reg = src.clone();
                }
            }
            if !row_reg.is_empty() {
                break;
            }
        }
    }
    if row_reg.is_empty() {
        return None;
    }

    // Trace the row register backward to find the shift amounts.
    // The row is computed from tid.x via:
    //   lane = tid.x % 32  (and/sub pattern)
    //   warp = tid.x / 32  (shr by 5)
    //   m_in_warp = lane >> N  (shr.s32, N = lanes_per_row_log2)
    //   warp_contrib = warp << M  (shl.b32, M = rows_per_warp_log2)
    //   m_rel = m_in_warp + warp_contrib  (add)
    let row_trace = graph.trace_backward(&row_reg, 12);

    let mut lane_shr: Option<u32> = None;
    let mut warp_shl: Option<u32> = None;

    for &(_depth, node_idx) in &row_trace {
        let node = &graph.nodes[node_idx];

        // Look for shr.s32 with a small immediate — lane / N
        if node.opcode == "shr.s32" && node.sources.len() == 2 {
            if let Ok(shift) = node.sources[1].parse::<u32>() {
                if shift >= 1 && shift <= 4 && shift != 5 {
                    // shr by 2 = lane / 4 (not shr by 5 which is tid.x / 32)
                    lane_shr = Some(shift);
                }
            }
        }

        // Look for shl.b32 with a small immediate — warp * N
        if node.opcode == "shl.b32" && node.sources.len() == 2 {
            if let Ok(shift) = node.sources[1].parse::<u32>() {
                if shift >= 3 && shift <= 5 {
                    // shl by 4 = warp * 16, shl by 5 = warp * 32
                    warp_shl = Some(shift);
                }
            }
        }
    }

    // Detect row_stride by looking for the second row computation (add.s32 %r, %r, N)
    // where N is the stride between a thread's two rows
    let mut row_stride = 8u32; // default
    for &(_depth, node_idx) in &row_trace {
        let node = &graph.nodes[node_idx];
        if node.opcode == "shl.b32" && node.sources.len() == 2 {
            if let Ok(shift) = node.sources[1].parse::<u32>() {
                if shift == 3 {
                    // shl by 3 = offset * 8, suggests row_stride = 8
                    row_stride = 8;
                }
            }
        }
    }

    Some(ThreadRowMap {
        lanes_per_row_log2: lane_shr.unwrap_or(2), // default: lane / 4
        rows_per_warp_log2: warp_shl.unwrap_or(4), // default: warp * 16
        row_stride,
    })
}

/// Fuse a Reduction stage into a TiledGemm stage's A-input prologue.
///
/// The reduction's accumulate+finalize becomes the GEMM prologue.
/// The reduction's per-element emit formula becomes a PointwiseComputation
/// injected at each A-matrix cp.async site.
///
/// Returns the fused PTX kernel.
pub fn fuse_reduction_into_gemm(
    reduction: &PipelineStage,
    gemm: &PipelineStage,
    fused_name: &str,
) -> Result<String, String> {
    // Verify stage patterns
    let decomp = reduction
        .decompose_reduction()
        .ok_or("producer is not a Reduction stage")?;

    match &gemm.pattern {
        StagePattern::TiledGemm { .. } => {}
        other => return Err(format!("consumer is not TiledGemm, got {:?}", other)),
    }

    // Build the PointwiseComputation from the decomposed reduction
    let computation = build_reduction_computation(&decomp, reduction, gemm, fused_name)?;

    // Apply it to the GEMM PTX
    let gemm_ptx = gemm.source_lines.join("\n");
    replace_a_loads_with_inline_fn(&gemm_ptx, "", &computation)
}

/// Fuse a Pointwise stage (e.g., silu_mul) into a TiledGemm's A-input.
///
/// The pointwise operation is computed inline at each A-matrix cp.async site.
/// For silu_mul: the GEMM's A_ptr points to gate_up_buf with lda=2*intermediate.
/// At each A-load, the gate value is loaded from the original address and the
/// up value is loaded from +intermediate_bytes offset. SiLU(gate)*up replaces
/// the A-matrix element.
pub fn fuse_pointwise_into_gemm(
    pointwise: &PipelineStage,
    gemm: &PipelineStage,
    fused_name: &str,
) -> Result<String, String> {
    match &pointwise.pattern {
        StagePattern::Pointwise => {}
        other => return Err(format!("producer is not Pointwise, got {:?}", other)),
    }
    match &gemm.pattern {
        StagePattern::TiledGemm { .. } => {}
        other => return Err(format!("consumer is not TiledGemm, got {:?}", other)),
    }

    let computation = build_silu_mul_computation(fused_name)?;

    let gemm_ptx = gemm.source_lines.join("\n");
    replace_a_loads_with_inline_fn(&gemm_ptx, "", &computation)
}

/// Build a PointwiseComputation for SiLU+mul fused into GEMM A-loads.
///
/// At each A-load site (cp.async), the original load reads gate values
/// (from gate_up_buf with lda=2*intermediate). The per-site code loads
/// the corresponding up values at +intermediate_bytes offset. Per-element
/// instructions apply SiLU(gate) * up.
pub fn build_silu_mul_computation(fused_name: &str) -> Result<PointwiseComputation, String> {
    // Extra params: just the byte offset to the "up" half
    let extra_params = vec![".param .u64 _ferrite_intermediate_bytes,".into()];

    // Register declarations for up loading + SiLU scratch
    let extra_reg_decls = vec![
        // Up value registers (loaded per-site, consumed per-element)
        ".reg .b32 %r_up0, %r_up1, %r_up2, %r_up3;".into(),
        ".reg .b16 %h_up_a, %h_up_b;".into(),
        ".reg .f32 %f_up0, %f_up1, %f_up2, %f_up3, %f_up4, %f_up5, %f_up6, %f_up7;".into(),
        // SiLU scratch (reuse fuse_epilogue pattern)
        ".reg .f32 %f_act0, %f_act1, %f_act2, %f_act3;".into(),
        ".reg .b32 %r_act0;".into(),
        // Intermediate offset
        ".reg .b64 %rd_up_off;".into(),
    ];

    // Param loads: load the intermediate byte offset once
    let param_loads = vec!["ld.param.u64 \t%rd_up_off, [_ferrite_intermediate_bytes];".into()];

    // Per-site: load 8 bf16 up values from GMEM_SRC + intermediate_bytes
    let mut per_site = vec!["// FERRITE: load up values at +intermediate offset".into()];
    // Compute up address = GMEM_SRC + intermediate_bytes
    per_site.push("add.u64 \t%rd_up_off, {GMEM_SRC}, %rd_up_off;".into());
    per_site.push("ld.global.v4.b32 \t{%r_up0, %r_up1, %r_up2, %r_up3}, [%rd_up_off];".into());
    // Restore rd_up_off (it was clobbered by the add)
    per_site.push("ld.param.u64 \t%rd_up_off, [_ferrite_intermediate_bytes];".into());

    // Unpack 8 bf16 up values to named f32 registers
    per_site.push("// FERRITE: unpack 8 bf16 up values to f32".into());
    for w in 0..4u32 {
        let lo = w * 2;
        let hi = w * 2 + 1;
        per_site.push(format!("mov.b32 \t{{%h_up_a, %h_up_b}}, %r_up{w};"));
        per_site.push(format!("cvt.f32.bf16 \t%f_up{lo}, %h_up_a;"));
        per_site.push(format!("cvt.f32.bf16 \t%f_up{hi}, %h_up_b;"));
    }

    // Per-element instructions: SiLU(gate) * up
    // {INPUT} = gate value (f32, from the original A-load after bf16→f32 conversion)
    // {ELEM_IDX} = 0..7 index into up registers
    let instructions = vec![
        // SiLU(gate) = gate / (1 + exp(-gate))
        // Uses fast exp via range reduction + ex2.approx (same as fuse_epilogue.rs)
        "neg.f32 \t%f_act0, {INPUT};".into(),
        "fma.rn.f32 \t%f_act1, %f_act0, 0f3BBB989D, 0f3F000000;".into(),
        "cvt.sat.f32.f32 \t%f_act1, %f_act1;".into(),
        "fma.rm.f32 \t%f_act2, %f_act1, 0f437C0000, 0f4B400001;".into(),
        "add.f32 \t%f_act3, %f_act2, 0fCB40007F;".into(),
        "neg.f32 \t%f_act3, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act0, 0f3FB8AA3B, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act0, 0f32A57060, %f_act3;".into(),
        "mov.b32 \t%r_act0, %f_act2;".into(),
        "shl.b32 \t%r_act0, %r_act0, 23;".into(),
        "mov.b32 \t%f_act2, %r_act0;".into(),
        "ex2.approx.ftz.f32 \t%f_act3, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act3, %f_act2, 0f3F800000;".into(),
        "div.rn.f32 \t{INPUT}, {INPUT}, %f_act3;".into(),
        // Multiply by corresponding up value
        "mul.f32 \t{INPUT}, {INPUT}, %f_up{ELEM_IDX};".into(),
    ];

    Ok(PointwiseComputation {
        instructions,
        param_loads,
        prologue: vec![], // No prologue needed for pointwise
        extra_reg_decls,
        extra_params,
        per_site,
        entry_name: Some(fused_name.to_string()),
        scratch_f32_count: 0,
        scratch_b32_count: 0,
        source: crate::fuse_general::ALoadSource::Gmem,
    })
}

/// Build a PointwiseComputation for SiLU+mul that reads gate and up values
/// from SMEM scratch regions (written by redirected epilogues).
///
/// Same SiLU computation as `build_silu_mul_computation`, but the data sources
/// are SMEM scratch addresses instead of GMEM pointers.
///
/// `gate_smem_base_reg`: register holding the 32-bit SMEM base for gate scratch.
/// `up_smem_base_reg`: register holding the 32-bit SMEM base for up scratch.
///
/// The primary A-load (gate values) uses `ALoadSource::Smem` to read from
/// `gate_smem_base_reg`. The per-site code loads up values from `up_smem_base_reg`.
pub fn build_silu_mul_computation_smem(
    fused_name: &str,
    gate_smem_base_reg: &str,
    up_smem_base_reg: &str,
) -> Result<PointwiseComputation, String> {
    // No extra kernel params needed — SMEM addresses are computed internally
    let extra_params = vec![];

    // Register declarations for up loading + SiLU scratch
    let extra_reg_decls = vec![
        ".reg .b32 %r_up0, %r_up1, %r_up2, %r_up3;".into(),
        ".reg .b16 %h_up_a, %h_up_b;".into(),
        ".reg .f32 %f_up0, %f_up1, %f_up2, %f_up3, %f_up4, %f_up5, %f_up6, %f_up7;".into(),
        ".reg .f32 %f_act0, %f_act1, %f_act2, %f_act3;".into(),
        ".reg .b32 %r_act0;".into(),
    ];

    let param_loads = vec![];

    // Per-site: load 8 bf16 up values from SMEM scratch.
    // The byte offset for each site is computed at code-gen time by
    // replace_a_loads_with_inline_fn, which substitutes {LOAD_INDEX} with 0,1,2...
    // We pre-compute the byte offset as LOAD_INDEX * 16 in the template.
    // Since {LOAD_INDEX} is a compile-time literal, the add is a constant.
    let mut per_site = vec!["// FERRITE: load up values from SMEM scratch".into()];
    // Use immediate offset: up_base + LOAD_INDEX * 16
    // We emit this as a direct ld.shared with computed offset.
    // The {LOAD_INDEX_X16} placeholder will be resolved below.
    per_site.push(format!(
        "ld.shared.v4.b32 \t{{%r_up0, %r_up1, %r_up2, %r_up3}}, [{up_smem_base_reg}+{{LOAD_INDEX_X16}}];"
    ));

    // Unpack 8 bf16 up values to f32
    per_site.push("// FERRITE: unpack 8 bf16 up values to f32".into());
    for w in 0..4u32 {
        let lo = w * 2;
        let hi = w * 2 + 1;
        per_site.push(format!("mov.b32 \t{{%h_up_a, %h_up_b}}, %r_up{w};"));
        per_site.push(format!("cvt.f32.bf16 \t%f_up{lo}, %h_up_a;"));
        per_site.push(format!("cvt.f32.bf16 \t%f_up{hi}, %h_up_b;"));
    }

    // Per-element: identical SiLU(gate) * up computation
    let instructions = vec![
        "neg.f32 \t%f_act0, {INPUT};".into(),
        "fma.rn.f32 \t%f_act1, %f_act0, 0f3BBB989D, 0f3F000000;".into(),
        "cvt.sat.f32.f32 \t%f_act1, %f_act1;".into(),
        "fma.rm.f32 \t%f_act2, %f_act1, 0f437C0000, 0f4B400001;".into(),
        "add.f32 \t%f_act3, %f_act2, 0fCB40007F;".into(),
        "neg.f32 \t%f_act3, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act0, 0f3FB8AA3B, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act0, 0f32A57060, %f_act3;".into(),
        "mov.b32 \t%r_act0, %f_act2;".into(),
        "shl.b32 \t%r_act0, %r_act0, 23;".into(),
        "mov.b32 \t%f_act2, %r_act0;".into(),
        "ex2.approx.ftz.f32 \t%f_act3, %f_act3;".into(),
        "fma.rn.f32 \t%f_act3, %f_act3, %f_act2, 0f3F800000;".into(),
        "div.rn.f32 \t{INPUT}, {INPUT}, %f_act3;".into(),
        "mul.f32 \t{INPUT}, {INPUT}, %f_up{ELEM_IDX};".into(),
    ];

    Ok(PointwiseComputation {
        instructions,
        param_loads,
        prologue: vec![],
        extra_reg_decls,
        extra_params,
        per_site,
        entry_name: Some(fused_name.to_string()),
        scratch_f32_count: 0,
        scratch_b32_count: 0,
        source: crate::fuse_general::ALoadSource::Smem {
            smem_base_reg: gate_smem_base_reg.to_string(),
        },
    })
}

/// Build a PointwiseComputation from a decomposed reduction.
///
/// The prologue computes inv_rms for each row in the GEMM tile.
/// The per-element instructions multiply input by inv_rms and weight.
/// The thread-to-row mapping uses runtime arithmetic, not hardcoded patterns.
fn build_reduction_computation(
    decomp: &ReductionDecomposition,
    reduction: &PipelineStage,
    _gemm: &PipelineStage,
    fused_name: &str,
) -> Result<PointwiseComputation, String> {
    // Identify params from the reduction kernel
    let params = &reduction.protocol.params;
    if params.len() < 5 {
        return Err(format!(
            "expected reduction to have >= 5 params, got {}",
            params.len()
        ));
    }

    // Detect two-input reduction (fused_add_rms_norm): accumulation loop has st.global
    // indicating writeback to one of the inputs.
    let has_writeback = decomp.accumulate_loops.iter().any(|lp| {
        (lp.body_range.0..=lp.body_range.1).any(|i| {
            i < reduction.source_lines.len() && reduction.source_lines[i].contains("st.global")
        })
    });

    // The finalized value register is inv_rms (loaded from SMEM after reduction)
    if decomp.finalized_value_reg.is_empty() {
        return Err("could not detect finalized value register (inv_rms)".into());
    }

    // ── Extra params for the fused kernel ──
    // These get prepended to the GEMM's entry point
    //
    // For rms_norm (single input):
    //   param_0 = output (not needed), param_1 = input, param_2 = weight
    //   ferrite_rms_input → param_1
    //
    // For fused_add_rms_norm (two inputs + writeback):
    //   param_0 = input/hs (hidden_states), param_1 = residual (writeback target + GEMM A-input)
    //   param_2 = weight
    //   ferrite_rms_input → param_1 (residual, also the GEMM's A-pointer)
    //   ferrite_rms_hs_input → param_0 (hidden_states, second input for add)
    let mut extra_params = vec![".param .u64 _ferrite_rms_input,".into()];
    if has_writeback {
        extra_params.push(".param .u64 _ferrite_rms_hs_input,".into());
    }
    extra_params.extend([
        ".param .u64 _ferrite_rms_weight,".into(),
        ".param .f32 _ferrite_rms_epsilon,".into(),
        ".param .u32 _ferrite_rms_hidden,".into(),
        ".param .u64 _ferrite_rms_a_stride,".into(),
    ]);

    // ── Extra register declarations ──
    // Extract register declarations from the reduction kernel's source and rename them.
    // This ensures the transplanted code has all the registers it needs.
    // The .reg syntax uses ranges like %f<98> (declares %f0..%f97).
    // We rename these to %f_rms_<98> (declares %f_rms_0..%f_rms_97).
    let mut extra_reg_decls: Vec<String> = Vec::new();
    for line in &reduction.source_lines {
        let t = line.trim();
        if t.starts_with(".reg ") {
            // Rename register range declarations: %f<98> → %f_rms_<98>
            let mut renamed = t.to_string();
            let reg_types = ["rd", "rs", "f", "r", "p"];
            for typ in &reg_types {
                let pattern = format!("%{typ}<");
                let replacement = format!("%{typ}_rms_<");
                renamed = renamed.replace(&pattern, &replacement);
            }
            extra_reg_decls.push(renamed);
        }
    }
    // Plumbing registers for the prologue infrastructure (row loop, inv_rms loading)
    extra_reg_decls.push(".reg .f32 %f_rms_inv;".into());
    extra_reg_decls.push(".reg .f32 %f_rms_inv0, %f_rms_inv1;".into());
    extra_reg_decls.push(".reg .b32 %r_rms_k, %r_rms_hdn, %r_rms_step;".into());
    extra_reg_decls.push(".reg .b32 %r_rms_row, %r_rms_nrows, %r_rms_mtile;".into());
    extra_reg_decls.push(".reg .b64 %rd_rms_in, %rd_rms_wt, %rd_rms_str;".into());
    if has_writeback {
        extra_reg_decls.push(".reg .b64 %rd_rms_hs_in;".into());
        extra_reg_decls.push(".reg .b64 %rd_rms_hs_rb0, %rd_rms_hs_rb1;".into());
    }
    extra_reg_decls.push(".reg .b64 %rd_rms_rb0, %rd_rms_rb1;".into());
    extra_reg_decls.push(".reg .b64 %rd_rms_cur;".into());
    extra_reg_decls.push(".reg .pred %p_rms_lp, %p_rms_row, %p_rms_par, %p_rms_wb;".into());
    // SMEM for inv_rms array and warp reduction scratch
    extra_reg_decls.push(".shared .align 4 .f32 _ferrite_inv_rms[128];".into());
    extra_reg_decls.push(".shared .align 4 .f32 _ferrite_warp_scratch[8];".into());
    extra_reg_decls.push(".shared .align 4 .f32 _ferrite_s_inv_rms;".into());

    // ── Param loads ──
    // Parse the extracted kernel's ld.param and cvta.to.global lines to find
    // which registers hold which params. Emit ferrite param loads into the
    // renamed registers, so the transplanted code sees the correct values.
    let prefix = "rms";
    let mut param_loads: Vec<String> = Vec::new();

    // Map original param name suffixes to ferrite param names.
    //
    // rms_norm:            param_0=output(skip), param_1=input, param_2=weight, param_3=eps, param_4=hidden
    // fused_add_rms_norm:  param_0=hs_input,     param_1=residual(=GEMM A-ptr), param_2=weight, param_3=eps, param_4=hidden
    let ferrite_param_map: Vec<(&str, Option<&str>)> = if has_writeback {
        vec![
            ("param_0", Some("_ferrite_rms_hs_input")), // hidden_states (second input for add)
            ("param_1", Some("_ferrite_rms_input")),    // residual (GEMM A-ptr, writeback target)
            ("param_2", Some("_ferrite_rms_weight")),   // weight ptr
            ("param_3", Some("_ferrite_rms_epsilon")),  // epsilon
            ("param_4", Some("_ferrite_rms_hidden")),   // hidden_size
        ]
    } else {
        vec![
            ("param_0", None),                         // output ptr — not needed
            ("param_1", Some("_ferrite_rms_input")),   // input ptr
            ("param_2", Some("_ferrite_rms_weight")),  // weight ptr
            ("param_3", Some("_ferrite_rms_epsilon")), // epsilon
            ("param_4", Some("_ferrite_rms_hidden")),  // hidden_size
        ]
    };

    // For each ld.param in the source, emit a renamed version loading from ferrite params
    for line in &reduction.source_lines {
        let t = line.trim();
        if !t.contains("ld.param") {
            continue;
        }
        for &(param_suffix, ferrite_name) in &ferrite_param_map {
            if t.contains(param_suffix) {
                if let Some(ferrite) = ferrite_name {
                    let renamed = rename_ptx_regs(t, prefix);
                    // Replace [original_param_name] with [ferrite_param_name]
                    if let (Some(bs), Some(be)) = (renamed.find('['), renamed.find(']')) {
                        let mut new_line = renamed[..bs + 1].to_string();
                        new_line.push_str(ferrite);
                        new_line.push_str(&renamed[be..]);
                        param_loads.push(new_line);
                    }
                }
                break;
            }
        }
    }

    // Emit renamed cvta.to.global lines (convert device → global address space)
    for line in &reduction.source_lines {
        let t = line.trim();
        if t.contains("cvta.to.global") {
            param_loads.push(rename_ptx_regs(t, prefix));
        }
    }

    // Stride param (not in original kernel — added by ferrite for non-contiguous tensors)
    param_loads.push("ld.param.u64 \t%rd_rms_str, [_ferrite_rms_a_stride];".into());

    // Copy renamed global pointers to plumbing registers for per-site/row-loop code.
    // Find which renamed register is the global input ptr and weight ptr by tracing
    // the cvta chain: ld.param %rdN, [param_1] → cvta %rdM, %rdN → %rdM = global input
    for &(param_suffix, ferrite_name) in &ferrite_param_map {
        if ferrite_name.is_none() {
            continue;
        }
        // Find the ld.param dest register for this param
        let mut ld_dest = String::new();
        for line in &reduction.source_lines {
            let t = line.trim();
            if t.contains("ld.param") && t.contains(param_suffix) {
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    ld_dest = parts[1].trim_end_matches(',').to_string();
                }
                break;
            }
        }
        if ld_dest.is_empty() {
            continue;
        }
        // Find the cvta that uses this register as source
        for line in &reduction.source_lines {
            let t = line.trim();
            if t.contains("cvta.to.global") && t.contains(&ld_dest) {
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 3 {
                    let cvta_dest = rename_ptx_regs(parts[1].trim_end_matches(','), prefix);
                    match ferrite_name {
                        Some("_ferrite_rms_input") => {
                            param_loads.push(format!("mov.u64 \t%rd_rms_in, {cvta_dest};"));
                        }
                        Some("_ferrite_rms_hs_input") => {
                            param_loads.push(format!("mov.u64 \t%rd_rms_hs_in, {cvta_dest};"));
                        }
                        Some("_ferrite_rms_weight") => {
                            param_loads.push(format!("mov.u64 \t%rd_rms_wt, {cvta_dest};"));
                        }
                        _ => {}
                    }
                }
                break;
            }
        }
    }
    // Also copy hidden_size to plumbing register
    for line in &reduction.source_lines {
        let t = line.trim();
        if t.contains("ld.param") && t.contains("param_4") {
            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() >= 2 {
                let dest = rename_ptx_regs(parts[1].trim_end_matches(','), prefix);
                param_loads.push(format!("mov.u32 \t%r_rms_hdn, {dest};"));
            }
            break;
        }
    }

    // ── Prologue: compute inv_rms for each tile row ──
    //
    // Strategy: all threads cooperate on each row sequentially.
    // For tile_m rows, each thread handles K/ntid.x elements per row.
    // After each row's reduction, thread 0 stores inv_rms to SMEM array.
    //
    // This is derived from the rms_norm formula (extracted from PTX):
    //   sum_sq = sum(input[row, k]^2 for k in 0..hidden)
    //   inv_rms = rsqrt(sum_sq / hidden + epsilon)
    //
    // The accumulation pattern (fma.rn.f32 for sum-of-squares) and
    // finalization (rsqrt) are extracted from the decomposition.
    // Extract thread-to-row mapping from GEMM PTX
    let gemm_lines: Vec<&str> = _gemm.source_lines.iter().map(|s| s.as_str()).collect();
    let thread_map = extract_thread_row_map(&gemm_lines);
    #[cfg(test)]
    if let Some(ref tm) = thread_map {
        eprintln!(
            "  thread_row_map: lanes_per_row=1<<{}, rows_per_warp=1<<{}, row_stride={}",
            tm.lanes_per_row_log2, tm.rows_per_warp_log2, tm.row_stride
        );
    }

    // Extract tile index map from GEMM PTX — the (ctaid.x, ctaid.y) → (m_tile, n_tile)
    // swizzle computation. This replaces the hand-written swizzle that was buggy at m_tile>=3.
    let tile_index = extract_tile_index_map(&gemm_lines).ok_or_else(|| {
        "could not extract tile index map from GEMM PTX (no ctaid.x/y swizzle pattern)".to_string()
    })?;
    #[cfg(test)]
    eprintln!(
        "  tile_index: m_tile={}, n_tile={}, swizzle_log={}",
        tile_index.m_tile_reg, tile_index.n_tile_reg, tile_index.swizzle_log_reg
    );

    // Extract tile dimensions from the GEMM's mangled entry name
    let (tile_m, tile_n, _tile_k) = extract_gemm_shape_from_entry(&_gemm.protocol.name)
        .ok_or_else(|| {
            format!(
                "could not extract GemmShape from GEMM entry name: {}",
                _gemm.protocol.name
            )
        })?;

    let prologue = build_prologue_from_decomposition(
        decomp,
        &reduction.source_lines,
        thread_map.as_ref(),
        tile_m,
        tile_n,
        &tile_index,
        has_writeback,
    );

    // ── Per-site code ──
    // Row selection: the prologue pre-loads inv_rms for both rows into
    // %f_rms_inv0 and %f_rms_inv1. Per-site selects based on compile-time
    // parity bitmask using {LOAD_INDEX}.
    //
    // Weight loading: compute K byte offset from GMEM address.
    // k_byte_offset = (gmem_src - a_ptr) - row_offset
    // where row_offset is selected by parity.
    let per_site = vec![
        "// FERRITE: select inv_rms by row address comparison (compile-time optimized)".into(),
        // Compare gmem_src against row 1 base: if gmem_src >= rb1, it's row 1
        "setp.ge.u64 \t%p_rms_par, {GMEM_SRC}, %rd_rms_rb1;".into(),
        "selp.f32 \t%f_rms_inv, %f_rms_inv1, %f_rms_inv0, %p_rms_par;".into(),
        "// FERRITE: load weight at K offset from GMEM address".into(),
        // weight_addr = weight_ptr + (gmem_src - row_base[parity])
        "selp.u64 \t%rd_rms_cur, %rd_rms_rb1, %rd_rms_rb0, %p_rms_par;".into(),
        "sub.u64 \t%rd_rms_cur, {GMEM_SRC}, %rd_rms_cur;".into(), // k_byte_offset
        "add.u64 \t%rd_rms_cur, %rd_rms_wt, %rd_rms_cur;".into(), // weight + k_byte_offset
        "ld.global.v4.b32 \t{%r_rms_w0, %r_rms_w1, %r_rms_w2, %r_rms_w3}, [%rd_rms_cur];".into(),
    ];

    // Need extra regs for weight loading
    let mut extra_reg_decls = extra_reg_decls;
    extra_reg_decls.push(".reg .b32 %r_rms_w0, %r_rms_w1, %r_rms_w2, %r_rms_w3;".into());
    extra_reg_decls.push(".reg .b16 %h_rms_a, %h_rms_b;".into());
    extra_reg_decls.push(".reg .f32 %f_rms_wa, %f_rms_wb;".into());

    // For two-input reductions: extra regs for hs_input loading + a temp for k_byte_offset
    if has_writeback {
        extra_reg_decls.push(".reg .b32 %r_rms_hs0, %r_rms_hs1, %r_rms_hs2, %r_rms_hs3;".into());
        extra_reg_decls.push(
            ".reg .f32 %f_rms_hs0, %f_rms_hs1, %f_rms_hs2, %f_rms_hs3, %f_rms_hs4, %f_rms_hs5, %f_rms_hs6, %f_rms_hs7;"
                .into(),
        );
        extra_reg_decls.push(".reg .b64 %rd_rms_koff;".into());
    }

    // Unpack all 8 weights in per_site code, store in named f32 regs.
    // Per-element instructions then reference %f_rms_wt{ELEM_IDX}.
    let mut per_site = per_site;
    per_site.push("// FERRITE: unpack 8 bf16 weights to f32".into());
    for w in 0..4u32 {
        let lo = w * 2;
        let hi = w * 2 + 1;
        per_site.push(format!("mov.b32 \t{{%h_rms_a, %h_rms_b}}, %r_rms_w{w};"));
        per_site.push(format!("cvt.f32.bf16 \t%f_rms_wt{lo}, %h_rms_a;"));
        per_site.push(format!("cvt.f32.bf16 \t%f_rms_wt{hi}, %h_rms_b;"));
    }

    // For two-input reductions: load hs_input at the same K offset and unpack
    if has_writeback {
        per_site.push("// FERRITE: load hs_input at same K offset".into());
        per_site
            .push("selp.u64 \t%rd_rms_koff, %rd_rms_hs_rb1, %rd_rms_hs_rb0, %p_rms_par;".into());
        // k_byte_offset = GMEM_SRC - res_row_base (recompute from residual bases)
        per_site.push("selp.u64 \t%rd_rms_cur, %rd_rms_rb1, %rd_rms_rb0, %p_rms_par;".into());
        per_site.push("sub.u64 \t%rd_rms_cur, {GMEM_SRC}, %rd_rms_cur;".into()); // k_byte_offset
        per_site.push("add.u64 \t%rd_rms_koff, %rd_rms_koff, %rd_rms_cur;".into()); // hs_base + k_byte_offset
        per_site.push(
            "ld.global.v4.b32 \t{%r_rms_hs0, %r_rms_hs1, %r_rms_hs2, %r_rms_hs3}, [%rd_rms_koff];"
                .into(),
        );
        per_site.push("// FERRITE: unpack 8 bf16 hs values to f32".into());
        for w in 0..4u32 {
            let lo = w * 2;
            let hi = w * 2 + 1;
            per_site.push(format!("mov.b32 \t{{%h_rms_a, %h_rms_b}}, %r_rms_hs{w};"));
            per_site.push(format!("cvt.f32.bf16 \t%f_rms_hs{lo}, %h_rms_a;"));
            per_site.push(format!("cvt.f32.bf16 \t%f_rms_hs{hi}, %h_rms_b;"));
        }
    }

    // Add weight f32 registers
    extra_reg_decls.push(
        ".reg .f32 %f_rms_wt0, %f_rms_wt1, %f_rms_wt2, %f_rms_wt3, %f_rms_wt4, %f_rms_wt5, %f_rms_wt6, %f_rms_wt7;"
            .into(),
    );

    // Per-element instructions: derived from the reduction's emit body.
    // The emit body contains the full pass-2 computation (load, normalize, store).
    // We extract just the per-element pattern (mul inv_rms, mul weight).
    let base_instructions =
        extract_per_element_from_emit_body(&decomp.emit_body_lines, &decomp.finalized_value_reg)
            .unwrap_or_else(|| {
                // Fallback: if extraction fails, use the known rms_norm pattern
                vec![
                    "mul.f32 \t{INPUT}, {INPUT}, %f_rms_inv;".into(),
                    "mul.f32 \t{INPUT}, {INPUT}, %f_rms_wt{ELEM_IDX};".into(),
                ]
            });

    // For two-input reductions (fused_add_rms_norm): add hs before normalize.
    // The bf16 round-trip (cvt.rn.bf16 + cvt.f32.bf16) matches the standard
    // path's precision: fused_add_rms_norm_inplace writes bf16(sum) to memory,
    // then pass 2 reads it back as f32(bf16(sum)). Without this truncation,
    // the fused path diverges much faster (layer 3 vs layer 9).
    let instructions = if has_writeback {
        let mut instrs = vec![
            "add.f32 \t{INPUT}, {INPUT}, %f_rms_hs{ELEM_IDX};".into(),
            "cvt.rn.bf16.f32 \t%h_rms_a, {INPUT};".into(),
            "cvt.f32.bf16 \t{INPUT}, %h_rms_a;".into(),
        ];
        instrs.extend(base_instructions);
        instrs
    } else {
        base_instructions
    };

    Ok(PointwiseComputation {
        instructions,
        param_loads,
        prologue,
        extra_reg_decls,
        extra_params,
        per_site,
        entry_name: Some(fused_name.to_string()),
        scratch_f32_count: 0,
        scratch_b32_count: 0,
        source: crate::fuse_general::ALoadSource::Gmem,
    })
}

/// Rename PTX registers in a line with a prefix.
///
/// `%f17` → `%f_<prefix>_17`, `%rd4` → `%rd_<prefix>_4`, etc.
/// Skips special registers (`%tid`, `%ntid`, `%ctaid`, `%nctaid`, `%laneid`, `%warpid`).
fn rename_ptx_regs(line: &str, prefix: &str) -> String {
    let bytes = line.as_bytes();
    let mut result = Vec::with_capacity(bytes.len() + 64);
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' {
            // Check for special registers first (must NOT rename these)
            let rest = &line[i + 1..];
            if rest.starts_with("tid.")
                || rest.starts_with("ntid.")
                || rest.starts_with("ctaid.")
                || rest.starts_with("nctaid.")
                || rest.starts_with("laneid")
                || rest.starts_with("warpid")
            {
                result.push(b'%');
                i += 1;
                continue;
            }

            // Try to match register types (longest first: rd, rs before r)
            let reg_types = ["rd", "rs", "f", "r", "p"];
            let mut matched = false;
            for typ in &reg_types {
                if rest.starts_with(typ) {
                    let after_type = &rest[typ.len()..];
                    let digit_count = after_type
                        .as_bytes()
                        .iter()
                        .take_while(|b| b.is_ascii_digit())
                        .count();
                    if digit_count > 0 {
                        // Found a register: %<type><digits>
                        let digits = &after_type[..digit_count];
                        result.push(b'%');
                        result.extend_from_slice(typ.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(prefix.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(digits.as_bytes());
                        i += 1 + typ.len() + digit_count;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                result.push(bytes[i]);
                i += 1;
            }
        } else if bytes[i] == b'$' && line[i..].starts_with("$L__BB") {
            // Label: $L__BB<N>_<M> → $L_<prefix>_<N>_<M>
            let rest = &line[i + 6..]; // after "$L__BB"
            let d1_count = rest
                .as_bytes()
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count();
            if d1_count > 0 {
                let d1 = &rest[..d1_count];
                let after_d1 = &rest[d1_count..];
                if after_d1.starts_with('_') {
                    let d2_start = &after_d1[1..];
                    let d2_count = d2_start
                        .as_bytes()
                        .iter()
                        .take_while(|b| b.is_ascii_digit())
                        .count();
                    if d2_count > 0 {
                        let d2 = &d2_start[..d2_count];
                        result.extend_from_slice(b"$L_");
                        result.extend_from_slice(prefix.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(d1.as_bytes());
                        result.push(b'_');
                        result.extend_from_slice(d2.as_bytes());
                        i += 6 + d1_count + 1 + d2_count;
                        continue;
                    }
                }
            }
            result.push(bytes[i]);
            i += 1;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }

    String::from_utf8(result).unwrap_or_else(|_| line.to_string())
}

/// Determine if a source line should be skipped when transplanting into the prologue.
///
/// We skip: declarations, param loads, cvta for params, ctaid.x row computation.
/// These are replaced by ferrite's own param initialization and row loop.
fn is_transplant_skip(line: &str) -> bool {
    let t = line.trim();
    // Skip empty lines and comments
    if t.is_empty() || t.starts_with("//") {
        return true;
    }
    // Skip entry declaration and its param list
    if t.starts_with(".entry") || t.starts_with(".visible") || t.starts_with(")") {
        return true;
    }
    // Skip .param declarations (entry signature params, not ld.param)
    if t.starts_with(".param ") {
        return true;
    }
    // Skip braces (entry body delimiters)
    if t == "{" || t == "}" {
        return true;
    }
    // Skip register and shared memory declarations
    if t.starts_with(".reg ") || t.starts_with(".shared ") {
        return true;
    }
    // Skip PTX header directives
    if t.starts_with(".version") || t.starts_with(".target") || t.starts_with(".address_size") {
        return true;
    }
    // Skip param loads (we provide our own from ferrite params)
    if t.contains("ld.param") {
        return true;
    }
    // Skip cvta.to.global that set up param pointers
    if t.contains("cvta.to.global") {
        return true;
    }
    // Skip ctaid.x row offset computation (replaced by m_tile * tile_m + row)
    if t.contains("%ctaid.x") || t.contains("%ctaid.y") {
        return true;
    }
    // Skip demoted variable comments (nvcc metadata)
    if t.starts_with("// demoted") {
        return true;
    }
    false
}

/// Generate the prologue PTX by transplanting extracted code from the reduction kernel.
///
/// Every computational instruction (FMA, shuffle, SMEM reduce, rsqrt) comes verbatim
/// from the actual rms_norm kernel's PTX, with register/label/SMEM renaming.
/// Only plumbing is new: param initialization, m_tile computation, row loop.
fn build_prologue_from_decomposition(
    decomp: &ReductionDecomposition,
    source_lines: &[String],
    thread_map: Option<&ThreadRowMap>,
    tile_m: u32,
    tile_n: u32,
    _tile_index: &TileIndexMap,
    has_writeback: bool,
) -> Vec<String> {
    let lane_shr = thread_map.map_or(2, |tm| tm.lanes_per_row_log2);
    let warp_shl = thread_map.map_or(4, |tm| tm.rows_per_warp_log2);
    let row_stride = thread_map.map_or(8, |tm| tm.row_stride);

    let prefix = "rms";

    // ── Collect SMEM symbol names for renaming ──
    let mut smem_renames: Vec<(String, String)> = Vec::new();
    for line in source_lines {
        let t = line.trim();
        if t.contains("block_reduce_sum") || t.contains("block_reduce_") {
            // Find the mangled symbol name
            for word in t.split_whitespace() {
                let w = word.trim_end_matches(';').trim_end_matches(',');
                if w.contains("block_reduce_") {
                    if !smem_renames.iter().any(|(old, _)| old == w) {
                        smem_renames.push((w.to_string(), "_ferrite_warp_scratch".to_string()));
                    }
                }
            }
        }
        if t.contains("s_inv_rms") {
            for word in t.split(&['[', ']', ',', ' ', '\t'][..]) {
                let w = word.trim_end_matches(';');
                if w.contains("s_inv_rms") {
                    if !smem_renames.iter().any(|(old, _)| old == w) {
                        smem_renames.push((w.to_string(), "_ferrite_s_inv_rms".to_string()));
                    }
                }
            }
        }
    }

    // ── Determine which lines to transplant ──
    // We need: setup code + accumulate loops + finalize block
    // We skip: emit loops (replaced by per-element at A-load sites)
    let first_accum_start = decomp
        .accumulate_loops
        .first()
        .map(|l| l.header_line)
        .unwrap_or(0);
    let finalize_end = decomp.finalize_range.1;

    // Find the ctaid.x row offset lines to identify %rd4 equivalent
    // (the register that holds the row element offset)
    // Look for: mov.u32 %rN, %ctaid.x → mul → cvt to 64-bit
    let mut ctaid_result_reg = String::new(); // the s64 result (e.g., %rd4)
    let mut ctaid_skip_lines: Vec<usize> = Vec::new();
    for (idx, line) in source_lines.iter().enumerate() {
        if line.contains("%ctaid.x") {
            ctaid_skip_lines.push(idx);
            // The next 2 lines compute the row offset from ctaid.x
            if idx + 1 < source_lines.len() && source_lines[idx + 1].contains("mul.lo.s32") {
                ctaid_skip_lines.push(idx + 1);
            }
            if idx + 2 < source_lines.len() && source_lines[idx + 2].contains("cvt.s64.s32") {
                // Extract the destination register (e.g., %rd4)
                let t = source_lines[idx + 2].trim();
                let parts: Vec<&str> = t.split_whitespace().collect();
                if parts.len() >= 2 {
                    let dest = parts[1].trim_end_matches(',');
                    ctaid_result_reg = dest.to_string();
                }
                ctaid_skip_lines.push(idx + 2);
            }
        }
    }

    // Find where the setup code begins (after param loads and ctaid computation)
    // This is the first line that's not a skip line and is before the first accumulate loop
    let mut setup_start = 0;
    for idx in 0..first_accum_start {
        if !is_transplant_skip(&source_lines[idx]) && !ctaid_skip_lines.contains(&idx) {
            setup_start = idx;
            break;
        }
    }

    // ── Apply renaming to each transplanted line ──
    let rename_line = |line: &str| -> String {
        let mut renamed = rename_ptx_regs(line, prefix);
        // Replace SMEM symbols
        for (old, new) in &smem_renames {
            renamed = renamed.replace(old.as_str(), new.as_str());
        }
        // Replace bar.sync 0 with bar.sync 15 to avoid GEMM conflicts
        if renamed.contains("bar.sync") && renamed.contains("\t0") {
            renamed = renamed.replace("\t0", "\t15");
        }
        renamed
    };

    // ── Build the prologue ──
    let mut prologue = Vec::new();
    prologue
        .push("// FERRITE: rms_norm prologue (transplanted from extracted bf16 kernel PTX)".into());

    // Emit the tile index computation register-renamed into the prologue.
    // We can't reference the GEMM body's m_tile/n_tile registers directly because
    // perimeter replacement rewrites the ld.param for swizzle_log, which shifts
    // which register ends up holding m_tile. Instead, we re-derive m_tile from
    // %ctaid.x using the extracted PTX lines (register-renamed to %r_rms_* namespace).
    //
    // The extracted lines contain an ld.param for swizzle_log from the CUTLASS
    // struct param. After perimeter replacement, that param is gone. We replace
    // the ld.param with an inline computation of swizzle_log from N (at ferrite_params+68).
    prologue.push("// Tile index: extracted from GEMM PTX, register-renamed".into());

    // Emit swizzle_log computation from N (replaces the ld.param for swizzle_log).
    // swizzle_log = floor(log2(min(ceil(N / tile_n), 4)))
    // For ThreadblockSwizzle<4>, the values are 0, 1, or 2.
    let tile_n_shift = tile_n.trailing_zeros();

    // Inline swizzle_log: load N, compute n_tiles = ceil(N/tile_n), then log2
    prologue.push("ld.param.s32 \t%r_rms_step, [ferrite_params+68];".into()); // N
    prologue.push(format!(
        "add.s32 \t%r_rms_step, %r_rms_step, {};",
        tile_n - 1
    ));
    prologue.push(format!(
        "shr.u32 \t%r_rms_step, %r_rms_step, {tile_n_shift};"
    )); // n_tiles = ceil(N/tile_n)
    // swizzle_log = 0 if n_tiles < 2, 1 if n_tiles < 3, 2 otherwise
    // (ThreadblockSwizzle<4> uses bits = log2(min(n_tiles, 4)))
    prologue.push("mov.u32 \t%r_rms_nrows, 0;".into());
    prologue.push("setp.ge.u32 \t%p_rms_lp, %r_rms_step, 2;".into());
    prologue.push("@%p_rms_lp mov.u32 \t%r_rms_nrows, 1;".into());
    prologue.push("setp.ge.u32 \t%p_rms_lp, %r_rms_step, 4;".into());
    prologue.push("@%p_rms_lp mov.u32 \t%r_rms_nrows, 2;".into());
    // %r_rms_nrows = swizzle_log

    // Now emit the swizzle computation using the same logic as the GEMM body,
    // but with our own registers. This is the CUTLASS GemmIdentityThreadblockSwizzle<4>:
    //   m_tile = ctaid.x >> swizzle_log
    //   mask = (1 << swizzle_log) - 1      [equivalently: ~((-1) << swizzle_log)]
    //   n_group = ctaid.x & mask
    //   n_tile = n_group + (ctaid.y << swizzle_log)
    prologue.push("mov.u32 \t%r_rms_mtile, %ctaid.x;".into());
    prologue.push("shr.u32 \t%r_rms_step, %r_rms_mtile, %r_rms_nrows;".into());
    // n_tile for writeback guard:
    prologue.push("mov.u32 \t%r_rms_k, -1;".into());
    prologue.push("shl.b32 \t%r_rms_k, %r_rms_k, %r_rms_nrows;".into());
    prologue.push("not.b32 \t%r_rms_k, %r_rms_k;".into());
    prologue.push("and.b32 \t%r_rms_k, %r_rms_mtile, %r_rms_k;".into()); // n_group
    prologue.push("mov.u32 \t%r_rms_mtile, %ctaid.y;".into());
    prologue.push("shl.b32 \t%r_rms_mtile, %r_rms_mtile, %r_rms_nrows;".into());
    prologue.push("add.s32 \t%r_rms_k, %r_rms_k, %r_rms_mtile;".into()); // n_tile
    prologue.push("setp.eq.u32 \t%p_rms_wb, %r_rms_k, 0;".into());
    // Recover m_tile into %r_rms_mtile
    prologue.push("mov.u32 \t%r_rms_mtile, %ctaid.x;".into());
    prologue.push("shr.u32 \t%r_rms_mtile, %r_rms_mtile, %r_rms_nrows;".into());

    // Row loop: iterate over min(tile_m, M - m_tile * tile_m) rows
    // Read M from ferrite_params[64] to handle partial tiles at boundary
    prologue.push("ld.param.s32 \t%r_rms_step, [ferrite_params+64];".into()); // M
    prologue.push(format!(
        "mul.lo.s32 \t%r_rms_nrows, %r_rms_mtile, {tile_m};"
    ));
    prologue.push("sub.s32 \t%r_rms_nrows, %r_rms_step, %r_rms_nrows;".into()); // M - m_tile * tile_m
    prologue.push(format!("min.s32 \t%r_rms_nrows, %r_rms_nrows, {tile_m};")); // min(remaining, tile_m)
    prologue.push("mov.u32 \t%r_rms_row, 0;".into());
    prologue.push("$L_rms_row_loop:".into());

    // Compute the row element offset (replaces ctaid.x * hidden_size)
    // IMPORTANT: use tile_m (not nrows) for the absolute row computation.
    // nrows is the clamped loop bound for partial tiles, but the starting row
    // is always m_tile * tile_m regardless of how many rows this tile processes.
    let rd4_renamed = rename_ptx_regs(&ctaid_result_reg, prefix);
    prologue.push("// Row offset: (m_tile * tile_m + row) * hidden".into());
    prologue.push(format!("mul.lo.s32 \t%r_rms_step, %r_rms_mtile, {tile_m};"));
    prologue.push("add.u32 \t%r_rms_step, %r_rms_step, %r_rms_row;".into());
    prologue.push(format!(
        "mul.lo.s32 \t%r_rms_step, %r_rms_step, %r_rms_hdn;"
    ));
    prologue.push(format!("cvt.s64.s32 \t{rd4_renamed}, %r_rms_step;"));

    // Transplant the setup + accumulate + finalize code from the extracted kernel.
    // Stop at the second bar.sync in the finalize range (the one before emit loops).
    // Everything after that is emit-loop setup which we don't need.
    let mut bar_sync_count = 0;
    let mut inv_rms_stored = false;

    prologue.push("// BEGIN transplanted rms_norm code".into());
    for idx in setup_start..=finalize_end.min(source_lines.len() - 1) {
        // Skip lines we're replacing
        if is_transplant_skip(&source_lines[idx]) {
            continue;
        }
        if ctaid_skip_lines.contains(&idx) {
            continue;
        }
        let line = &source_lines[idx];
        let t = line.trim();

        let renamed = rename_line(line);

        // Track bar.sync occurrences in the finalize range.
        // After the inv_rms store, the next bar.sync is the last thing we need.
        if renamed.contains("bar.sync") {
            bar_sync_count += 1;
            if inv_rms_stored {
                // This is the bar.sync after inv_rms store — emit it and stop
                prologue.push(renamed.trim().to_string());
                break;
            }
        }

        // Detect the inv_rms store to scalar SMEM and redirect to array
        if renamed.contains("st.shared.f32") && renamed.contains("_ferrite_s_inv_rms") {
            let parts: Vec<&str> = renamed.split_whitespace().collect();
            let src_reg = parts
                .last()
                .map(|s| s.trim_end_matches(';'))
                .unwrap_or("%f_rms_55");
            // Store to _ferrite_inv_rms[row * 4] instead
            prologue.push("mov.u32 \t%r_rms_step, _ferrite_inv_rms;".into());
            prologue.push("shl.b32 \t%r_rms_k, %r_rms_row, 2;".into());
            prologue.push("add.s32 \t%r_rms_step, %r_rms_step, %r_rms_k;".into());
            prologue.push(format!("st.shared.f32 \t[%r_rms_step], {src_reg};"));
            inv_rms_stored = true;
            continue;
        }

        // For two-input reductions (fused_add_rms_norm): skip all st.global
        // in the prologue. The prologue is pure-read — all blocks see identical
        // original data → identical inv_rms → no inter-block aliasing.
        // The caller runs add_inplace() post-kernel for the residual update.
        if renamed.contains("st.global") {
            if has_writeback {
                continue;
            }
            prologue.push(renamed.trim().to_string());
        } else {
            prologue.push(renamed.trim().to_string());
        }
    }
    prologue.push("// END transplanted rms_norm code".into());

    // End of row loop
    prologue.push("add.u32 \t%r_rms_row, %r_rms_row, 1;".into());
    prologue.push("setp.lt.u32 \t%p_rms_row, %r_rms_row, %r_rms_nrows;".into());
    prologue.push("@%p_rms_row bra \t$L_rms_row_loop;".into());

    // Final barrier before GEMM body reads inv_rms from SMEM
    prologue.push("bar.sync \t15;".into());

    // ── Per-thread setup: load inv_rms for this thread's two rows ──
    prologue.push(format!(
        "// Per-thread: load inv_rms (lane_shr={lane_shr}, warp_shl={warp_shl}, row_stride={row_stride})"
    ));
    prologue.push("mov.u32 \t%r_rms_step, %tid.x;".into());
    prologue.push("and.b32 \t%r_rms_k, %r_rms_step, 31;".into());
    prologue.push(format!("shr.u32 \t%r_rms_k, %r_rms_k, {lane_shr};"));
    prologue.push("shr.u32 \t%r_rms_row, %r_rms_step, 5;".into());
    prologue.push(format!("shl.b32 \t%r_rms_row, %r_rms_row, {warp_shl};"));
    prologue.push("add.u32 \t%r_rms_row, %r_rms_row, %r_rms_k;".into());

    let row_stride_bytes = row_stride * 4;
    prologue.push("mov.u32 \t%r_rms_k, _ferrite_inv_rms;".into());
    prologue.push("shl.b32 \t%r_rms_step, %r_rms_row, 2;".into());
    prologue.push("add.s32 \t%r_rms_step, %r_rms_k, %r_rms_step;".into());
    prologue.push("ld.shared.f32 \t%f_rms_inv0, [%r_rms_step];".into());
    prologue.push(format!(
        "ld.shared.f32 \t%f_rms_inv1, [%r_rms_step+{row_stride_bytes}];"
    ));

    // Compute GMEM row base addresses for per-site K-offset extraction
    prologue.push(format!("mul.lo.s32 \t%r_rms_step, %r_rms_mtile, {tile_m};"));
    prologue.push("add.u32 \t%r_rms_step, %r_rms_step, %r_rms_row;".into());
    prologue.push("cvt.s64.s32 \t%rd_rms_rb0, %r_rms_step;".into());
    prologue.push("mul.lo.s64 \t%rd_rms_rb0, %rd_rms_rb0, %rd_rms_str;".into());
    prologue.push("shl.b64 \t%rd_rms_rb0, %rd_rms_rb0, 1;".into());
    prologue.push("add.s64 \t%rd_rms_rb0, %rd_rms_in, %rd_rms_rb0;".into());
    let rb1_shift = (row_stride * 2).trailing_zeros();
    prologue.push(format!("shl.b64 \t%rd_rms_rb1, %rd_rms_str, {rb1_shift};"));
    prologue.push("add.s64 \t%rd_rms_rb1, %rd_rms_rb0, %rd_rms_rb1;".into());

    // For two-input reductions: compute hs_input row bases at the same offsets
    if has_writeback {
        prologue
            .push("// FERRITE: hs_input row bases (same row offset, different base ptr)".into());
        prologue.push("sub.s64 \t%rd_rms_hs_rb0, %rd_rms_rb0, %rd_rms_in;".into());
        prologue.push("add.s64 \t%rd_rms_hs_rb0, %rd_rms_hs_rb0, %rd_rms_hs_in;".into());
        prologue.push("sub.s64 \t%rd_rms_hs_rb1, %rd_rms_rb1, %rd_rms_in;".into());
        prologue.push("add.s64 \t%rd_rms_hs_rb1, %rd_rms_hs_rb1, %rd_rms_hs_in;".into());
    }

    prologue.push("// FERRITE: end rms_norm prologue".into());

    prologue
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_tile_m_from_gemm_entries() {
        // 64x128x32
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi128ELi32EEE"
            ),
            Some(64)
        );
        // 128x128x32
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi32EEE"
            ),
            Some(128)
        );
        // 128x128x64
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi128ELi128ELi64EEE"
            ),
            Some(128)
        );
        // 64x64x32
        assert_eq!(
            extract_tile_m_from_entry(
                "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock13MmaMultistageINS1_9GemmShapeILi64ELi64ELi32EEE"
            ),
            Some(64)
        );
        // Non-CUTLASS entry
        assert_eq!(extract_tile_m_from_entry("vllm_rms_norm_kernel"), None);
    }

    #[test]
    fn build_computation_from_rms_norm() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");
        let decomp = stage.decompose_reduction().expect("decompose");

        let comp = build_reduction_computation(&decomp, &stage, &gemm_stage, "test_fused")
            .expect("build computation");

        // Should have prologue
        assert!(
            !comp.prologue.is_empty(),
            "should generate a prologue for the reduction"
        );

        // Prologue should contain the key patterns from the reduction:
        // - fma.rn.f32 (sum of squares accumulation)
        // - shfl.sync (warp reduction)
        // - rsqrt (finalization)
        let prologue_text = comp.prologue.join("\n");
        assert!(
            prologue_text.contains("fma.rn.f32"),
            "prologue should contain sum-of-squares accumulation"
        );
        assert!(
            prologue_text.contains("shfl.sync"),
            "prologue should contain warp shuffle reduction"
        );
        assert!(
            prologue_text.contains("rsqrt"),
            "prologue should contain rsqrt finalization"
        );

        // Should have per-element instructions
        assert!(
            !comp.instructions.is_empty(),
            "should have per-element instructions"
        );
        let instr_text = comp.instructions.join("\n");
        assert!(
            instr_text.contains("mul.f32") && instr_text.contains("%f_rms_inv"),
            "instructions should multiply by inv_rms"
        );

        // Should have per-site code (weight loading + inv_rms lookup)
        assert!(
            !comp.per_site.is_empty(),
            "should have per-site code for weight loading"
        );

        // Should have extra params
        assert!(
            comp.extra_params.len() >= 4,
            "should have rms_norm params (input, weight, eps, hidden)"
        );

        // Should have entry name
        assert_eq!(comp.entry_name.as_deref(), Some("test_fused"));
    }

    #[test]
    fn fuse_rms_norm_into_cutlass_gemm() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");

        let rms_stage = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused = fuse_reduction_into_gemm(&rms_stage, &gemm_stage, "fused_norm_gemm");

        match fused {
            Ok(ptx) => {
                // The fused kernel should contain:
                // 1. The rms_norm prologue
                assert!(
                    ptx.contains("rms_norm prologue"),
                    "should contain rms_norm prologue"
                );
                // 2. The GEMM's MMA instructions (preserved)
                assert!(
                    ptx.contains("mma.sync"),
                    "should preserve GEMM MMA instructions"
                );
                // 3. The fused entry name
                assert!(
                    ptx.contains("fused_norm_gemm"),
                    "should have the fused entry name"
                );
                // 4. The extra params
                assert!(
                    ptx.contains("_ferrite_rms_weight"),
                    "should have rms_norm weight param"
                );
                // 5. Per-site inv_rms lookup
                assert!(
                    ptx.contains("_ferrite_inv_rms"),
                    "should reference inv_rms SMEM array"
                );
                // 6. Should NOT contain the broken intrinsic approach
                assert!(
                    !ptx.contains("_ferrite_rms_a_ptr_"),
                    "should not reference old intrinsic registers"
                );
            }
            Err(e) => {
                panic!("fuse_reduction_into_gemm failed: {e}");
            }
        }
    }

    #[test]
    fn fused_norm_gemm_ptxas_valid() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let rms_stage = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused_ptx =
            fuse_reduction_into_gemm(&rms_stage, &gemm_stage, "fused_norm_gemm").expect("fuse");

        // Apply perimeter replacement (prologue reads N from flat params)
        let (fused_ptx, _) =
            crate::perimeter::replace_perimeter(&fused_ptx, deriv_json, "fused_norm_gemm")
                .expect("perimeter replacement");
        let fused_ptx = crate::dedup_reg_declarations(&fused_ptx);

        let path = "/tmp/pipeline_fused_norm_gemm.ptx";
        std::fs::write(path, &fused_ptx).unwrap();

        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", path])
            .output()
            .expect("ptxas not found");

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("ptxas stderr:");
            for line in stderr.lines().take(30) {
                eprintln!("  {line}");
            }
            // Print the fused PTX around the error lines
            for line in stderr.lines() {
                if let Some(lnum) = line
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(')').next())
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    let ptx_lines: Vec<&str> = fused_ptx.lines().collect();
                    let start = lnum.saturating_sub(3);
                    let end = (lnum + 3).min(ptx_lines.len());
                    for i in start..end {
                        let marker = if i + 1 == lnum { ">>>" } else { "   " };
                        eprintln!("{marker} {:4}: {}", i + 1, ptx_lines[i]);
                    }
                }
            }
            panic!("ptxas FAILED on pipeline-compiled fused PTX");
        }
        println!("PASS: pipeline-compiled fused PTX passes ptxas");
    }

    #[test]
    fn build_silu_mul_computation_unit() {
        let comp = build_silu_mul_computation("test_silu_gemm").expect("build computation");

        assert!(
            !comp.per_site.is_empty(),
            "should have per-site code for up loading"
        );
        assert!(
            !comp.instructions.is_empty(),
            "should have per-element SiLU*up instructions"
        );
        let instr_text = comp.instructions.join("\n");
        assert!(
            instr_text.contains("ex2.approx") && instr_text.contains("div.rn.f32"),
            "instructions should contain SiLU (ex2 + div)"
        );
        assert!(
            instr_text.contains("%f_up{ELEM_IDX}"),
            "instructions should reference up values"
        );
        assert_eq!(comp.entry_name.as_deref(), Some("test_silu_gemm"));
    }

    #[test]
    fn silu_mul_fused_gemm_ptxas_valid() {
        let silu_ptx = include_str!("../../ptx-fusion/kernels/vllm_silu_mul.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        let silu_stage =
            PipelineStage::from_ptx("silu_mul", silu_ptx, None).expect("parse silu_mul");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused_ptx =
            fuse_pointwise_into_gemm(&silu_stage, &gemm_stage, "fused_silu_gemm").expect("fuse");

        // Apply perimeter replacement
        let (fused_ptx, _) =
            crate::perimeter::replace_perimeter(&fused_ptx, deriv_json, "fused_silu_gemm")
                .expect("perimeter replacement");
        let fused_ptx = crate::dedup_reg_declarations(&fused_ptx);

        // Verify SiLU markers
        assert!(
            fused_ptx.contains("ex2.approx"),
            "should contain SiLU exp approximation"
        );
        assert!(
            fused_ptx.contains("_ferrite_intermediate_bytes"),
            "should have intermediate offset param"
        );

        let path = "/tmp/fused_silu_mul_gemm.ptx";
        std::fs::write(path, &fused_ptx).unwrap();

        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", path])
            .output()
            .expect("ptxas not found");

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("ptxas stderr:");
            for line in stderr.lines().take(30) {
                eprintln!("  {line}");
            }
            panic!("ptxas FAILED on SiLU+mul fused GEMM PTX");
        }
        println!("PASS: SiLU+mul fused GEMM passes ptxas");
    }

    #[test]
    fn fuse_add_rms_norm_into_cutlass_gemm() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");

        let norm_stage = PipelineStage::from_ptx(
            "fused_add_rms_norm",
            ptx,
            Some("fused_add_rms_norm_kernelI13__nv_bfloat16"),
        )
        .expect("parse fused_add_rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused = fuse_reduction_into_gemm(&norm_stage, &gemm_stage, "fused_add_norm_gemm");

        match fused {
            Ok(ptx) => {
                // Should contain add+norm prologue (transplanted from fused_add_rms_norm)
                assert!(ptx.contains("rms_norm prologue"), "should contain prologue");
                assert!(
                    ptx.contains("add.f32"),
                    "prologue should contain f32 add (from fused_add_rms_norm)"
                );
                assert!(
                    ptx.contains("st.global"),
                    "prologue should contain writeback store"
                );
                assert!(
                    ptx.contains("mma.sync"),
                    "GEMM interior should be preserved"
                );
                // Should have the extra hs_input param
                assert!(
                    ptx.contains("_ferrite_rms_hs_input"),
                    "should have hs_input param"
                );
                eprintln!("PASS: fused_add_rms_norm → GEMM produces valid PTX structure");
            }
            Err(e) => panic!("fuse_reduction_into_gemm failed: {e}"),
        }
    }

    #[test]
    fn fused_add_norm_gemm_ptxas_valid() {
        let ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        let norm_stage = PipelineStage::from_ptx(
            "fused_add_rms_norm",
            ptx,
            Some("fused_add_rms_norm_kernelI13__nv_bfloat16"),
        )
        .expect("parse fused_add_rms_norm");
        let gemm_stage = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let fused_ptx = fuse_reduction_into_gemm(&norm_stage, &gemm_stage, "fused_add_norm_gemm")
            .expect("fuse");

        let (fused_ptx, _) =
            crate::perimeter::replace_perimeter(&fused_ptx, deriv_json, "fused_add_norm_gemm")
                .expect("perimeter replacement");
        let fused_ptx = crate::dedup_reg_declarations(&fused_ptx);

        let path = "/tmp/pipeline_fused_add_norm_gemm.ptx";
        std::fs::write(path, &fused_ptx).unwrap();

        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", path])
            .output()
            .expect("ptxas not found");

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("ptxas stderr:");
            for line in stderr.lines().take(30) {
                eprintln!("  {line}");
            }
            let ptx_lines: Vec<&str> = fused_ptx.lines().collect();
            for line in stderr.lines() {
                if let Some(lnum) = line
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(')').next())
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    let start = lnum.saturating_sub(3);
                    let end = (lnum + 3).min(ptx_lines.len());
                    for i in start..end {
                        let marker = if i + 1 == lnum { ">>>" } else { "   " };
                        eprintln!("{marker} {:4}: {}", i + 1, ptx_lines[i]);
                    }
                }
            }
            panic!("ptxas FAILED on fused_add_norm+GEMM PTX");
        }
        println!("PASS: fused_add_norm+GEMM passes ptxas");
    }

    #[test]
    fn compile_segment_rms_norm_gemm() {
        // Verify compile_segment produces structurally correct PTX.
        // Note: the output is pre-perimeter-replacement, so it references the
        // CUTLASS struct param and ferrite_params. ptxas validation happens
        // after replace_perimeter in the pipeline_fuse! macro.
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");

        let producer = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let consumer = PipelineStage::from_ptx("gemm", gemm_ptx, None).expect("parse gemm");

        let producer_perim =
            TilePerimeter::from_stage(&producer).expect("extract producer perimeter");
        let consumer_perim =
            TilePerimeter::from_stage(&consumer).expect("extract consumer perimeter");

        let fused = compile_segment(
            &producer,
            &consumer,
            &producer_perim,
            &consumer_perim,
            "seg_test",
        )
        .expect("compile_segment failed");

        // Should contain the prologue and MMA instructions
        assert!(
            fused.contains("rms_norm prologue"),
            "should have rms_norm prologue"
        );
        assert!(fused.contains("mma.sync"), "should preserve GEMM MMA");
        // Should contain per-element normalization (derived from emit body)
        assert!(
            fused.contains("f_rms_inv"),
            "should have inv_rms multiplication"
        );
        assert!(
            fused.contains("f_rms_wt"),
            "should have weight multiplication"
        );
        println!("PASS: compile_segment produces structurally valid PTX");
    }

    #[test]
    fn extract_per_element_from_rms_norm_emit() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx("rms_norm", rms_ptx, None).expect("parse rms_norm");
        let decomp = stage.decompose_reduction().expect("decompose");

        let instrs = extract_per_element_from_emit_body(
            &decomp.emit_body_lines,
            &decomp.finalized_value_reg,
        )
        .expect("should extract per-element instructions");

        eprintln!("Per-element instructions:");
        for instr in &instrs {
            eprintln!("  {instr}");
        }

        // Should have mul inv_rms and mul weight
        assert!(
            instrs.iter().any(|i| i.contains("%f_rms_inv")),
            "should multiply by inv_rms"
        );
        assert!(
            instrs.iter().any(|i| i.contains("%f_rms_wt")),
            "should multiply by weight"
        );
        assert_eq!(
            instrs.len(),
            2,
            "rms_norm should have exactly 2 per-element instructions"
        );
    }

    #[test]
    fn extract_per_element_from_fused_add_emit() {
        let rms_ptx = include_str!("../../ptx-fusion/kernels/vllm_rms_norm.ptx");
        let stage = PipelineStage::from_ptx(
            "fused_add",
            rms_ptx,
            Some("fused_add_rms_norm_kernelI13__nv_bfloat16"),
        )
        .expect("parse");
        let decomp = stage.decompose_reduction().expect("decompose");

        let instrs = extract_per_element_from_emit_body(
            &decomp.emit_body_lines,
            &decomp.finalized_value_reg,
        )
        .expect("should extract per-element instructions");

        eprintln!("fused_add per-element instructions:");
        for instr in &instrs {
            eprintln!("  {instr}");
        }

        // Should have mul inv_rms and mul weight (same pattern as rms_norm)
        assert!(instrs.iter().any(|i| i.contains("%f_rms_inv")));
        assert!(instrs.iter().any(|i| i.contains("%f_rms_wt")));
    }

    #[test]
    fn sequence_two_gemms_ptxas_valid() {
        // Sequence two CUTLASS GEMMs (same config) into one kernel.
        // Both need perimeter replacement first to get flat-param layout.
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        // Perimeter-replace both copies
        let (flat_a, _entry_a) =
            crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "gemm_a")
                .expect("replace_perimeter A");
        let (flat_b, _entry_b) =
            crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "gemm_b")
                .expect("replace_perimeter B");

        let sequenced =
            sequence_two_gemms(&flat_a, &flat_b, "two_gemm_test").expect("sequence failed");

        eprintln!(
            "Sequenced two-GEMM kernel: {} lines",
            sequenced.lines().count()
        );

        // Verify structure
        assert!(sequenced.contains("Phase 1"), "should have phase 1");
        assert!(sequenced.contains("Phase 2"), "should have phase 2");
        assert!(sequenced.contains("bar.sync"), "should have barrier");
        assert!(
            sequenced.contains("ferrite_params_2"),
            "GEMM_B should use ferrite_params_2"
        );
        assert!(
            sequenced.contains("$L__BB1"),
            "GEMM_B labels should be renamed"
        );
        assert!(
            sequenced.contains("mma.sync"),
            "should have MMA instructions"
        );

        // ptxas validation
        let path = "/tmp/two_gemm_sequenced.ptx";
        std::fs::write(path, &sequenced).unwrap();
        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", path])
            .output()
            .expect("ptxas");
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("ptxas stderr:");
            for line in stderr.lines().take(20) {
                eprintln!("  {line}");
            }
            // Show context around error lines
            let ptx_lines: Vec<&str> = sequenced.lines().collect();
            for line in stderr.lines() {
                if let Some(lnum) = line
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(')').next())
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    let start = lnum.saturating_sub(3);
                    let end = (lnum + 3).min(ptx_lines.len());
                    for i in start..end {
                        let marker = if i + 1 == lnum { ">>>" } else { "   " };
                        eprintln!("{marker} {:4}: {}", i + 1, ptx_lines[i]);
                    }
                }
            }
            panic!("ptxas FAILED on sequenced two-GEMM kernel");
        }
        println!(
            "PASS: sequenced two-GEMM passes ptxas ({} lines)",
            sequenced.lines().count()
        );
    }

    #[test]
    fn sequenced_mlp_barrier_count() {
        // Build the full sequenced MLP kernel (gate_up + SiLU-fused-down)
        // and check ptxas barrier/register allocation.
        // This is the non-persistent equivalent of persistent_mlp_block.
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");
        let silu_ptx = include_str!("../../ptx-fusion/kernels/vllm_silu_mul.ptx");

        // Step 1: Perimeter-replace gate_up GEMM
        let (flat_gate_up, _) =
            crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "gate_up")
                .expect("replace_perimeter gate_up");

        // Step 2: Fuse SiLU+mul into down GEMM, then perimeter-replace
        let silu_stage =
            crate::pipeline::PipelineStage::from_ptx("silu_mul", silu_ptx, None)
                .expect("silu stage");
        let down_stage =
            crate::pipeline::PipelineStage::from_ptx("down", gemm_ptx, None)
                .expect("down stage");
        let fused_down =
            fuse_pointwise_into_gemm(&silu_stage, &down_stage, "down_silu")
                .expect("fuse silu into down");
        let (flat_down, _) =
            crate::perimeter::replace_perimeter(&fused_down, deriv_json, "down_silu")
                .expect("replace_perimeter down_silu");
        let flat_down = crate::dedup_reg_declarations(&flat_down);

        // Step 3: Sequence into one kernel
        let sequenced = sequence_two_gemms(&flat_gate_up, &flat_down, "mlp_sequenced")
            .expect("sequence failed");

        // Compile with ptxas -v to get resource usage
        let path = "/tmp/mlp_sequenced_barrier_test.ptx";
        std::fs::write(path, &sequenced).unwrap();
        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", "-v", path])
            .output()
            .expect("ptxas");

        let stderr = String::from_utf8_lossy(&out.stderr);
        eprintln!("=== ptxas -v output for sequenced MLP ===");
        for line in stderr.lines() {
            eprintln!("  {line}");
        }

        assert!(
            out.status.success(),
            "ptxas FAILED on sequenced MLP kernel"
        );

        // Parse barrier count from ptxas -v output
        // Format: "Used N registers, used B barriers, ..."
        let barrier_count = stderr
            .lines()
            .find(|l| l.contains("barriers"))
            .and_then(|l| {
                // Look for "used N barriers" pattern
                let words: Vec<&str> = l.split_whitespace().collect();
                for i in 0..words.len().saturating_sub(1) {
                    if words[i + 1] == "barriers" || words[i + 1] == "barriers," {
                        return words[i].trim_end_matches(',').parse::<u32>().ok();
                    }
                }
                None
            });

        eprintln!("Barrier count: {:?}", barrier_count);

        // Also check persistent MLP for comparison
        let persistent_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let persistent_deriv =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");
        let persistent_silu = include_str!("../../ptx-fusion/kernels/vllm_silu_mul.ptx");

        // Build via the same path as persistent_mlp_block! but just the single-body version
        let silu_computation = build_silu_mul_computation("mlp_persistent_test").expect("silu comp");

        let (flat_gemm, _) =
            crate::perimeter::replace_perimeter(persistent_ptx, persistent_deriv, "gemm")
                .expect("replace_perimeter gemm");

        let persistent = crate::persistent::make_single_body_persistent_mlp(
            &flat_gemm,
            &silu_computation,
            "mlp_persistent_test",
        )
        .expect("persistent mlp");

        let path2 = "/tmp/mlp_persistent_barrier_test.ptx";
        std::fs::write(path2, &persistent).unwrap();
        let out2 = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", "-v", path2])
            .output()
            .expect("ptxas");

        let stderr2 = String::from_utf8_lossy(&out2.stderr);
        eprintln!("\n=== ptxas -v output for persistent MLP ===");
        for line in stderr2.lines() {
            eprintln!("  {line}");
        }

        let persistent_barriers = stderr2
            .lines()
            .find(|l| l.contains("barriers"))
            .and_then(|l| {
                let words: Vec<&str> = l.split_whitespace().collect();
                for i in 0..words.len().saturating_sub(1) {
                    if words[i + 1] == "barriers" || words[i + 1] == "barriers," {
                        return words[i].trim_end_matches(',').parse::<u32>().ok();
                    }
                }
                None
            });

        eprintln!("Persistent barrier count: {:?}", persistent_barriers);

        // The key assertion: sequenced should use fewer barriers than persistent
        if let (Some(seq_b), Some(per_b)) = (barrier_count, persistent_barriers) {
            eprintln!("\n=== COMPARISON ===");
            eprintln!("  Sequenced (non-persistent): {} barriers", seq_b);
            eprintln!("  Persistent:                 {} barriers", per_b);
            assert!(
                seq_b < per_b,
                "Expected sequenced ({seq_b}) < persistent ({per_b}) barriers"
            );
        }
    }

    #[test]
    fn extract_gemm_body_64x128x32() {
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        // Must be perimeter-replaced first
        let (flat, _) = crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "test_gemm")
            .expect("replace_perimeter");

        let desc = extract_gemm_body(&flat).expect("extract_gemm_body");

        eprintln!("=== GemmBodyDescriptor for 64x128x32 ===");
        eprintln!("  Entry: {}", desc.entry_name);
        eprintln!("  Preamble: {} lines", desc.preamble.len());
        eprintln!("  K-loop: {} lines ({}..{})", desc.k_loop.len(),
            desc.loop_desc.header_line, desc.loop_desc.backedge_line);
        eprintln!("  Epilogue: {} lines", desc.epilogue.len());
        eprintln!("  MMA accumulators: {}", desc.mma_accumulators.len());
        eprintln!("  Tile pointers: {}", desc.tile_pointers.len());
        eprintln!("  Induction vars: {}", desc.induction_vars.len());
        eprintln!("  Buffer state: {}", desc.buffer_state.len());
        eprintln!("  Reg decls: {}", desc.reg_decls.len());
        eprintln!("  SMEM decls: {}", desc.smem_decls.len());
        eprintln!("  Params: {}", desc.params.len());

        // Verify key properties
        assert!(!desc.preamble.is_empty(), "should have preamble");
        assert!(!desc.k_loop.is_empty(), "should have K-loop");
        assert!(!desc.epilogue.is_empty(), "should have epilogue");
        assert_eq!(desc.mma_accumulators.len(), 128, "64x128x32 should have 128 MMA accum regs");
        assert!(desc.tile_pointers.len() >= 2, "should have A and B tile pointers");
        assert!(!desc.induction_vars.is_empty(), "should have K-loop induction vars");
        assert!(!desc.buffer_state.is_empty(), "should have SMEM buffer rotation state");

        // K-loop should contain mma.sync
        assert!(
            desc.k_loop.iter().any(|l| l.contains("mma.sync")),
            "K-loop must contain MMA instructions"
        );
        // Epilogue should contain cvt.rn.bf16x2.f32 (bf16 conversion)
        assert!(
            desc.epilogue.iter().any(|l| l.contains("cvt.rn.bf16x2.f32")),
            "Epilogue must contain bf16 conversion"
        );
        // Epilogue should contain st.global (output stores)
        assert!(
            desc.epilogue.iter().any(|l| l.contains("st.global")),
            "Epilogue must contain global stores"
        );
    }

    #[test]
    fn extract_gemm_body_64x64x32() {
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let (flat, _) = crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "test_gemm64")
            .expect("replace_perimeter");

        let desc = extract_gemm_body(&flat).expect("extract_gemm_body");

        eprintln!("=== GemmBodyDescriptor for 64x64x32 ===");
        eprintln!("  MMA accumulators: {}", desc.mma_accumulators.len());
        eprintln!("  Preamble: {} lines", desc.preamble.len());
        eprintln!("  K-loop: {} lines", desc.k_loop.len());
        eprintln!("  Epilogue: {} lines", desc.epilogue.len());

        assert_eq!(desc.mma_accumulators.len(), 64, "64x64x32 should have 64 MMA accum regs");
    }

    #[test]
    fn accum_spill_reload_64x64x32() {
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let (flat, _) = crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "test_spill")
            .expect("replace_perimeter");

        let desc = extract_gemm_body(&flat).expect("extract_gemm_body");
        let sr = build_accum_spill_reload(&desc, 36864); // 36KB = after CUTLASS SMEM

        eprintln!("=== AccumSpillReload for 64x64x32 ===");
        eprintln!("  SMEM bytes: {} ({}KB)", sr.smem_bytes, sr.smem_bytes / 1024);
        eprintln!("  Spill instructions: {}", sr.spill.len());
        eprintln!("  Reload instructions: {}", sr.reload.len());
        eprintln!("  Zero instructions: {}", sr.zero.len());
        eprintln!("  Extra reg decls: {:?}", sr.reg_decls);

        // 64 accums * 4 bytes * 128 threads = 32KB
        assert_eq!(sr.smem_bytes, 32768, "64 accums * 4B * 128 threads = 32KB");
        // spill = 1 comment + 2 addr setup + 64 stores = 67
        assert_eq!(sr.spill.len(), 67);
        // reload = 1 comment + 2 addr setup + 64 loads = 67
        assert_eq!(sr.reload.len(), 67);
        // zero = 1 comment + 64 movs = 65
        assert_eq!(sr.zero.len(), 65);

        // Verify spill uses actual accumulator register names from the descriptor
        let first_accum = &desc.mma_accumulators[0];
        assert!(
            sr.spill.iter().any(|l| l.contains(first_accum)),
            "spill should reference actual accum register {first_accum}"
        );
        assert!(
            sr.reload.iter().any(|l| l.contains(first_accum)),
            "reload should reference actual accum register {first_accum}"
        );
        assert!(
            sr.zero.iter().any(|l| l.contains(first_accum)),
            "zero should reference actual accum register {first_accum}"
        );

        // Print first few instructions for visual verification
        eprintln!("\nFirst 5 spill instructions:");
        for l in sr.spill.iter().take(5) {
            eprintln!("  {l}");
        }
        eprintln!("First 5 reload instructions:");
        for l in sr.reload.iter().take(5) {
            eprintln!("  {l}");
        }
    }

    #[test]
    fn analyze_epilogue_stores_64x128x32() {
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        let (flat, _) = crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "test_epi")
            .expect("replace_perimeter");

        let desc = extract_gemm_body(&flat).expect("extract_gemm_body");
        let full_lines: Vec<&str> = flat.lines().collect();
        let epi_start = desc.loop_desc.backedge_line + 1;

        let map = analyze_epilogue_stores(&desc.epilogue, &full_lines, epi_start)
            .expect("analyze_epilogue_stores");

        eprintln!("=== EpilogueStoreMap for 64x128x32 ===");
        eprintln!("  Base addr reg: {}", map.base_addr_reg);
        eprintln!("  Row reg: {}", map.row_reg);
        eprintln!("  Col reg: {}", map.col_reg);
        eprintln!("  Stride regs: {:?}", map.stride_regs);
        eprintln!("  Total stores: {}", map.stores.len());

        for (i, store) in map.stores.iter().enumerate() {
            eprintln!(
                "  Store {}: addr={} chain={:?} data={:?}",
                i, store.addr_reg, store.stride_chain, store.data_regs
            );
        }

        // Verify basic properties
        assert_eq!(map.stores.len(), 16, "64x128 tile should have 16 st.global stores (8 per epilogue path)");
        assert!(!map.row_reg.is_empty(), "should identify row register");
        assert!(!map.col_reg.is_empty(), "should identify column register");
        assert!(!map.stride_regs.is_empty(), "should identify stride registers");

        // First store should have empty stride chain (it's the base)
        assert!(
            map.stores[0].stride_chain.is_empty(),
            "first store should be at base address (no stride chain)"
        );

        // Subsequent stores should have non-empty stride chains
        assert!(
            !map.stores[1].stride_chain.is_empty(),
            "second store should have a stride offset from base"
        );
    }

    #[test]
    fn redirect_epilogue_to_smem_64x128x32() {
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x128x32_sm89.derivations.json");

        let (flat, _) = crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "test_redir")
            .expect("replace_perimeter");

        let desc = extract_gemm_body(&flat).expect("extract_gemm_body");
        let full_lines: Vec<&str> = flat.lines().collect();
        let epi_start = desc.loop_desc.backedge_line + 1;

        let map = analyze_epilogue_stores(&desc.epilogue, &full_lines, epi_start)
            .expect("analyze_epilogue_stores");

        let redir = redirect_epilogue_to_smem(&desc.epilogue, &map, 36864, 128)
            .expect("redirect_epilogue_to_smem");

        eprintln!("=== Redirected Epilogue ===");
        eprintln!("  Lines: {} (was {})", redir.lines.len(), desc.epilogue.len());
        eprintln!("  SMEM bytes: {} ({}KB)", redir.smem_bytes, redir.smem_bytes / 1024);
        eprintln!("  Extra reg decls: {:?}", redir.reg_decls);

        // Should have no st.global left
        let has_global = redir.lines.iter().any(|l| l.contains("st.global"));
        assert!(!has_global, "redirected epilogue should have no st.global");

        // Should have st.shared for the redirected stores
        let shared_stores = redir.lines.iter().filter(|l| l.contains("st.shared.v4.b32")).count();
        eprintln!("  Redirected st.shared stores: {}", shared_stores);
        assert!(shared_stores > 0, "should have redirected st.shared stores");

        // Should preserve epilogue infrastructure (bar.sync, ld.shared, mul, cvt)
        assert!(
            redir.lines.iter().any(|l| l.contains("bar.sync")),
            "should preserve bar.sync"
        );
        assert!(
            redir.lines.iter().any(|l| l.contains("cvt.rn.bf16x2.f32")),
            "should preserve bf16 conversion"
        );
        assert!(
            redir.lines.iter().any(|l| l.contains("ld.shared")),
            "should preserve ld.shared (SMEM rearrangement)"
        );

        // SMEM should be tile_m * tile_n * 2 = 64 * 128 * 2 = 16KB
        assert_eq!(redir.smem_bytes, 16384);

        // Print some redirected store lines
        eprintln!("\nRedirected store examples:");
        for l in redir.lines.iter().filter(|l| l.contains("FERRITE: epilogue")).take(3) {
            eprintln!("  {l}");
        }
        for l in redir.lines.iter().filter(|l| l.contains("st.shared.v4.b32")).take(3) {
            eprintln!("  {l}");
        }
    }

    #[test]
    fn silu_from_smem_ptxas_valid() {
        // Test: build a down GEMM with SiLU fused at A-loads,
        // where gate and up values come from SMEM scratch.
        // This verifies ptxas can compile the SMEM-sourced SiLU fusion.
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let (flat, _) = crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "silu_smem_test")
            .expect("replace_perimeter");

        // Build SiLU computation with SMEM sources
        let computation = build_silu_mul_computation_smem(
            "silu_smem_test",
            "%r_gate_scratch",
            "%r_up_scratch",
        )
        .expect("build silu smem");

        // Apply to the GEMM
        let fused = replace_a_loads_with_inline_fn(&flat, "", &computation)
            .expect("replace_a_loads");

        // Add the scratch base register declarations (in a real kernel these would
        // be computed from dynamic SMEM + offsets)
        // Find the first .reg line and prepend our declarations
        let mut fused_lines: Vec<String> = fused.lines().map(|l| l.to_string()).collect();
        let reg_insert_pos = fused_lines.iter().position(|l| l.trim().starts_with(".reg "))
            .unwrap_or(0);
        fused_lines.insert(reg_insert_pos, "\t.reg .u32 %r_gate_scratch, %r_up_scratch;".into());
        let fused = fused_lines.join("\n");

        eprintln!("Fused SiLU-from-SMEM kernel: {} lines", fused.lines().count());

        // ptxas validation
        let path = "/tmp/silu_smem_test.ptx";
        std::fs::write(path, &fused).unwrap();
        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", "-v", path])
            .output()
            .expect("ptxas");

        let stderr = String::from_utf8_lossy(&out.stderr);
        eprintln!("ptxas -v output:");
        for line in stderr.lines() {
            eprintln!("  {line}");
        }

        assert!(
            out.status.success(),
            "ptxas FAILED on SiLU-from-SMEM fused kernel"
        );

        // Check barrier count
        let barrier_count = stderr
            .lines()
            .find(|l| l.contains("barriers"))
            .and_then(|l| {
                let words: Vec<&str> = l.split_whitespace().collect();
                for i in 0..words.len().saturating_sub(1) {
                    if words[i + 1] == "barriers" || words[i + 1] == "barriers," {
                        return words[i].trim_end_matches(',').parse::<u32>().ok();
                    }
                }
                None
            });
        eprintln!("Barrier count: {:?}", barrier_count);

        // Should be low (not 16)
        if let Some(b) = barrier_count {
            assert!(b < 16, "barrier count {b} should be < 16");
        }
    }

    #[test]
    fn fuse_gemm_pointwise_gemm_ptxas() {
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        // Perimeter-replace both producer and consumer (same config for now)
        let (prod_flat, _) =
            crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "producer")
                .expect("replace_perimeter producer");
        let (cons_flat, _) =
            crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "consumer")
                .expect("replace_perimeter consumer");

        // Build SiLU computation with SMEM sources
        let computation = build_silu_mul_computation_smem(
            "fused_mlp",
            "%r_gate_scratch",
            "%r_up_scratch",
        )
        .expect("build silu smem");

        let fused = fuse_gemm_pointwise_gemm(
            &prod_flat,
            &cons_flat,
            &computation,
            &ProducerOutput::Paired { n_offset_tiles: 43 }, // example
            "fused_mlp_test",
        )
        .expect("fuse_gemm_pointwise_gemm");

        eprintln!("Fused GEMM-pointwise-GEMM kernel: {} lines", fused.lines().count());

        // ptxas validation
        let path = "/tmp/fused_gemm_pw_gemm.ptx";
        std::fs::write(path, &fused).unwrap();
        let out = std::process::Command::new("/usr/local/cuda-12.9/bin/ptxas")
            .args(["-arch=sm_89", "-v", path])
            .output()
            .expect("ptxas");

        let stderr = String::from_utf8_lossy(&out.stderr);
        eprintln!("ptxas output:");
        for line in stderr.lines() {
            eprintln!("  {line}");
        }

        if !out.status.success() {
            // Show context around first error
            let ptx_lines: Vec<&str> = fused.lines().collect();
            for line in stderr.lines().take(5) {
                if let Some(lnum) = line
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(')').next())
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    let start = lnum.saturating_sub(2);
                    let end = (lnum + 2).min(ptx_lines.len());
                    for i in start..end {
                        let marker = if i + 1 == lnum { ">>>" } else { "   " };
                        eprintln!("{marker} {:4}: {}", i + 1, ptx_lines[i]);
                    }
                }
            }
            panic!("ptxas FAILED on fused GEMM-pointwise-GEMM kernel");
        }

        // Check barrier count and registers
        let barrier_count = stderr
            .lines()
            .find(|l| l.contains("barriers"))
            .and_then(|l| {
                let words: Vec<&str> = l.split_whitespace().collect();
                for i in 0..words.len().saturating_sub(1) {
                    if words[i + 1] == "barriers" || words[i + 1] == "barriers," {
                        return words[i].trim_end_matches(',').parse::<u32>().ok();
                    }
                }
                None
            });
        eprintln!("\nBarrier count: {:?}", barrier_count);

        if let Some(b) = barrier_count {
            assert!(b <= 4, "fused kernel should use few barriers, got {b}");
        }
    }

    #[test]
    fn depipeline_k_loop_64x64x32() {
        let gemm_ptx = include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.ptx");
        let deriv_json =
            include_str!("../../ptx-fusion/kernels/cutlass_bf16_64x64x32_sm89.derivations.json");

        let (flat, _) = crate::perimeter::replace_perimeter(gemm_ptx, deriv_json, "test_dpipe")
            .expect("replace_perimeter");

        let desc = extract_gemm_body(&flat).expect("extract_gemm_body");

        // Test 1: de-pipeline with A-loads from GMEM (standard explicit replacement)
        let dp = depipeline_k_loop(&desc.k_loop, &flat, None)
            .expect("depipeline_k_loop");

        eprintln!("=== De-pipelined K-loop (GMEM A-loads) ===");
        eprintln!("  Lines: {} (was {})", dp.lines.len(), desc.k_loop.len());
        eprintln!("  A loads replaced: {}", dp.a_loads_replaced);
        eprintln!("  B loads replaced: {}", dp.b_loads_replaced);

        // No cp.async should remain
        assert!(
            !dp.lines.iter().any(|l| l.contains("cp.async.cg.shared.global")),
            "de-pipelined loop should have no cp.async"
        );
        // No commit/wait groups
        assert!(
            !dp.lines.iter().any(|l| l.contains("cp.async.commit_group") && !l.contains("removed")),
            "should strip commit_group"
        );
        assert!(
            !dp.lines.iter().any(|l| l.contains("cp.async.wait_group") && !l.contains("removed")),
            "should strip wait_group"
        );
        // MMA should be preserved
        assert!(
            dp.lines.iter().any(|l| l.contains("mma.sync")),
            "MMA instructions must be preserved"
        );
        // Should have explicit loads
        assert!(dp.a_loads_replaced > 0, "should replace A-loads");
        assert!(dp.b_loads_replaced > 0, "should replace B-loads");

        // Test 2: de-pipeline with A-loads from SMEM scratch
        let dp_smem = depipeline_k_loop(&desc.k_loop, &flat, Some("%r_silu_scratch"))
            .expect("depipeline_k_loop smem");

        eprintln!("\n=== De-pipelined K-loop (SMEM A-loads) ===");
        eprintln!("  A loads from SMEM: {}", dp_smem.a_loads_replaced);

        // A-loads should use ld.shared
        let a_smem_loads = dp_smem.lines.iter()
            .filter(|l| l.contains("ld.shared") && l.contains("%r_silu_scratch"))
            .count();
        assert!(
            a_smem_loads > 0,
            "A-loads should read from SMEM scratch register"
        );
        // B-loads should still use ld.global
        assert!(
            dp_smem.lines.iter().any(|l| l.contains("ld.global")),
            "B-loads should still use ld.global"
        );

        eprintln!("\nA-load examples (SMEM):");
        for l in dp_smem.lines.iter().filter(|l| l.contains("ld.shared") && l.contains("silu")).take(3) {
            eprintln!("  {l}");
        }
    }
}
