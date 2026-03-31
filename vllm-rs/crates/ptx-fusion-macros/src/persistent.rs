//! Persistent kernel wrapper: wrap a fused kernel in a work-queue loop.
//!
//! Transforms fused PTX (from fuse_real) into a persistent kernel where
//! a GPU-filling grid loops, grabbing tiles from an atomic counter.
//! Each iteration runs the full fused pipeline (e.g., rms_norm -> GEMM).

/// Wrap a fused kernel PTX in a persistent work-queue loop.
///
/// The persistent kernel:
/// 1. Thread 0 of each block atomicAdds a tile counter
/// 2. Broadcasts the tile index to all threads via SMEM
/// 3. Runs the fused phases (with %ctaid.x replaced by the tile index)
/// 4. Loops until all tiles are processed
///
/// Adds a new `.param .u64 _persistent_counter` as the first parameter.
/// The caller must pass a device pointer to a u32 initialized to 0.
pub fn make_persistent(
    fused_ptx: &str,
    new_name: &str,
    total_rows_param: &str,
) -> Result<String, String> {
    let mut lines: Vec<String> = fused_ptx.lines().map(|l| l.to_string()).collect();

    // Find and rename the entry point, add counter param
    let entry_idx = lines
        .iter()
        .position(|l| l.contains(".visible") && l.contains(".entry"))
        .ok_or("no .entry found")?;

    // Find the old entry name for replacement
    let entry_line = &lines[entry_idx];
    let old_name = extract_entry_name(entry_line).ok_or("could not parse entry name")?;

    // Replace entry name
    lines[entry_idx] = lines[entry_idx].replace(&old_name, new_name);

    // Find the opening paren of params, insert counter param after it
    let first_param_idx = lines
        .iter()
        .position(|l| l.contains(".param") && l.contains("_param_"))
        .ok_or("no params found")?;

    lines.insert(
        first_param_idx,
        "\t.param .u64 _persistent_counter,".to_string(),
    );

    // Find the total_rows param name (substring match in the param list)
    let total_rows_full = lines
        .iter()
        .filter_map(|l| {
            let t = l.trim();
            if t.starts_with(".param") && t.contains(total_rows_param) {
                // Extract the param name
                let parts: Vec<&str> = t.split_whitespace().collect();
                parts.get(2).map(|s| s.trim_end_matches(',').to_string())
            } else {
                None
            }
        })
        .next()
        .ok_or_else(|| format!("no param matching '{total_rows_param}' found"))?;

    // Find where register declarations end (first non-.reg, non-.shared, non-empty line after '{')
    let body_start = lines
        .iter()
        .position(|l| {
            let t = l.trim();
            t == "{" || t.ends_with('{')
        })
        .ok_or("no '{' found")?;

    // Find the first instruction line after reg/shared declarations
    let mut insert_after_decls = body_start + 1;
    for (i, line) in lines.iter().enumerate().skip(body_start + 1) {
        let t = line.trim();
        if !t.starts_with(".reg")
            && !t.starts_with(".shared")
            && !t.starts_with(".local")
            && !t.starts_with("//")
            && !t.is_empty()
        {
            insert_after_decls = i;
            break;
        }
    }

    // Insert persistent scratch declarations before the first instruction
    let persistent_decls = [
        "\t// FERRITE: persistent kernel scratch".to_string(),
        "\t.reg .u32 \t%r_ptile;".to_string(),
        "\t.reg .u64 \t%rd_pctr;".to_string(),
        "\t.reg .u32 \t%r_ptotal;".to_string(),
        "\t.reg .pred \t%p_pdone;".to_string(),
        "\t.reg .pred \t%p_pt0;".to_string(),
        "\t.shared .align 4 .u32 _ptile_smem[1];".to_string(),
        String::new(),
    ];
    for (j, decl) in persistent_decls.iter().enumerate() {
        lines.insert(insert_after_decls + j, decl.clone());
    }
    let offset = persistent_decls.len();

    // The first instruction is now at insert_after_decls + offset
    // Insert the persistent loop preamble + tile grab before the first instruction
    let loop_preamble = vec![
        "\t// FERRITE: persistent loop setup".to_string(),
        "\tld.param.u64 \t%rd_pctr, [_persistent_counter];".to_string(),
        "\tcvta.to.global.u64 \t%rd_pctr, %rd_pctr;".to_string(),
        format!("\tld.param.u32 \t%r_ptotal, [{total_rows_full}];"),
        String::new(),
        "$L_persistent_loop:".to_string(),
        "\t// FERRITE: grab next tile (thread 0 atomicAdd, broadcast via SMEM)".to_string(),
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
    let preamble_insert = insert_after_decls + offset;
    for (j, line) in loop_preamble.iter().enumerate() {
        lines.insert(preamble_insert + j, line.clone());
    }

    // Replace all `mov.u32 %rN, %ctaid.x` with `mov.u32 %rN, %r_ptile`
    for line in &mut lines {
        if line.contains("%ctaid.x") && line.contains("mov.u32") {
            *line = line.replace("%ctaid.x", "%r_ptile");
        }
    }

    // Replace `ret;` with loop back + exit
    let ret_idx = lines
        .iter()
        .rposition(|l| l.trim() == "ret;")
        .ok_or("no ret; found")?;

    lines[ret_idx] = [
        "\tbar.sync \t15;",
        "\tbra \t$L_persistent_loop;",
        "",
        "$L_persistent_exit:",
        "\tret;",
    ]
    .join("\n");

    Ok(lines.join("\n"))
}

/// Wrap a flat-param CUTLASS GEMM in a persistent work-queue loop.
///
/// Unlike `make_persistent` (which only replaces %ctaid.x for 1D grids),
/// this handles the 2D swizzled grid by replacing BOTH %ctaid.x and %ctaid.y.
///
/// The linear tile index from the atomic counter is decomposed:
///   ctaid_x = tile_idx % grid_x
///   ctaid_y = tile_idx / grid_x
///
/// Added params (prepended):
///   - `_persistent_counter`: u64 ptr to atomic u32 tile counter (init to 0)
///   - `_persistent_grid_x`: u32 grid dim X (swizzled)
///   - `_persistent_total`: u32 total tiles (grid_x * grid_y)
pub fn make_persistent_gemm(ptx: &str, new_name: &str) -> Result<String, String> {
    let mut lines: Vec<String> = ptx.lines().map(|l| l.to_string()).collect();

    // Rename entry
    let entry_idx = lines
        .iter()
        .position(|l| l.contains(".visible") && l.contains(".entry"))
        .ok_or("no .entry found")?;
    let old_name = extract_entry_name(&lines[entry_idx]).ok_or("could not parse entry name")?;
    lines[entry_idx] = lines[entry_idx].replace(&old_name, new_name);

    // Insert persistent params before existing params
    let first_param_idx = lines
        .iter()
        .position(|l| l.trim().starts_with(".param"))
        .ok_or("no .param found")?;
    for p in [
        "\t.param .u32 _persistent_total,",
        "\t.param .u32 _persistent_grid_x,",
        "\t.param .u64 _persistent_counter,",
    ] {
        lines.insert(first_param_idx, p.to_string());
    }

    // Find body start
    let body_start = lines
        .iter()
        .position(|l| {
            let t = l.trim();
            t == "{" || t.ends_with('{')
        })
        .ok_or("no '{' found")?;

    // Find first instruction after declarations
    let mut insert_pos = body_start + 1;
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

    // Insert declarations
    let decls = [
        "\t// FERRITE: persistent GEMM scratch",
        "\t.reg .u32 \t%r_ptile, %r_ptile_x, %r_ptile_y;",
        "\t.reg .u32 \t%r_pgrid_x, %r_ptotal;",
        "\t.reg .u64 \t%rd_pctr;",
        "\t.reg .pred \t%p_pdone, %p_pt0;",
        "\t.shared .align 4 .u32 _ptile_smem[1];",
        "",
    ];
    for (j, d) in decls.iter().enumerate() {
        lines.insert(insert_pos + j, d.to_string());
    }
    insert_pos += decls.len();

    // Insert persistent loop
    let preamble = [
        "\t// FERRITE: persistent loop",
        "\tld.param.u64 \t%rd_pctr, [_persistent_counter];",
        "\tcvta.to.global.u64 \t%rd_pctr, %rd_pctr;",
        "\tld.param.u32 \t%r_ptotal, [_persistent_total];",
        "\tld.param.u32 \t%r_pgrid_x, [_persistent_grid_x];",
        "",
        "$L_persistent_loop:",
        "\t// Grab next tile (thread 0 atomicAdd, broadcast via SMEM)",
        "\tmov.u32 \t%r_ptile, %tid.x;",
        "\tsetp.eq.u32 \t%p_pt0, %r_ptile, 0;",
        "\t@%p_pt0 atom.global.add.u32 \t%r_ptile, [%rd_pctr], 1;",
        "\t@%p_pt0 st.shared.u32 \t[_ptile_smem], %r_ptile;",
        "\tbar.sync \t14;",
        "\tld.shared.u32 \t%r_ptile, [_ptile_smem];",
        "\tsetp.ge.u32 \t%p_pdone, %r_ptile, %r_ptotal;",
        "\t@%p_pdone bra \t$L_persistent_exit;",
        "",
        "\t// Decompose linear tile to (ctaid_x, ctaid_y)",
        "\trem.u32 \t%r_ptile_x, %r_ptile, %r_pgrid_x;",
        "\tdiv.u32 \t%r_ptile_y, %r_ptile, %r_pgrid_x;",
        "",
    ];
    for (j, line) in preamble.iter().enumerate() {
        lines.insert(insert_pos + j, line.to_string());
    }

    // Replace ctaid.x and ctaid.y reads
    for line in &mut lines {
        if line.contains("%ctaid.x") && line.contains("mov.u32") {
            *line = line.replace("%ctaid.x", "%r_ptile_x");
        }
        if line.contains("%ctaid.y") && line.contains("mov.u32") {
            *line = line.replace("%ctaid.y", "%r_ptile_y");
        }
    }

    // Replace ret; with loop back + exit
    let ret_idx = lines
        .iter()
        .rposition(|l| l.trim() == "ret;")
        .ok_or("no ret; found")?;

    lines[ret_idx] = [
        "\tbar.sync \t15;",
        "\tbra \t$L_persistent_loop;",
        "",
        "$L_persistent_exit:",
        "\tret;",
    ]
    .join("\n");

    Ok(lines.join("\n"))
}

/// Wrap a two-phase sequenced kernel in a persistent work-queue loop
/// with per-M-tile barriers between phases.
///
/// Modifies the sequenced PTX in-place (same strategy as make_persistent_gemm):
/// 1. Renames entry, adds persistent params
/// 2. Adds persistent declarations + loop preamble
/// 3. Replaces ctaid.x/y with dispatched values (phase-aware)
/// 4. Replaces global barrier with phase-0-completion + phase-1-dispatch
/// 5. Replaces ret; with loop-back
pub fn make_persistent_two_phase(sequenced_ptx: &str, new_name: &str) -> Result<String, String> {
    let mut lines: Vec<String> = sequenced_ptx.lines().map(|l| l.to_string()).collect();

    // Find phase boundaries from the comments inserted by sequence_gemm_phases
    let phase1_start = lines
        .iter()
        .position(|l| l.contains("====== Phase 1 ======"))
        .ok_or("no Phase 1 marker")?;
    let barrier_start = lines
        .iter()
        .position(|l| l.contains("====== Global barrier 1") || l.contains("====== Phase barrier"))
        .ok_or("no barrier marker")?;
    let phase2_start = lines
        .iter()
        .position(|l| l.contains("====== Phase 2 ======"))
        .ok_or("no Phase 2 marker")?;

    // Extract phase bodies (instructions only, no declarations)
    let phase0_body: Vec<&str> = lines[phase1_start + 1..barrier_start]
        .iter()
        .map(|s| s.as_str())
        .collect();
    // Extract phase 1 body with brace-depth tracking (inline asm uses { })
    let phase1_body: Vec<&str> = {
        let mut body = Vec::new();
        let mut depth = 0i32;
        for l in &lines[phase2_start + 1..] {
            let t = l.trim();
            for ch in t.chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            // The kernel's closing } brings depth to -1 (we started inside the body)
            if depth < 0 {
                break;
            }
            if t == "ret;" {
                break;
            }
            body.push(l.as_str());
        }
        body
    };

    // Extract header (version/target/extern shared)
    let header: Vec<&str> = lines
        .iter()
        .take_while(|l| !l.contains(".entry"))
        .map(|s| s.as_str())
        .collect();

    // Extract existing params (between .entry and {)
    let mut existing_params = Vec::new();
    let mut in_params = false;
    for l in &lines {
        if l.contains(".entry") {
            in_params = true;
            continue;
        }
        if !in_params {
            continue;
        }
        let t = l.trim();
        if t.starts_with(".param") {
            existing_params.push(t.trim_end_matches(',').to_string());
        }
        if t == ")" || t == "{" || t.ends_with('{') {
            break;
        }
    }

    // Extract declarations (.reg, .shared) from the body
    let body_start = lines
        .iter()
        .position(|l| l.trim() == "{" || l.trim().ends_with('{'))
        .ok_or("no {")?;
    let mut decl_lines = Vec::new();
    for l in &lines[body_start + 1..phase1_start] {
        let t = l.trim();
        if t.starts_with(".reg") || t.starts_with(".shared") || t.is_empty() || t.starts_with("//")
        {
            decl_lines.push(l.as_str());
        }
    }

    // Build the persistent kernel
    let mut out = String::new();

    // Header
    for l in &header {
        out.push_str(l);
        out.push('\n');
    }

    // Entry with persistent + existing params
    out.push_str(&format!(".visible .entry {new_name}(\n"));
    let persistent_params = [
        ".param .u64 _persistent_counter",
        ".param .u32 _persistent_total",
        ".param .u32 _persistent_grid_x_0",
        ".param .u32 _persistent_phase0_tiles",
        ".param .u32 _persistent_grid_x_1",
        ".param .u32 _persistent_ntiles_per_m_0",
        ".param .u64 _persistent_mtile_done",
    ];
    for p in &persistent_params {
        out.push_str(&format!("\t{p},\n"));
    }
    // Remove _phase_barriers from existing params (replaced by persistent barrier)
    for (i, p) in existing_params.iter().enumerate() {
        if p.contains("_phase_barriers") {
            continue;
        }
        let comma = if i + 1 < existing_params.len()
            && !existing_params[i + 1].contains("_phase_barriers")
        {
            ","
        } else if i + 1 == existing_params.len() {
            ""
        } else {
            ","
        };
        out.push_str(&format!("\t{p}{comma}\n"));
    }
    out.push_str(")\n{\n");

    // Declarations from original kernel
    for l in &decl_lines {
        out.push_str(l);
        out.push('\n');
    }

    // Persistent scratch
    out.push_str("\t// FERRITE: persistent two-phase scratch\n");
    out.push_str("\t.reg .u32 \t%r_ptile, %r_ptile_x, %r_ptile_y;\n");
    out.push_str("\t.reg .u32 \t%r_ptotal, %r_pgx0, %r_pgx1;\n");
    out.push_str("\t.reg .u32 \t%r_pp0tiles, %r_plocal, %r_pmtile;\n");
    out.push_str("\t.reg .u32 \t%r_pntpm0, %r_pbarval, %r_pswiz;\n");
    out.push_str("\t.reg .u64 \t%rd_pctr, %rd_pmtdone, %rd_pbar;\n");
    out.push_str("\t.reg .pred \t%p_pdone, %p_pt0, %p_pphase, %p_pswiz;\n");
    out.push_str("\t.shared .align 4 .u32 _ptile_smem[1];\n\n");

    // Persistent loop setup
    out.push_str("\t// FERRITE: persistent loop setup\n");
    out.push_str("\tld.param.u64 \t%rd_pctr, [_persistent_counter];\n");
    out.push_str("\tcvta.to.global.u64 \t%rd_pctr, %rd_pctr;\n");
    out.push_str("\tld.param.u32 \t%r_ptotal, [_persistent_total];\n");
    out.push_str("\tld.param.u32 \t%r_pgx0, [_persistent_grid_x_0];\n");
    out.push_str("\tld.param.u32 \t%r_pgx1, [_persistent_grid_x_1];\n");
    out.push_str("\tld.param.u32 \t%r_pp0tiles, [_persistent_phase0_tiles];\n");
    out.push_str("\tld.param.u32 \t%r_pntpm0, [_persistent_ntiles_per_m_0];\n");
    out.push_str("\tld.param.u64 \t%rd_pmtdone, [_persistent_mtile_done];\n");
    out.push_str("\tcvta.to.global.u64 \t%rd_pmtdone, %rd_pmtdone;\n\n");

    // Persistent loop
    out.push_str("$L_persistent_loop:\n");
    out.push_str("\t// Grab next tile\n");
    out.push_str("\tmov.u32 \t%r_ptile, %tid.x;\n");
    out.push_str("\tsetp.eq.u32 \t%p_pt0, %r_ptile, 0;\n");
    out.push_str("\t@%p_pt0 atom.global.add.u32 \t%r_ptile, [%rd_pctr], 1;\n");
    out.push_str("\t@%p_pt0 st.shared.u32 \t[_ptile_smem], %r_ptile;\n");
    out.push_str("\tbar.sync \t14;\n");
    out.push_str("\tld.shared.u32 \t%r_ptile, [_ptile_smem];\n");
    out.push_str("\tsetp.ge.u32 \t%p_pdone, %r_ptile, %r_ptotal;\n");
    out.push_str("\t@%p_pdone bra \t$L_persistent_exit;\n\n");

    // Phase dispatch
    out.push_str("\t// Phase dispatch\n");
    out.push_str("\tsetp.lt.u32 \t%p_pphase, %r_ptile, %r_pp0tiles;\n");
    out.push_str("\t@%p_pphase bra \t$L_dispatch_phase0;\n");
    out.push_str("\tbra \t$L_dispatch_phase1;\n\n");

    // Phase 0 dispatch: compute ctaid_x/y, run body, increment M-tile counter
    out.push_str("$L_dispatch_phase0:\n");
    out.push_str("\tmov.u32 \t%r_plocal, %r_ptile;\n");
    out.push_str("\trem.u32 \t%r_ptile_x, %r_plocal, %r_pgx0;\n");
    out.push_str("\tdiv.u32 \t%r_ptile_y, %r_plocal, %r_pgx0;\n");
    out.push_str("\tbra \t$L_phase0_body;\n\n");

    // Phase 1 dispatch: compute ctaid_x/y, wait for M-tile barrier
    out.push_str("$L_dispatch_phase1:\n");
    out.push_str("\tsub.u32 \t%r_plocal, %r_ptile, %r_pp0tiles;\n");
    out.push_str("\trem.u32 \t%r_ptile_x, %r_plocal, %r_pgx1;\n");
    out.push_str("\tdiv.u32 \t%r_ptile_y, %r_plocal, %r_pgx1;\n");
    // Compute m_tile from ctaid_x (need swizzle_log from phase 1's N param)
    // Read N from ferrite_params_2+68, compute swizzle, extract m_tile
    // Actually, we can compute m_tile more simply: m_tile = local_idx / (grid_x_1 * grid_y_1 / grid_m)
    // But grid_m isn't a param... Let me use the GEMM's own approach:
    // The GEMM body will compute m_tile from ctaid_x after the swizzle.
    // For the barrier, I need m_tile BEFORE the body runs.
    // Simplest: m_tile = r_plocal / n_tiles_phase1_per_mtile
    // where n_tiles_phase1_per_mtile = total_phase1_tiles / grid_m
    // But I don't have grid_m as a param.
    //
    // Alternative: compute swizzle_log from phase 1's N.
    // Phase 1's N is at ferrite_params_2 + 68.
    out.push_str("\t// Compute m_tile for barrier check\n");
    out.push_str("\t// Read swizzle_log from phase 1 params\n");
    out.push_str("\tld.param.s32 \t%r_pbarval, [ferrite_params_2+68];\n"); // N for phase 1
    out.push_str("\tadd.s32 \t%r_pbarval, %r_pbarval, 127;\n");
    out.push_str("\tshr.u32 \t%r_pbarval, %r_pbarval, 7;\n"); // ceil(N/128) = grid_n
    out.push_str("\tmov.u32 \t%r_pswiz, 0;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 2;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pswiz, 1;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 4;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pswiz, 2;\n");
    // m_tile = ctaid_x >> swizzle_log
    out.push_str("\tshr.u32 \t%r_pmtile, %r_ptile_x, %r_pswiz;\n\n");

    // Per-M-tile barrier: spin until mtile_done[m_tile] >= ntiles_per_m_0
    out.push_str("\t// Per-M-tile barrier: wait for phase 0 completion\n");
    out.push_str("\tshl.b32 \t%r_pbarval, %r_pmtile, 2;\n"); // m_tile * 4
    out.push_str("\tcvt.u64.u32 \t%rd_pbar, %r_pbarval;\n");
    out.push_str("\tadd.u64 \t%rd_pbar, %rd_pmtdone, %rd_pbar;\n");
    out.push_str("$L_mtile_wait:\n");
    out.push_str("\tld.global.acquire.gpu.u32 \t%r_pbarval, [%rd_pbar];\n");
    out.push_str("\tsetp.ge.u32 \t%p_pdone, %r_pbarval, %r_pntpm0;\n");
    out.push_str("\t@!%p_pdone bra \t$L_mtile_wait;\n");
    out.push_str("\tbra \t$L_phase1_body;\n\n");

    // Phase 0 body
    out.push_str("$L_phase0_body:\n");
    // Replace ctaid.x/y in phase 0 body
    for l in &phase0_body {
        let mut line = l.to_string();
        if line.contains("%ctaid.x") && line.contains("mov.u32") {
            line = line.replace("%ctaid.x", "%r_ptile_x");
        }
        if line.contains("%ctaid.y") && line.contains("mov.u32") {
            line = line.replace("%ctaid.y", "%r_ptile_y");
        }
        out.push_str(&line);
        out.push('\n');
    }

    // After phase 0: increment M-tile completion counter
    out.push_str("\n\t// Phase 0 done: increment M-tile counter\n");
    out.push_str("\tbar.sync \t0;\n"); // ensure all threads finished
    out.push_str("\tmov.u32 \t%r_pbarval, %tid.x;\n");
    out.push_str("\tsetp.eq.u32 \t%p_pt0, %r_pbarval, 0;\n");
    // Compute m_tile from phase 0's ctaid_x (same swizzle approach)
    out.push_str("\tld.param.s32 \t%r_pbarval, [ferrite_params+68];\n"); // N for phase 0
    out.push_str("\tadd.s32 \t%r_pbarval, %r_pbarval, 127;\n");
    out.push_str("\tshr.u32 \t%r_pbarval, %r_pbarval, 7;\n");
    out.push_str("\tmov.u32 \t%r_pmtile, 0;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 2;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pmtile, 1;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 4;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pmtile, 2;\n");
    out.push_str("\tshr.u32 \t%r_pmtile, %r_ptile_x, %r_pmtile;\n");
    // atomicAdd mtile_done[m_tile]
    out.push_str("\tshl.b32 \t%r_pbarval, %r_pmtile, 2;\n");
    out.push_str("\tcvt.u64.u32 \t%rd_pbar, %r_pbarval;\n");
    out.push_str("\tadd.u64 \t%rd_pbar, %rd_pmtdone, %rd_pbar;\n");
    out.push_str("\t@%p_pt0 atom.global.add.u32 \t%r_pbarval, [%rd_pbar], 1;\n");
    out.push_str("\tbra \t$L_persistent_loop_back;\n\n");

    // Phase 1 body
    out.push_str("$L_phase1_body:\n");
    for l in &phase1_body {
        let mut line = l.to_string();
        if line.contains("%ctaid.x") && line.contains("mov.u32") {
            line = line.replace("%ctaid.x", "%r_ptile_x");
        }
        if line.contains("%ctaid.y") && line.contains("mov.u32") {
            line = line.replace("%ctaid.y", "%r_ptile_y");
        }
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str("\tbra \t$L_persistent_loop_back;\n\n");

    // Loop back
    out.push_str("$L_persistent_loop_back:\n");
    out.push_str("\tbar.sync \t15;\n");
    out.push_str("\tbra \t$L_persistent_loop;\n\n");
    out.push_str("$L_persistent_exit:\n");
    out.push_str("\tret;\n");
    out.push_str("}\n");

    Ok(out)
}

fn extract_entry_name(line: &str) -> Option<String> {
    // .visible .entry name(
    let entry_pos = line.find(".entry")?;
    let after = line[entry_pos + 6..].trim();
    let paren = after.find('(')?;
    Some(after[..paren].trim().to_string())
}

/// Build a single-body persistent MLP block kernel.
///
/// Instead of duplicating the GEMM body for each phase (gate_up and SiLU-fused-down),
/// this uses ONE GEMM body with:
/// - Params read from shared memory (copied from the active phase's param block at dispatch)
/// - Conditional SiLU at A-load sites (branched on a phase predicate)
///
/// This keeps the instruction footprint ~40KB (fits in 32KB L0 I-cache with minimal misses)
/// and uses ~4 hardware barriers (vs 16 for the two-body approach), allowing 2+ blocks/SM.
///
/// The `silu_computation` contains the per-site and per-element instructions for
/// SiLU+mul fusion (from `build_silu_mul_computation`).
pub fn make_single_body_persistent_mlp(
    flat_gemm_ptx: &str,
    silu_computation: &crate::fuse_general::PointwiseComputation,
    new_name: &str,
) -> Result<String, String> {
    use crate::fuse_cp_async::{
        CpAsyncClass, classify_cp_async_loads, find_all_reg_counts, identify_a_matrix_param,
        parse_cp_async,
    };
    use crate::parser::PtxParser;

    let lines: Vec<&str> = flat_gemm_ptx.lines().collect();
    let proto = PtxParser::parse(flat_gemm_ptx)?;
    let reg_to_param = PtxParser::trace_param_registers_pub(&lines, &proto.params);

    // Classify cp.async loads as A or B matrix
    let a_param_name = identify_a_matrix_param(&lines, &reg_to_param, "ferrite_params")?;
    let a_addr_regs: Vec<String> = reg_to_param
        .iter()
        .filter(|(_, p)| **p == a_param_name)
        .map(|(r, _)| r.clone())
        .collect();
    let classifications = classify_cp_async_loads(&lines, &a_addr_regs);

    let regs = find_all_reg_counts(&lines);

    // Allocate temp registers for the SiLU path (same as replace_a_loads_with_inline_fn)
    let p_mask_idx = regs.pred;
    let r_base = regs.b32;
    let f_base = regs.f32_;

    let new_pred_count = regs.pred + 2; // +1 for mask, +1 for %p_silu
    let new_b32_count = regs.b32 + 4 + silu_computation.scratch_b32_count;
    let new_f32_count = regs.f32_ + 2 + silu_computation.scratch_f32_count;

    let p_mask = format!("%p{p_mask_idx}");
    let r_t = |i: usize| format!("%r{}", r_base + i);
    let f_v = |i: usize| format!("%f{}", f_base + i);

    // ── Extract header, declarations, and body from the flat GEMM ──

    // Header: everything before .entry
    let entry_idx = lines
        .iter()
        .position(|l| l.contains(".visible") && l.contains(".entry"))
        .ok_or("no .entry found")?;
    let header: Vec<&str> = lines[..entry_idx].iter().copied().collect();

    // Body start: first '{'
    let body_start = lines
        .iter()
        .position(|l| l.trim() == "{" || l.trim().ends_with('{'))
        .ok_or("no '{' found")?;

    // Declarations: from body_start+1 to first non-declaration line
    let mut decl_end = body_start + 1;
    while decl_end < lines.len() {
        let t = lines[decl_end].trim();
        if !t.starts_with(".reg")
            && !t.starts_with(".shared")
            && !t.starts_with(".local")
            && !t.starts_with("//")
            && !t.is_empty()
        {
            break;
        }
        decl_end += 1;
    }

    // Body instructions: from decl_end to ret;
    let ret_idx = lines
        .iter()
        .rposition(|l| l.trim() == "ret;")
        .ok_or("no ret; found")?;

    // ── Build the output PTX ──
    let mut out = String::new();

    // Header (version, target, extern shared)
    for l in &header {
        out.push_str(l);
        out.push('\n');
    }

    // Entry point with persistent + original + SiLU params
    out.push_str(&format!(".visible .entry {new_name}(\n"));
    let persistent_params = [
        ".param .u64 _persistent_counter",
        ".param .u32 _persistent_total",
        ".param .u32 _persistent_grid_x_0",
        ".param .u32 _persistent_phase0_tiles",
        ".param .u32 _persistent_grid_x_1",
        ".param .u32 _persistent_ntiles_per_m_0",
        ".param .u64 _persistent_mtile_done",
    ];
    for p in &persistent_params {
        out.push_str(&format!("\t{p},\n"));
    }
    // Phase 0 params
    out.push_str("\t.param .align 1 .b8 ferrite_params[88],\n");
    // SiLU intermediate offset
    out.push_str("\t.param .u64 _ferrite_intermediate_bytes,\n");
    // Phase 1 params
    out.push_str("\t.param .align 1 .b8 ferrite_params_2[88]\n");
    out.push_str(")\n{\n");

    // Original declarations with bumped register counts
    for i in (body_start + 1)..decl_end {
        let t = lines[i].trim();
        if t.starts_with(".reg .pred") && t.contains(&format!("%p<{}>", regs.pred)) {
            out.push_str(&lines[i].replace(
                &format!("%p<{}>", regs.pred),
                &format!("%p<{new_pred_count}>"),
            ));
            out.push('\n');
        } else if t.starts_with(".reg .b32") && t.contains(&format!("%r<{}>", regs.b32)) {
            out.push_str(&lines[i].replace(
                &format!("%r<{}>", regs.b32),
                &format!("%r<{new_b32_count}>"),
            ));
            out.push('\n');
        } else if t.starts_with(".reg .f32") && t.contains(&format!("%f<{}>", regs.f32_)) {
            out.push_str(&lines[i].replace(
                &format!("%f<{}>", regs.f32_),
                &format!("%f<{new_f32_count}>"),
            ));
            out.push('\n');
            // Add bf16 unpacking regs and SiLU scratch
            out.push_str("\t.reg .b16 \t%h_fn<4>;\n");
            if silu_computation.scratch_f32_count > 0 {
                out.push_str(&format!(
                    "\t.reg .f32 \t%f_epi<{}>;\n",
                    silu_computation.scratch_f32_count
                ));
            }
            if silu_computation.scratch_b32_count > 0 {
                out.push_str(&format!(
                    "\t.reg .b32 \t%r_epi<{}>;\n",
                    silu_computation.scratch_b32_count
                ));
            }
            // SiLU-specific named registers
            for decl in &silu_computation.extra_reg_decls {
                out.push_str(&format!("\t{decl}\n"));
            }
        } else {
            out.push_str(lines[i]);
            out.push('\n');
        }
    }

    // Persistent + param-switching scratch registers
    out.push_str("\t// FERRITE: persistent single-body scratch\n");
    out.push_str("\t.reg .u32 \t%r_ptile, %r_ptile_x, %r_ptile_y;\n");
    out.push_str("\t.reg .u32 \t%r_ptotal, %r_pgx0, %r_pgx1;\n");
    out.push_str("\t.reg .u32 \t%r_pp0tiles, %r_plocal, %r_pmtile;\n");
    out.push_str("\t.reg .u32 \t%r_pntpm0, %r_pbarval, %r_pswiz;\n");
    out.push_str("\t.reg .u64 \t%rd_pctr, %rd_pmtdone, %rd_pbar;\n");
    out.push_str("\t.reg .u64 \t%rd_ptmp;\n");
    out.push_str("\t.reg .pred \t%p_pdone, %p_pt0, %p_pphase, %p_pswiz;\n");
    out.push_str("\t.reg .pred \t%p_silu;\n");
    out.push_str("\t.shared .align 4 .u32 _ptile_smem[1];\n");
    out.push_str("\t.shared .align 8 .b8 _active_params[88];\n\n");

    // ── Persistent loop setup ──
    out.push_str("\t// FERRITE: persistent loop setup\n");
    out.push_str("\tld.param.u64 \t%rd_pctr, [_persistent_counter];\n");
    out.push_str("\tcvta.to.global.u64 \t%rd_pctr, %rd_pctr;\n");
    out.push_str("\tld.param.u32 \t%r_ptotal, [_persistent_total];\n");
    out.push_str("\tld.param.u32 \t%r_pgx0, [_persistent_grid_x_0];\n");
    out.push_str("\tld.param.u32 \t%r_pgx1, [_persistent_grid_x_1];\n");
    out.push_str("\tld.param.u32 \t%r_pp0tiles, [_persistent_phase0_tiles];\n");
    out.push_str("\tld.param.u32 \t%r_pntpm0, [_persistent_ntiles_per_m_0];\n");
    out.push_str("\tld.param.u64 \t%rd_pmtdone, [_persistent_mtile_done];\n");
    out.push_str("\tcvta.to.global.u64 \t%rd_pmtdone, %rd_pmtdone;\n\n");

    // ── Persistent loop ──
    out.push_str("$L_persistent_loop:\n");
    out.push_str("\t// Grab next tile\n");
    out.push_str("\tmov.u32 \t%r_ptile, %tid.x;\n");
    out.push_str("\tsetp.eq.u32 \t%p_pt0, %r_ptile, 0;\n");
    out.push_str("\t@%p_pt0 atom.global.add.u32 \t%r_ptile, [%rd_pctr], 1;\n");
    out.push_str("\t@%p_pt0 st.shared.u32 \t[_ptile_smem], %r_ptile;\n");
    out.push_str("\tbar.sync \t14;\n");
    out.push_str("\tld.shared.u32 \t%r_ptile, [_ptile_smem];\n");
    out.push_str("\tsetp.ge.u32 \t%p_pdone, %r_ptile, %r_ptotal;\n");
    out.push_str("\t@%p_pdone bra \t$L_persistent_exit;\n\n");

    // Phase dispatch
    out.push_str("\t// Phase dispatch\n");
    out.push_str("\tsetp.lt.u32 \t%p_pphase, %r_ptile, %r_pp0tiles;\n");
    out.push_str("\t@%p_pphase bra \t$L_dispatch_phase0;\n");
    out.push_str("\tbra \t$L_dispatch_phase1;\n\n");

    // ── Phase 0 dispatch: copy ferrite_params to _active_params, set %p_silu=false ──
    out.push_str("$L_dispatch_phase0:\n");
    out.push_str("\t// Compute ctaid_x/y\n");
    out.push_str("\tmov.u32 \t%r_plocal, %r_ptile;\n");
    out.push_str("\trem.u32 \t%r_ptile_x, %r_plocal, %r_pgx0;\n");
    out.push_str("\tdiv.u32 \t%r_ptile_y, %r_plocal, %r_pgx0;\n");
    // Copy 88 bytes: ferrite_params → _active_params (thread 0 only)
    out.push_str("\t// Copy phase 0 params to shared memory (thread 0 only)\n");
    out.push_str("\tmov.u32 \t%r_pbarval, %tid.x;\n");
    out.push_str("\tsetp.eq.u32 \t%p_pt0, %r_pbarval, 0;\n");
    for offset in (0..88).step_by(8) {
        out.push_str(&format!(
            "\t@%p_pt0 ld.param.u64 \t%rd_ptmp, [ferrite_params+{offset}];\n"
        ));
        out.push_str(&format!(
            "\t@%p_pt0 st.shared.u64 \t[_active_params+{offset}], %rd_ptmp;\n"
        ));
    }
    out.push_str("\tmov.pred \t%p_silu, 0;\n");
    out.push_str("\tbra \t$L_gemm_body;\n\n");

    // ── Phase 1 dispatch: wait for M-tile barrier, copy ferrite_params_2, set %p_silu=true ──
    out.push_str("$L_dispatch_phase1:\n");
    out.push_str("\tsub.u32 \t%r_plocal, %r_ptile, %r_pp0tiles;\n");
    out.push_str("\trem.u32 \t%r_ptile_x, %r_plocal, %r_pgx1;\n");
    out.push_str("\tdiv.u32 \t%r_ptile_y, %r_plocal, %r_pgx1;\n");

    // Compute m_tile for barrier check (same logic as make_persistent_two_phase)
    out.push_str("\t// Compute m_tile for barrier\n");
    out.push_str("\tld.param.s32 \t%r_pbarval, [ferrite_params_2+68];\n"); // N
    out.push_str("\tadd.s32 \t%r_pbarval, %r_pbarval, 127;\n");
    out.push_str("\tshr.u32 \t%r_pbarval, %r_pbarval, 7;\n"); // ceil(N/128)
    out.push_str("\tmov.u32 \t%r_pswiz, 0;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 2;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pswiz, 1;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 4;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pswiz, 2;\n");
    out.push_str("\tshr.u32 \t%r_pmtile, %r_ptile_x, %r_pswiz;\n\n");

    // Per-M-tile barrier: spin until mtile_done[m_tile] >= ntiles_per_m_0
    out.push_str("\t// Per-M-tile barrier\n");
    out.push_str("\tshl.b32 \t%r_pbarval, %r_pmtile, 2;\n");
    out.push_str("\tcvt.u64.u32 \t%rd_pbar, %r_pbarval;\n");
    out.push_str("\tadd.u64 \t%rd_pbar, %rd_pmtdone, %rd_pbar;\n");
    out.push_str("$L_mtile_wait:\n");
    out.push_str("\tld.global.acquire.gpu.u32 \t%r_pbarval, [%rd_pbar];\n");
    out.push_str("\tsetp.ge.u32 \t%p_pdone, %r_pbarval, %r_pntpm0;\n");
    out.push_str("\t@!%p_pdone bra \t$L_mtile_wait;\n\n");

    // Copy phase 1 params to shared memory
    out.push_str("\t// Copy phase 1 params to shared memory (thread 0 only)\n");
    out.push_str("\tmov.u32 \t%r_pbarval, %tid.x;\n");
    out.push_str("\tsetp.eq.u32 \t%p_pt0, %r_pbarval, 0;\n");
    for offset in (0..88).step_by(8) {
        out.push_str(&format!(
            "\t@%p_pt0 ld.param.u64 \t%rd_ptmp, [ferrite_params_2+{offset}];\n"
        ));
        out.push_str(&format!(
            "\t@%p_pt0 st.shared.u64 \t[_active_params+{offset}], %rd_ptmp;\n"
        ));
    }
    // Load intermediate_bytes for SiLU
    out.push_str("\tld.param.u64 \t%rd_up_off, [_ferrite_intermediate_bytes];\n");
    out.push_str("\tmov.pred \t%p_silu, 1;\n");
    out.push_str("\tbra \t$L_gemm_body;\n\n");

    // ── GEMM body ──
    out.push_str("$L_gemm_body:\n");
    out.push_str("\tbar.sync \t13;\n"); // ensure param copy visible

    // Emit the GEMM body with modifications:
    // 1. Replace ld.param [ferrite_params+N] → ld.shared [_active_params+N]
    // 2. Replace %ctaid.x/y with persistent tile registers
    // 3. At A-load cp.async sites: add conditional SiLU
    let mut a_load_index: usize = 0;
    for i in decl_end..ret_idx {
        let line = lines[i];
        let trimmed = line.trim();

        // Replace ctaid.x/y
        if trimmed.contains("%ctaid.x") && trimmed.contains("mov.u32") {
            out.push_str(&line.replace("%ctaid.x", "%r_ptile_x"));
            out.push('\n');
            continue;
        }
        if trimmed.contains("%ctaid.y") && trimmed.contains("mov.u32") {
            out.push_str(&line.replace("%ctaid.y", "%r_ptile_y"));
            out.push('\n');
            continue;
        }

        // Replace ld.param [ferrite_params+N] → ld.shared [_active_params+N]
        if trimmed.contains("ld.param") && trimmed.contains("ferrite_params") {
            let modified = line
                .replace("ld.param", "ld.shared")
                .replace("ferrite_params", "_active_params");
            out.push_str(&modified);
            out.push('\n');
            continue;
        }

        // Conditional SiLU at A-load cp.async sites
        if trimmed.contains("cp.async.cg.shared.global")
            && classifications.get(&i) == Some(&CpAsyncClass::AMatrix)
        {
            if let Some((smem_dst, gmem_src, mask)) = parse_cp_async(trimmed) {
                let label_n = a_load_index;
                // Branch: if not SiLU phase, do normal cp.async
                out.push_str(&format!("\t@!%p_silu bra \t$L_normal_aload_{label_n};\n"));

                // ── SiLU path ──
                // Mask check
                out.push_str(&format!("\tsetp.ne.b32 \t{p_mask}, {mask}, 0;\n"));
                // Load gate values (16 bytes = 8 bf16)
                out.push_str(&format!(
                    "\t@{p_mask} ld.global.v4.b32 \t{{{}, {}, {}, {}}}, [{gmem_src}];\n",
                    r_t(0),
                    r_t(1),
                    r_t(2),
                    r_t(3)
                ));
                for j in 0..4 {
                    out.push_str(&format!("\t@!{p_mask} mov.b32 \t{}, 0;\n", r_t(j)));
                }

                // Per-site instructions (load up values)
                if !silu_computation.per_site.is_empty() {
                    let idx_str = a_load_index.to_string();
                    for instr in &silu_computation.per_site {
                        let concrete = instr
                            .replace("{GMEM_SRC}", &gmem_src)
                            .replace("{MASK_PRED}", &p_mask)
                            .replace("{LOAD_INDEX}", &idx_str);
                        out.push_str(&format!("\t{concrete}\n"));
                    }
                }

                // Apply SiLU+mul per bf16 pair
                for j in 0..4usize {
                    out.push_str(&format!("\tmov.b32 \t{{%h_fn0, %h_fn1}}, {};\n", r_t(j)));
                    out.push_str(&format!("\tcvt.f32.bf16 \t{}, %h_fn0;\n", f_v(0)));
                    out.push_str(&format!("\tcvt.f32.bf16 \t{}, %h_fn1;\n", f_v(1)));

                    let elem_lo = (j * 2).to_string();
                    let elem_hi = (j * 2 + 1).to_string();
                    for instr in &silu_computation.instructions {
                        let c = instr
                            .replace("{INPUT}", &f_v(0))
                            .replace("{ELEM_IDX}", &elem_lo);
                        out.push_str(&format!("\t{c}\n"));
                    }
                    for instr in &silu_computation.instructions {
                        let c = instr
                            .replace("{INPUT}", &f_v(1))
                            .replace("{ELEM_IDX}", &elem_hi);
                        out.push_str(&format!("\t{c}\n"));
                    }

                    out.push_str(&format!("\tcvt.rn.bf16.f32 \t%h_fn0, {};\n", f_v(0)));
                    out.push_str(&format!("\tcvt.rn.bf16.f32 \t%h_fn1, {};\n", f_v(1)));
                    out.push_str(&format!("\tmov.b32 \t{}, {{%h_fn0, %h_fn1}};\n", r_t(j)));
                }

                // Store to SMEM
                out.push_str(&format!(
                    "\tst.shared.v4.b32 \t[{smem_dst}], {{{}, {}, {}, {}}};\n",
                    r_t(0),
                    r_t(1),
                    r_t(2),
                    r_t(3)
                ));
                out.push_str(&format!("\tbra \t$L_after_aload_{label_n};\n"));

                // ── Normal path (cp.async unchanged) ──
                out.push_str(&format!("$L_normal_aload_{label_n}:\n"));
                out.push_str(line);
                out.push('\n');

                out.push_str(&format!("$L_after_aload_{label_n}:\n"));
                a_load_index += 1;
                continue;
            }
        }

        // Default: emit line unchanged
        out.push_str(line);
        out.push('\n');
    }

    // ── Post-body: phase 0 signals M-tile barrier ──
    out.push_str("\n\t// Post-body: phase 0 increments M-tile done\n");
    out.push_str("\t@%p_silu bra \t$L_persistent_loop_back;\n"); // phase 1 skips
    out.push_str("\tbar.sync \t0;\n");
    out.push_str("\tmov.u32 \t%r_pbarval, %tid.x;\n");
    out.push_str("\tsetp.eq.u32 \t%p_pt0, %r_pbarval, 0;\n");
    // Compute m_tile from phase 0's ctaid_x
    out.push_str("\tld.shared.s32 \t%r_pbarval, [_active_params+68];\n"); // N from active params
    out.push_str("\tadd.s32 \t%r_pbarval, %r_pbarval, 127;\n");
    out.push_str("\tshr.u32 \t%r_pbarval, %r_pbarval, 7;\n");
    out.push_str("\tmov.u32 \t%r_pmtile, 0;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 2;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pmtile, 1;\n");
    out.push_str("\tsetp.ge.u32 \t%p_pswiz, %r_pbarval, 4;\n");
    out.push_str("\t@%p_pswiz mov.u32 \t%r_pmtile, 2;\n");
    out.push_str("\tshr.u32 \t%r_pmtile, %r_ptile_x, %r_pmtile;\n");
    // atomicAdd mtile_done[m_tile]
    out.push_str("\tshl.b32 \t%r_pbarval, %r_pmtile, 2;\n");
    out.push_str("\tcvt.u64.u32 \t%rd_pbar, %r_pbarval;\n");
    out.push_str("\tadd.u64 \t%rd_pbar, %rd_pmtdone, %rd_pbar;\n");
    out.push_str("\t@%p_pt0 atom.global.add.u32 \t%r_pbarval, [%rd_pbar], 1;\n");

    // Loop back
    out.push_str("$L_persistent_loop_back:\n");
    out.push_str("\tbar.sync \t15;\n");
    out.push_str("\tbra \t$L_persistent_loop;\n\n");
    out.push_str("$L_persistent_exit:\n");
    out.push_str("\tret;\n");
    out.push_str("}\n");

    Ok(out)
}
