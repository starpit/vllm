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

fn extract_entry_name(line: &str) -> Option<String> {
    // .visible .entry name(
    let entry_pos = line.find(".entry")?;
    let after = line[entry_pos + 6..].trim();
    let paren = after.find('(')?;
    Some(after[..paren].trim().to_string())
}
