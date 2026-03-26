/// Extract the entry point matching `entry_name` (substring match on the mangled name)
/// from a multi-entry PTX module. nvcc compiles all template instantiations into one
/// PTX file; this extracts just the header + one specific entry as standalone valid PTX.
/// Returns standalone PTX with header + that single entry.
pub fn extract_entry(ptx: &str, entry_name: &str) -> Result<String, String> {
    let lines: Vec<&str> = ptx.lines().collect();

    // Find the target entry point
    let mut entry_start = None;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.contains(".entry") && t.contains(entry_name) {
            entry_start = Some(i);
            break;
        }
    }
    let start = entry_start.ok_or_else(|| format!("entry '{entry_name}' not found in PTX"))?;

    // Find the end of this entry (matching closing brace)
    let mut brace_depth = 0;
    let mut entry_end = start;
    let mut found_open = false;

    for (i, line) in lines.iter().enumerate().skip(start) {
        let t = line.trim();
        if t.contains('{') {
            brace_depth += t.matches('{').count();
            found_open = true;
        }
        if t.contains('}') {
            brace_depth -= t.matches('}').count();
            if found_open && brace_depth == 0 {
                entry_end = i;
                break;
            }
        }
    }

    if !found_open {
        return Err(format!("no opening brace found for entry '{entry_name}'"));
    }

    // Collect the entry body text
    let entry_text: String = lines[start..=entry_end]
        .iter()
        .map(|l| format!("{l}\n"))
        .collect();

    // Scan the entry body for referenced symbols, then find matching top-level
    // .shared declarations that need to be included.
    let mut shared_decls = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if i >= start {
            break; // Only look at top-level declarations before our entry
        }
        let t = line.trim();
        if t.starts_with(".shared") || (t.starts_with(".extern") && t.contains(".shared")) {
            // Extract the symbol name from the declaration
            // e.g., ".shared .align 4 .b8 _ZZ16block_reduce_sumfE6shared[128];"
            if let Some(name) = extract_shared_name(t) {
                // Check if the entry body references this symbol
                if entry_text.contains(&name) {
                    shared_decls.push(*line);
                }
            }
        }
    }

    // Build the result: header + referenced shared decls + entry
    let mut result = String::new();

    // Header: .version, .target, .address_size
    for line in &lines {
        let t = line.trim();
        if t.starts_with(".version") || t.starts_with(".target") || t.starts_with(".address_size") {
            result.push_str(line);
            result.push('\n');
        }
        // Stop after address_size (everything after is entries/declarations)
        if t.starts_with(".address_size") {
            break;
        }
    }
    result.push('\n');

    // Referenced shared declarations
    for decl in &shared_decls {
        result.push_str(decl);
        result.push('\n');
    }
    if !shared_decls.is_empty() {
        result.push('\n');
    }

    // The entry itself
    result.push_str(&entry_text);

    Ok(result)
}

/// Extract the symbol name from a .shared declaration line.
fn extract_shared_name(line: &str) -> Option<String> {
    // Patterns:
    //   .shared .align 4 .b8 _ZZ16block_reduce_sumfE6shared[128];
    //   .extern .shared .align 16 .b8 smem[];
    //   .shared .align 4 .f32 some_name;
    let parts: Vec<&str> = line.split_whitespace().collect();
    for part in &parts {
        let clean = part.trim_end_matches(';');
        // The name is the part that contains a letter and isn't a directive
        if !clean.starts_with('.') && clean.contains(|c: char| c.is_alphabetic()) {
            // Strip array suffix [N] if present
            let name = if let Some(bracket) = clean.find('[') {
                &clean[..bracket]
            } else {
                clean
            };
            return Some(name.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_from_multi_entry() {
        let ptx = r#"
.version 8.8
.target sm_89
.address_size 64

.visible .entry foo_f32(
    .param .u64 foo_f32_param_0
)
{
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [foo_f32_param_0];
    ret;
}

.visible .entry foo_f16(
    .param .u64 foo_f16_param_0
)
{
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [foo_f16_param_0];
    ret;
}

.visible .entry bar_f32(
    .param .u64 bar_f32_param_0
)
{
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [bar_f32_param_0];
    ret;
}
"#;

        let extracted = extract_entry(ptx, "foo_f16").unwrap();
        assert!(extracted.contains(".version 8.8"));
        assert!(extracted.contains(".target sm_89"));
        assert!(extracted.contains("foo_f16"));
        assert!(!extracted.contains("foo_f32"));
        assert!(!extracted.contains("bar_f32"));

        let extracted2 = extract_entry(ptx, "bar_f32").unwrap();
        assert!(extracted2.contains("bar_f32"));
        assert!(!extracted2.contains("foo_f32"));
        assert!(!extracted2.contains("foo_f16"));
    }

    #[test]
    fn test_extract_includes_referenced_shared() {
        let ptx = r#".version 8.8
.target sm_89
.address_size 64

.shared .align 4 .b8 my_shared[128];
.shared .align 4 .b8 other_shared[64];

.visible .entry kernel_a(
    .param .u64 kernel_a_param_0
)
{
    .reg .b32 %r<2>;
    mov.u32 %r1, my_shared;
    ret;
}
"#;

        let extracted = extract_entry(ptx, "kernel_a").unwrap();
        assert!(
            extracted.contains("my_shared"),
            "should include referenced shared"
        );
        assert!(
            !extracted.contains("other_shared"),
            "should NOT include unreferenced shared"
        );
    }

    #[test]
    fn test_extract_not_found() {
        let ptx = ".version 8.8\n.target sm_89\n.address_size 64\n";
        assert!(extract_entry(ptx, "nonexistent").is_err());
    }
}
