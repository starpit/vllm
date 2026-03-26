use proc_macro::TokenStream;
use quote::quote;
use std::path::PathBuf;

mod parser;
use parser::{KernelProtocol, PtxParser};

/// Analyze a PTX kernel and emit its `KernelProtocol` as a const at compile time.
///
/// ```rust,ignore
/// analyze_kernel!("kernels/rms_norm.ptx");
/// // expands to:
/// // const RMS_NORM: ptx_fusion::KernelProtocol = KernelProtocol { ... };
/// ```
#[proc_macro]
pub fn analyze_kernel(input: TokenStream) -> TokenStream {
    let lit: syn::LitStr =
        syn::parse(input).expect("analyze_kernel! expects a string literal path");
    let path_str = lit.value();

    // Resolve path relative to CARGO_MANIFEST_DIR of the *calling* crate
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&path_str);

    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    let protocol = PtxParser::parse(&ptx_source)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", ptx_path.display()));

    let tokens = protocol_to_tokens(&protocol);
    tokens.into()
}

/// Rewrite a PTX kernel's registers according to a rename map, proving we can
/// transform PTX programmatically. Emits the rewritten PTX as a `&str` const.
///
/// ```rust,ignore
/// rewrite_kernel!("kernels/rms_norm.ptx", {
///     "%f3" => "%f30",
///     "%r3" => "%r30",
/// });
/// // expands to:
/// // const RMS_NORM_REWRITTEN: &str = "...";
/// // const RMS_NORM_REWRITTEN_PROTOCOL: ptx_fusion::KernelProtocol = ...;
/// ```
#[proc_macro]
pub fn rewrite_kernel(input: TokenStream) -> TokenStream {
    let input = input.to_string();

    // Parse: "path", { "old" => "new", ... }
    let (path_str, renames) = parse_rewrite_args(&input)
        .expect("rewrite_kernel! expects (\"path.ptx\", { \"%r3\" => \"%r30\", ... })");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&path_str);

    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    // Verify original parses
    let original_protocol =
        PtxParser::parse(&ptx_source).unwrap_or_else(|e| panic!("failed to parse original: {e}"));

    // Apply register renames
    let rewritten = apply_register_renames(&ptx_source, &renames);

    // Verify rewritten also parses and extract its protocol
    let rewritten_protocol = PtxParser::parse(&rewritten)
        .unwrap_or_else(|e| panic!("rewritten PTX failed to parse: {e}"));

    let const_name_upper = kernel_name_to_upper(&original_protocol.name);
    let ptx_const = syn::Ident::new(
        &format!("{const_name_upper}_REWRITTEN"),
        proc_macro2::Span::call_site(),
    );
    let proto_const = syn::Ident::new(
        &format!("{const_name_upper}_REWRITTEN_PROTOCOL"),
        proc_macro2::Span::call_site(),
    );

    let rewritten_str = rewritten.as_str();
    let proto_tokens = protocol_to_const_tokens(&rewritten_protocol, &proto_const);

    let output = quote! {
        const #ptx_const: &str = #rewritten_str;
        #proto_tokens
    };
    output.into()
}

// ── helpers ──────────────────────────────────────────────────────────

fn kernel_name_to_upper(name: &str) -> String {
    // rms_norm -> RMS_NORM
    name.to_uppercase()
}

fn protocol_to_tokens(protocol: &KernelProtocol) -> proc_macro2::TokenStream {
    let const_name = syn::Ident::new(
        &kernel_name_to_upper(&protocol.name),
        proc_macro2::Span::call_site(),
    );
    protocol_to_const_tokens(protocol, &const_name)
}

fn protocol_to_const_tokens(
    protocol: &KernelProtocol,
    const_name: &syn::Ident,
) -> proc_macro2::TokenStream {
    let name = &protocol.name;

    // Register budget lines
    let reg_lines: Vec<proc_macro2::TokenStream> = protocol
        .registers
        .iter()
        .map(|(ty, count)| {
            quote! { (#ty, #count) }
        })
        .collect();

    // SMEM regions
    let smem_lines: Vec<proc_macro2::TokenStream> = protocol
        .smem_regions
        .iter()
        .map(|r| {
            let rname = &r.name;
            let align = r.align;
            let elem_type = &r.elem_type;
            let count = r.count;
            let bytes = r.size_bytes;
            quote! {
                ptx_fusion::SmemRegion {
                    name: #rname,
                    align: #align,
                    elem_type: #elem_type,
                    count: #count,
                    size_bytes: #bytes,
                }
            }
        })
        .collect();

    // Params
    let param_lines: Vec<proc_macro2::TokenStream> = protocol
        .params
        .iter()
        .map(|p| {
            let pname = &p.name;
            let pty = &p.ptx_type;
            let is_ptr = p.is_pointer;
            let idx = p.index as u32;
            quote! {
                ptx_fusion::KernelParam {
                    name: #pname,
                    ptx_type: #pty,
                    is_pointer: #is_ptr,
                    index: #idx,
                }
            }
        })
        .collect();

    // Data ports (loads/stores)
    let load_lines: Vec<proc_macro2::TokenStream> = protocol
        .global_loads
        .iter()
        .map(|d| {
            let param = &d.param_name;
            let ty = &d.data_type;
            let line = d.line as u32;
            quote! {
                ptx_fusion::DataPort { param_name: #param, data_type: #ty, line: #line }
            }
        })
        .collect();

    let store_lines: Vec<proc_macro2::TokenStream> = protocol
        .global_stores
        .iter()
        .map(|d| {
            let param = &d.param_name;
            let ty = &d.data_type;
            let line = d.line as u32;
            quote! {
                ptx_fusion::DataPort { param_name: #param, data_type: #ty, line: #line }
            }
        })
        .collect();

    let smem_load_count = protocol.smem_loads as u32;
    let smem_store_count = protocol.smem_stores as u32;
    let barrier_ids: Vec<u32> = protocol.barriers.iter().map(|b| *b as u32).collect();
    let has_mma = protocol.has_mma;
    let total_smem = protocol.total_smem_bytes as u32;

    quote! {
        const #const_name: ptx_fusion::KernelProtocol = ptx_fusion::KernelProtocol {
            name: #name,
            registers: &[#(#reg_lines),*],
            smem_regions: &[#(#smem_lines),*],
            total_smem_bytes: #total_smem,
            params: &[#(#param_lines),*],
            global_loads: &[#(#load_lines),*],
            global_stores: &[#(#store_lines),*],
            smem_loads: #smem_load_count,
            smem_stores: #smem_store_count,
            barriers: &[#(#barrier_ids),*],
            has_mma: #has_mma,
        };
    }
}

fn parse_rewrite_args(input: &str) -> Result<(String, Vec<(String, String)>), String> {
    // Simple parser for: "path.ptx", { "%r3" => "%r30", "%f3" => "%f30" }
    let input = input.trim();

    // Extract the path string
    let first_quote = input.find('"').ok_or("expected opening quote")?;
    let rest = &input[first_quote + 1..];
    let second_quote = rest.find('"').ok_or("expected closing quote")?;
    let path = rest[..second_quote].to_string();

    let rest = &rest[second_quote + 1..];

    // Find the { ... } block
    let brace_start = rest.find('{').ok_or("expected '{'")?;
    let brace_end = rest.rfind('}').ok_or("expected '}'")?;
    let body = &rest[brace_start + 1..brace_end];

    let mut renames = Vec::new();
    for segment in body.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        // Parse "old" => "new"
        let parts: Vec<&str> = segment.split("=>").collect();
        if parts.len() != 2 {
            return Err(format!("expected 'old => new', got: {segment}"));
        }
        let old = parts[0].trim().trim_matches('"').to_string();
        let new = parts[1].trim().trim_matches('"').to_string();
        renames.push((old, new));
    }

    Ok((path, renames))
}

fn apply_register_renames(ptx: &str, renames: &[(String, String)]) -> String {
    let mut result = ptx.to_string();

    // Sort renames by length descending to avoid partial matches
    // e.g., rename %r10 before %r1
    let mut sorted_renames = renames.to_vec();
    sorted_renames.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    // Two-pass rename via placeholders to avoid collisions
    let mut placeholders: Vec<(String, String)> = Vec::new();
    for (i, (old, _new)) in sorted_renames.iter().enumerate() {
        let placeholder = format!("__PTX_RENAME_PLACEHOLDER_{i}__");
        // Only rename register *uses*, not declarations of different registers
        // We need word-boundary-aware replacement
        result = rename_register_in_ptx(&result, old, &placeholder);
        placeholders.push((placeholder, _new.clone()));
    }

    // Second pass: replace placeholders with final names
    for (placeholder, new_name) in &placeholders {
        result = result.replace(placeholder, new_name);
    }

    // Also update .reg declarations to include the new register numbers
    result = update_reg_declarations(&result, renames);

    result
}

fn rename_register_in_ptx(ptx: &str, old_reg: &str, new_reg: &str) -> String {
    // Replace register references: %r3 but not %r30
    // A register reference is followed by a non-alphanumeric character (or end of line)
    let mut result = String::with_capacity(ptx.len());
    let mut remaining = ptx;

    while let Some(pos) = remaining.find(old_reg) {
        // Check that the character after the match is not alphanumeric (word boundary)
        let after = pos + old_reg.len();
        let is_word_boundary =
            after >= remaining.len() || !remaining.as_bytes()[after].is_ascii_alphanumeric();

        if is_word_boundary {
            result.push_str(&remaining[..pos]);
            result.push_str(new_reg);
            remaining = &remaining[after..];
        } else {
            // Not a word boundary, skip this occurrence
            result.push_str(&remaining[..after]);
            remaining = &remaining[after..];
        }
    }
    result.push_str(remaining);
    result
}

fn update_reg_declarations(ptx: &str, renames: &[(String, String)]) -> String {
    // If we renamed %f3 -> %f30, the .reg .f32 %f<16> declaration needs to
    // become .reg .f32 %f<31> to accommodate the new register number.
    // Parse each .reg line and ensure the count is large enough.
    let mut lines: Vec<String> = ptx.lines().map(|l| l.to_string()).collect();

    for line in &mut lines {
        let trimmed = line.trim();
        if !trimmed.starts_with(".reg") {
            continue;
        }

        // Parse: .reg .f32 %f<16>;
        // Find the register prefix and count
        if let Some(angle_start) = trimmed.find('<')
            && let Some(angle_end) = trimmed.find('>')
        {
            let prefix_start = trimmed[..angle_start].rfind('%').unwrap_or(0);
            let prefix = &trimmed[prefix_start..angle_start]; // e.g., "%f"
            let current_count: usize = trimmed[angle_start + 1..angle_end].parse().unwrap_or(0);

            // Check if any rename targets this prefix
            let mut max_needed = current_count;
            for (_old, new) in renames {
                if let Some(num_str) = new.strip_prefix(prefix)
                    && let Ok(num) = num_str.parse::<usize>()
                {
                    max_needed = max_needed.max(num + 1);
                }
            }

            if max_needed > current_count {
                let old_decl = format!("<{current_count}>");
                let new_decl = format!("<{max_needed}>");
                *line = line.replace(&old_decl, &new_decl);
            }
        }
    }

    lines.join("\n")
}
