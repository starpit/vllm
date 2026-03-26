use proc_macro::TokenStream;
use quote::quote;
use std::path::PathBuf;

pub(crate) mod extract;
mod fuse;
pub(crate) mod fuse_epilogue;
pub(crate) mod fuse_real;
mod parser;
mod regfuse;
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

/// Like `analyze_kernel!` but lets you specify the const name.
/// Useful for nvcc-compiled PTX with mangled kernel names.
///
/// ```rust,ignore
/// analyze_kernel_as!("kernels/vllm_rms_norm.ptx", VLLM_RMS_NORM);
/// // expands to:
/// // const VLLM_RMS_NORM: ptx_fusion::KernelProtocol = KernelProtocol { ... };
/// ```
#[proc_macro]
pub fn analyze_kernel_as(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();

    // Parse: "path.ptx", CONST_NAME
    let (path_str, const_name_str) = parse_path_and_name(&input_str)
        .expect("analyze_kernel_as! expects (\"path.ptx\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&path_str);

    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    let protocol = PtxParser::parse(&ptx_source)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", ptx_path.display()));

    let const_name = syn::Ident::new(&const_name_str, proc_macro2::Span::call_site());
    let tokens = protocol_to_const_tokens(&protocol, &const_name);
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

/// Fuse two PTX kernels via SMEM handoff at compile time.
///
/// ```rust,ignore
/// fuse_kernels!(
///     "kernels/rms_norm.ptx",
///     "kernels/matvec.ptx",
///     fused_name = "fused_rms_norm_matvec",
///     output => input: "output" => "vec_in",
/// );
/// // expands to:
/// // const FUSED_RMS_NORM_MATVEC: &str = "...fused PTX...";
/// ```
#[proc_macro]
pub fn fuse_kernels(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_fuse_args(&input_str).expect("fuse_kernels! parse error");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    let ptx_a = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_a))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_a));
    let ptx_b = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_b))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_b));

    let binding = fuse::FuseBinding {
        a_output_param: args.a_output.clone(),
        b_input_param: args.b_input.clone(),
    };

    let fused = fuse::fuse_kernels(&ptx_a, &ptx_b, &args.fused_name, &binding)
        .unwrap_or_else(|e| panic!("fusion failed: {e}"));

    let fused_ptx = fused.ptx.as_str();
    let const_name = syn::Ident::new(
        &kernel_name_to_upper(&args.fused_name),
        proc_macro2::Span::call_site(),
    );

    let output = quote! {
        const #const_name: &str = #fused_ptx;
    };
    output.into()
}

struct FuseArgs {
    path_a: String,
    path_b: String,
    fused_name: String,
    a_output: String,
    b_input: String,
}

fn parse_fuse_args(input: &str) -> Result<FuseArgs, String> {
    // Parse: "path_a.ptx", "path_b.ptx", fused_name = "name", "a_output" => "b_input"
    let input = input.trim();

    // Extract quoted strings in order
    let mut strings = Vec::new();
    let mut rest = input;
    while let Some(q1) = rest.find('"') {
        let after = &rest[q1 + 1..];
        let q2 = after.find('"').ok_or("unclosed quote")?;
        strings.push(after[..q2].to_string());
        rest = &after[q2 + 1..];
    }

    if strings.len() < 5 {
        return Err(format!(
            "expected 5 quoted strings (path_a, path_b, fused_name, a_output, b_input), got {}",
            strings.len()
        ));
    }

    Ok(FuseArgs {
        path_a: strings[0].clone(),
        path_b: strings[1].clone(),
        fused_name: strings[2].clone(),
        a_output: strings[3].clone(),
        b_input: strings[4].clone(),
    })
}

fn parse_path_and_name(input: &str) -> Result<(String, String), String> {
    // Parse: "path.ptx", CONST_NAME
    let input = input.trim();
    let q1 = input.find('"').ok_or("expected opening quote")?;
    let rest = &input[q1 + 1..];
    let q2 = rest.find('"').ok_or("expected closing quote")?;
    let path = rest[..q2].to_string();
    let after = rest[q2 + 1..].trim().trim_start_matches(',').trim();
    let name = after.trim().to_string();
    if name.is_empty() {
        return Err("expected const name after path".to_string());
    }
    Ok((path, name))
}

/// Extract a single entry point from a multi-entry PTX file.
///
/// ```rust,ignore
/// extract_entry!("kernels/vllm_rms_norm.ptx", "rms_norm_kernelIf", VLLM_RMS_NORM_F32);
/// // expands to:
/// // const VLLM_RMS_NORM_F32: &str = "...standalone PTX with just the float entry...";
/// ```
#[proc_macro]
pub fn extract_entry(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_extract_args(&input_str)
        .expect("extract_entry! expects (\"path.ptx\", \"entry_substring\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&args.0);
    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    let extracted = extract::extract_entry(&ptx_source, &args.1)
        .unwrap_or_else(|e| panic!("extract_entry failed: {e}"));

    let extracted_str = extracted.as_str();
    let const_name = syn::Ident::new(&args.2, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #extracted_str;
    };
    output.into()
}

fn parse_extract_args(input: &str) -> Result<(String, String, String), String> {
    // Parse: "path.ptx", "entry_name", CONST_NAME
    let mut strings = Vec::new();
    let mut rest = input.trim();
    // Extract 2 quoted strings
    for _ in 0..2 {
        let q1 = rest.find('"').ok_or("expected quote")?;
        let after = &rest[q1 + 1..];
        let q2 = after.find('"').ok_or("unclosed quote")?;
        strings.push(after[..q2].to_string());
        rest = after[q2 + 1..].trim().trim_start_matches(',').trim();
    }
    // The remainder is the const name
    let name = rest.trim().to_string();
    if name.is_empty() {
        return Err("expected CONST_NAME after entry name".to_string());
    }
    Ok((strings[0].clone(), strings[1].clone(), name))
}

/// Register-level fusion: fuse two elementwise kernels where the intermediate
/// stays in registers — no SMEM, no GMEM, no barrier.
///
/// ```rust,ignore
/// regfuse_kernels!(
///     "kernels/rms_norm.ptx",
///     "kernels/scale.ptx",
///     "fused_rms_norm_scale",
///     "output",
///     "input"
/// );
/// // expands to:
/// // const FUSED_RMS_NORM_SCALE: &str = "...fused PTX...";
/// ```
#[proc_macro]
pub fn regfuse_kernels(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_fuse_args(&input_str).expect("regfuse_kernels! parse error");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    let ptx_a = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_a))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_a));
    let ptx_b = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_b))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_b));

    let binding = regfuse::RegFuseBinding {
        a_output_param: args.a_output.clone(),
        b_input_param: args.b_input.clone(),
    };

    let fused = regfuse::regfuse_kernels(&ptx_a, &ptx_b, &args.fused_name, &binding)
        .unwrap_or_else(|e| panic!("register fusion failed: {e}"));

    let fused_ptx = fused.ptx.as_str();
    let const_name = syn::Ident::new(
        &kernel_name_to_upper(&args.fused_name),
        proc_macro2::Span::call_site(),
    );

    let output = quote! {
        const #const_name: &str = #fused_ptx;
    };
    output.into()
}

/// Fuse two real nvcc-compiled kernels via SMEM handoff.
///
/// Takes multi-entry PTX files + entry name substrings to select the right specialization.
///
/// ```rust,ignore
/// fuse_real_kernels!(
///     "kernels/vllm_rms_norm.ptx", "rms_norm_kernelIfE",
///     "kernels/vllm_silu_mul.ptx", "act_and_mul_kernelIXadL_Z4silufEEfE",
///     "fused_rms_silu",
///     "param_0",  // A's output param (substring match)
///     "param_1",  // B's input param (substring match)
///     1024,       // SMEM handoff buffer size (elements)
///     FUSED_RMS_SILU
/// );
/// ```
#[proc_macro]
pub fn fuse_real_kernels(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_fuse_real_args(&input_str).expect("fuse_real_kernels! parse error");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    let ptx_a_full = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_a))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_a));
    let ptx_b_full = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_b))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_b));

    // Extract single entries
    let ptx_a = extract::extract_entry(&ptx_a_full, &args.entry_a)
        .unwrap_or_else(|e| panic!("extract A failed: {e}"));
    let ptx_b = extract::extract_entry(&ptx_b_full, &args.entry_b)
        .unwrap_or_else(|e| panic!("extract B failed: {e}"));

    let binding = fuse_real::RealFuseBinding {
        a_output_param: args.a_output.clone(),
        b_input_param: args.b_input.clone(),
    };

    let fused = fuse_real::fuse_real_kernels(
        &ptx_a,
        &ptx_b,
        &args.fused_name,
        &binding,
        args.smem_elements,
    )
    .unwrap_or_else(|e| panic!("real fusion failed: {e}"));

    let fused_ptx = fused.ptx.as_str();
    let const_name = syn::Ident::new(&args.const_name, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #fused_ptx;
    };
    output.into()
}

struct FuseRealArgs {
    path_a: String,
    entry_a: String,
    path_b: String,
    entry_b: String,
    fused_name: String,
    a_output: String,
    b_input: String,
    smem_elements: usize,
    const_name: String,
}

fn parse_fuse_real_args(input: &str) -> Result<FuseRealArgs, String> {
    let mut strings = Vec::new();
    let mut rest = input.trim();
    while let Some(q1) = rest.find('"') {
        let after = &rest[q1 + 1..];
        let q2 = after.find('"').ok_or("unclosed quote")?;
        strings.push(after[..q2].to_string());
        rest = &after[q2 + 1..];
    }

    if strings.len() < 7 {
        return Err(format!("expected 7 quoted strings, got {}", strings.len()));
    }

    let rest = rest.trim().trim_start_matches(',').trim();
    let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
    if parts.len() < 2 {
        return Err("expected smem_elements and CONST_NAME after quoted strings".to_string());
    }

    let smem_elements: usize = parts[0]
        .trim()
        .parse()
        .map_err(|e| format!("bad smem_elements: {e}"))?;
    let const_name = parts[1].trim().to_string();

    Ok(FuseRealArgs {
        path_a: strings[0].clone(),
        entry_a: strings[1].clone(),
        path_b: strings[2].clone(),
        entry_b: strings[3].clone(),
        fused_name: strings[4].clone(),
        a_output: strings[5].clone(),
        b_input: strings[6].clone(),
        smem_elements,
        const_name,
    })
}

/// Inject SiLU activation into a CUTLASS GEMM epilogue at compile time.
///
/// Extracts a single entry from a multi-entry PTX file, then injects SiLU
/// on every f32 value before bf16 conversion in the epilogue.
///
/// ```rust,ignore
/// inject_silu_epilogue!(
///     "kernels/cutlass_gemm_sm89.ptx",
///     "GemmShapeILi64ELi128ELi64",   // entry substring
///     CUTLASS_GEMM_WITH_SILU
/// );
/// ```
#[proc_macro]
pub fn inject_silu_epilogue(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_extract_args(&input_str)
        .expect("inject_silu_epilogue! expects (\"path.ptx\", \"entry_substr\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&args.0);
    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    let extracted = extract::extract_entry(&ptx_source, &args.1)
        .unwrap_or_else(|e| panic!("extract_entry failed: {e}"));

    // Try bf16x2 injection first, fall back to f32 store injection
    let modified = fuse_epilogue::inject_silu_into_epilogue(&extracted)
        .or_else(|_| fuse_epilogue::inject_silu_into_f32_stores(&extracted))
        .unwrap_or_else(|e| panic!("SiLU injection failed: {e}"));

    let modified_str = modified.as_str();
    let const_name = syn::Ident::new(&args.2, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #modified_str;
    };
    output.into()
}
