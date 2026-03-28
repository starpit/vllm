use proc_macro::TokenStream;
use quote::quote;
use std::path::PathBuf;

pub(crate) mod chain;
pub(crate) mod compile;
pub(crate) mod extract;
mod fuse;
pub(crate) mod fuse_cp_async;
pub(crate) mod fuse_epilogue;
pub(crate) mod fuse_general;
pub(crate) mod fuse_real;
pub(crate) mod intrinsic_rms_norm;
mod parser;
pub(crate) mod perimeter;
pub(crate) mod persistent;
mod pipeline;
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

    let async_load_lines: Vec<proc_macro2::TokenStream> = protocol
        .async_loads
        .iter()
        .map(|a| {
            let param = &a.param_name;
            let smem_dst = &a.smem_dst;
            let gmem_src = &a.gmem_src;
            let mask = &a.mask;
            let size = a.size_bytes as u32;
            let line = a.line as u32;
            quote! {
                ptx_fusion::AsyncCopyPort {
                    param_name: #param, smem_dst: #smem_dst, gmem_src: #gmem_src,
                    mask: #mask, size_bytes: #size, line: #line
                }
            }
        })
        .collect();

    let smem_load_count = protocol.smem_loads as u32;
    let smem_store_count = protocol.smem_stores as u32;
    let barrier_ids: Vec<u32> = protocol.barriers.iter().map(|b| *b as u32).collect();
    let has_mma = protocol.has_mma;
    let total_smem = protocol.total_smem_bytes as u32;

    let classified_param_lines: Vec<proc_macro2::TokenStream> = protocol
        .classified_params
        .iter()
        .map(|cp| {
            let offset = cp.offset;
            let ptx_type = &cp.ptx_type;
            let line = cp.line as u32;
            let role = match cp.role {
                parser::ParamRole::Pointer => quote! { ptx_fusion::ParamRole::Pointer },
                parser::ParamRole::Stride => quote! { ptx_fusion::ParamRole::Stride },
                parser::ParamRole::Dimension => quote! { ptx_fusion::ParamRole::Dimension },
                parser::ParamRole::Scalar => quote! { ptx_fusion::ParamRole::Scalar },
                parser::ParamRole::Derived => quote! { ptx_fusion::ParamRole::Derived },
            };
            quote! {
                ptx_fusion::ClassifiedParam {
                    offset: #offset,
                    ptx_type: #ptx_type,
                    role: #role,
                    line: #line,
                }
            }
        })
        .collect();

    quote! {
        const #const_name: ptx_fusion::KernelProtocol = ptx_fusion::KernelProtocol {
            name: #name,
            registers: &[#(#reg_lines),*],
            smem_regions: &[#(#smem_lines),*],
            total_smem_bytes: #total_smem,
            params: &[#(#param_lines),*],
            global_loads: &[#(#load_lines),*],
            global_stores: &[#(#store_lines),*],
            async_loads: &[#(#async_load_lines),*],
            smem_loads: #smem_load_count,
            smem_stores: #smem_store_count,
            barriers: &[#(#barrier_ids),*],
            has_mma: #has_mma,
            classified_params: &[#(#classified_param_lines),*],
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

fn parse_epilogue_args(input: &str) -> Result<(String, String, String, String), String> {
    // Parse: "path.ptx", "entry_name", Activation, CONST_NAME
    let mut strings = Vec::new();
    let mut rest = input.trim();
    for _ in 0..2 {
        let q1 = rest.find('"').ok_or("expected quote")?;
        let after = &rest[q1 + 1..];
        let q2 = after.find('"').ok_or("unclosed quote")?;
        strings.push(after[..q2].to_string());
        rest = after[q2 + 1..].trim().trim_start_matches(',').trim();
    }
    let comma_pos = rest.find(',').ok_or("expected comma after activation")?;
    let act_str = rest[..comma_pos].trim().to_string();
    rest = rest[comma_pos + 1..].trim();
    let name = rest.trim().to_string();
    if name.is_empty() {
        return Err("expected CONST_NAME".to_string());
    }
    Ok((strings[0].clone(), strings[1].clone(), act_str, name))
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

/// Fuse two real nvcc-compiled kernels via SMEM handoff AND wrap in a persistent
/// work-queue loop. The resulting kernel fills the GPU and loops, grabbing tiles
/// from an atomic counter until all rows are processed.
///
/// ```rust,ignore
/// persistent_fuse_real_kernels!(
///     "kernels/vllm_rms_norm.ptx", "rms_norm_kernelIfE",
///     "kernels/gemm_row_f32.ptx", "gemm_row_f32",
///     "persistent_rms_norm_gemm",
///     "param_0", "param_1",
///     4096,
///     "param_3",  // which B param holds the total row count (M)
///     PERSISTENT_RMS_GEMM_PTX
/// );
/// ```
#[proc_macro]
pub fn persistent_fuse_real_kernels(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args =
        parse_persistent_fuse_args(&input_str).expect("persistent_fuse_real_kernels! parse error");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    let ptx_a_full = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_a))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_a));
    let ptx_b_full = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_b))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_b));

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

    // Build the full param name: B_entry_name + "_" + total_rows_param
    // e.g., "gemm_row_f32" + "_" + "param_3" = "gemm_row_f32_param_3"
    let total_rows_full = format!("{}_{}", args.entry_b, args.total_rows_param);

    let persistent =
        persistent::make_persistent(&fused.ptx, &args.persistent_name, &total_rows_full)
            .unwrap_or_else(|e| panic!("persistent wrapper failed: {e}"));

    let persistent_str = persistent.as_str();
    let const_name = syn::Ident::new(&args.const_name, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #persistent_str;
    };
    output.into()
}

struct PersistentFuseArgs {
    path_a: String,
    entry_a: String,
    path_b: String,
    entry_b: String,
    fused_name: String,
    persistent_name: String,
    a_output: String,
    b_input: String,
    smem_elements: usize,
    total_rows_param: String,
    const_name: String,
}

fn parse_persistent_fuse_args(input: &str) -> Result<PersistentFuseArgs, String> {
    let mut strings = Vec::new();
    let mut rest = input.trim();
    while let Some(q1) = rest.find('"') {
        let after = &rest[q1 + 1..];
        let q2 = after.find('"').ok_or("unclosed quote")?;
        strings.push(after[..q2].to_string());
        rest = &after[q2 + 1..];
    }

    // 8 quoted strings: path_a, entry_a, path_b, entry_b, fused_name, a_output, b_input, total_rows_param
    if strings.len() < 8 {
        return Err(format!("expected 8 quoted strings, got {}", strings.len()));
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

    // The persistent entry name is the fused_name prefixed with "persistent_"
    // (but we use the fused_name for the intermediate fusion step)
    let persistent_name = format!("persistent_{}", strings[4]);

    Ok(PersistentFuseArgs {
        path_a: strings[0].clone(),
        entry_a: strings[1].clone(),
        path_b: strings[2].clone(),
        entry_b: strings[3].clone(),
        fused_name: strings[4].clone(),
        persistent_name,
        a_output: strings[5].clone(),
        b_input: strings[6].clone(),
        smem_elements,
        total_rows_param: strings[7].clone(),
        const_name,
    })
}

/// Fuse a 3-phase MLP pipeline: norm -> GEMM+SiLU -> GEMM, wrapped in a persistent loop.
///
/// Phase 1->2: SMEM handoff (eliminates norm->GEMM GMEM round-trip).
/// Phase 2->3: GMEM handoff (Phase 3 reads from Phase 2's output buffer).
/// SiLU is injected into Phase 2's f32 stores.
///
/// ```rust,ignore
/// fuse_3phase_mlp!(
///     "kernels/vllm_rms_norm.ptx", "rms_norm_kernelIfE",
///     "kernels/gemm_row_f32.ptx", "gemm_row_f32",
///     "kernels/gemm_row_f32.ptx", "gemm_row_f32",
///     "fused_mlp",
///     "param_0", "param_1",  // Phase 1->2 SMEM binding
///     "param_0", "param_1",  // Phase 2->3 GMEM binding
///     4096,                   // SMEM elements
///     "param_3",              // total_rows param
///     FUSED_MLP_PTX
/// );
/// ```
#[proc_macro]
pub fn fuse_3phase_mlp(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_3phase_args(&input_str).expect("fuse_3phase_mlp! parse error");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    // Read all PTX files
    let ptx_norm_full = std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_norm))
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_norm));
    let ptx_gemm1_full =
        std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_gemm1))
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_gemm1));
    let ptx_gemm2_full =
        std::fs::read_to_string(PathBuf::from(&manifest_dir).join(&args.path_gemm2))
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", args.path_gemm2));

    // Extract entries
    let ptx_norm = extract::extract_entry(&ptx_norm_full, &args.entry_norm)
        .unwrap_or_else(|e| panic!("extract norm failed: {e}"));
    let ptx_gemm1 = extract::extract_entry(&ptx_gemm1_full, &args.entry_gemm1)
        .unwrap_or_else(|e| panic!("extract gemm1 failed: {e}"));
    let ptx_gemm2 = extract::extract_entry(&ptx_gemm2_full, &args.entry_gemm2)
        .unwrap_or_else(|e| panic!("extract gemm2 failed: {e}"));

    // Step 1: fuse_real(norm, gemm1) -> fused_12
    let binding_12 = fuse_real::RealFuseBinding {
        a_output_param: args.smem_a_output.clone(),
        b_input_param: args.smem_b_input.clone(),
    };
    let fused_12 = fuse_real::fuse_real_kernels(
        &ptx_norm,
        &ptx_gemm1,
        &args.fused_name,
        &binding_12,
        args.smem_elements,
    )
    .unwrap_or_else(|e| panic!("Phase 1+2 fusion failed: {e}"));

    // Step 2: inject SiLU on Phase B's f32 stores
    let fused_12_silu = fuse_epilogue::inject_silu_into_f32_stores(&fused_12.ptx)
        .unwrap_or_else(|e| panic!("SiLU injection failed: {e}"));

    // Step 3: append Phase C (gemm2) with SMEM handoff
    let fused_123_name = format!("{}_3phase", args.fused_name);
    let fused_123 = chain::append_phase_smem(
        &fused_12_silu,
        &ptx_gemm2,
        &args.gmem_b_output,
        &args.gmem_c_input,
        &fused_123_name,
        args.smem_elements,
    )
    .unwrap_or_else(|e| panic!("Phase 3 SMEM append failed: {e}"));

    // Step 4: wrap in persistent loop
    let persistent_name = format!("persistent_{}", fused_123_name);
    let total_rows_full = format!("{}_{}", args.entry_gemm1, args.total_rows_param);
    let persistent = persistent::make_persistent(&fused_123, &persistent_name, &total_rows_full)
        .unwrap_or_else(|e| panic!("persistent wrapper failed: {e}"));

    let persistent_str = persistent.as_str();
    let const_name = syn::Ident::new(&args.const_name, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #persistent_str;
    };
    output.into()
}

struct ThreePhaseArgs {
    path_norm: String,
    entry_norm: String,
    path_gemm1: String,
    entry_gemm1: String,
    path_gemm2: String,
    entry_gemm2: String,
    fused_name: String,
    smem_a_output: String,
    smem_b_input: String,
    gmem_b_output: String,
    gmem_c_input: String,
    smem_elements: usize,
    total_rows_param: String,
    const_name: String,
}

fn parse_3phase_args(input: &str) -> Result<ThreePhaseArgs, String> {
    let mut strings = Vec::new();
    let mut rest = input.trim();
    while let Some(q1) = rest.find('"') {
        let after = &rest[q1 + 1..];
        let q2 = after.find('"').ok_or("unclosed quote")?;
        strings.push(after[..q2].to_string());
        rest = &after[q2 + 1..];
    }

    // 11 quoted strings
    if strings.len() < 11 {
        return Err(format!("expected 11 quoted strings, got {}", strings.len()));
    }

    let rest = rest.trim().trim_start_matches(',').trim();
    let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
    if parts.len() < 2 {
        return Err("expected smem_elements and CONST_NAME".to_string());
    }

    let smem_elements: usize = parts[0]
        .trim()
        .parse()
        .map_err(|e| format!("bad smem_elements: {e}"))?;
    let const_name = parts[1].trim().to_string();

    Ok(ThreePhaseArgs {
        path_norm: strings[0].clone(),
        entry_norm: strings[1].clone(),
        path_gemm1: strings[2].clone(),
        entry_gemm1: strings[3].clone(),
        path_gemm2: strings[4].clone(),
        entry_gemm2: strings[5].clone(),
        fused_name: strings[6].clone(),
        smem_a_output: strings[7].clone(),
        smem_b_input: strings[8].clone(),
        gmem_b_output: strings[9].clone(),
        gmem_c_input: strings[10].clone(),
        smem_elements,
        total_rows_param: strings
            .get(11)
            .cloned()
            .unwrap_or_else(|| "param_3".to_string()),
        const_name,
    })
}

/// Inject an activation function into a GEMM epilogue at compile time.
///
/// Extracts a single entry from a multi-entry PTX file, then injects the
/// specified activation on every f32 value before bf16 conversion or f32 store.
///
/// ```rust,ignore
/// inject_epilogue!(
///     "kernels/cutlass_gemm_sm89.ptx",
///     "GemmShapeILi64ELi128ELi64",   // entry substring
///     Gelu,                           // activation: Silu, Gelu, or Relu
///     CUTLASS_GEMM_WITH_GELU
/// );
/// ```
#[proc_macro]
pub fn inject_epilogue(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_epilogue_args(&input_str)
        .expect("inject_epilogue! expects (\"path.ptx\", \"entry\", Activation, CONST_NAME)");

    let act = fuse_epilogue::ActivationFn::from_str(&args.2).unwrap_or_else(|| {
        panic!(
            "unknown activation: '{}' (expected Silu, Gelu, or Relu)",
            args.2
        )
    });

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&args.0);
    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    let extracted = extract::extract_entry(&ptx_source, &args.1)
        .unwrap_or_else(|e| panic!("extract_entry failed: {e}"));

    let modified = fuse_epilogue::inject_activation_into_epilogue(&extracted, act)
        .or_else(|_| fuse_epilogue::inject_activation_into_f32_stores(&extracted, act))
        .unwrap_or_else(|e| panic!("{} injection failed: {e}", act.name()));

    let modified_str = modified.as_str();
    let const_name = syn::Ident::new(&args.3, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #modified_str;
    };
    output.into()
}

/// Inject SiLU activation into a CUTLASS GEMM epilogue at compile time.
///
/// Convenience wrapper around `inject_epilogue!` with SiLU hardcoded.
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

    let act = fuse_epilogue::ActivationFn::Silu;
    let modified = fuse_epilogue::inject_activation_into_epilogue(&extracted, act)
        .or_else(|_| fuse_epilogue::inject_activation_into_f32_stores(&extracted, act))
        .unwrap_or_else(|e| panic!("SiLU injection failed: {e}"));

    let modified_str = modified.as_str();
    let const_name = syn::Ident::new(&args.2, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #modified_str;
    };
    output.into()
}

/// Delete A-matrix cp.async loads from a CUTLASS GEMM.
///
/// The A-matrix data is expected to already be in SMEM (written by a prologue
/// phase). The cp.async instructions for A are deleted; B-matrix loads stay.
///
/// ```rust,ignore
/// delete_cutlass_a_loads!(
///     "kernels/cutlass_gemm_bf16_sm89.ptx",
///     "Gemm",
///     CUTLASS_GEMM_NO_A_LOADS
/// );
/// ```
#[proc_macro]
pub fn delete_cutlass_a_loads(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_extract_args(&input_str)
        .expect("delete_cutlass_a_loads! expects (\"path.ptx\", \"entry_substr\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&args.0);
    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    let extracted = extract::extract_entry(&ptx_source, &args.1)
        .unwrap_or_else(|e| panic!("extract_entry failed: {e}"));

    let result = fuse_cp_async::delete_a_matrix_loads(&extracted, "")
        .unwrap_or_else(|e| panic!("cp.async deletion failed: {e}"));

    let modified_str = result.ptx.as_str();
    let const_name = syn::Ident::new(&args.2, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #modified_str;
    };
    output.into()
}

/// Replace A-matrix cp.async loads with explicit ld.global + st.shared.
///
/// This is the passthrough prologue: same data, synchronous instructions.
/// Proves that explicit loads produce the same SMEM contents as cp.async.
///
/// ```rust,ignore
/// replace_cutlass_a_loads!(
///     "kernels/cutlass_gemm_bf16_sm89.ptx",
///     "Gemm",
///     CUTLASS_GEMM_EXPLICIT_A
/// );
/// ```
#[proc_macro]
pub fn replace_cutlass_a_loads(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let args = parse_extract_args(&input_str)
        .expect("replace_cutlass_a_loads! expects (\"path.ptx\", \"entry_substr\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_path = PathBuf::from(&manifest_dir).join(&args.0);
    let ptx_source = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));

    let extracted = extract::extract_entry(&ptx_source, &args.1)
        .unwrap_or_else(|e| panic!("extract_entry failed: {e}"));

    let modified = fuse_cp_async::replace_a_loads_with_explicit(&extracted, "")
        .unwrap_or_else(|e| panic!("cp.async replacement failed: {e}"));

    let modified_str = modified.as_str();
    let const_name = syn::Ident::new(&args.2, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #modified_str;
    };
    output.into()
}

/// Replace a CUTLASS kernel's param interface with a flat layout.
///
/// Reads the PTX and derivations JSON at compile time, rewrites all ld.param
/// instructions to use the flat layout, and emits the result as a `&str` const.
///
/// ```rust,ignore
/// replace_perimeter!("kernels/cutlass.ptx", "kernels/cutlass.derivations.json", "ferrite_gemm", FLAT_GEMM);
/// // expands to:
/// // const FLAT_GEMM: &str = "...rewritten PTX...";
/// ```
#[proc_macro]
pub fn replace_perimeter_macro(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();

    let (ptx_path, json_path, entry_name, const_name_str) = parse_perimeter_args(&input_str)
        .expect(
            "replace_perimeter! expects (\"ptx_path\", \"json_path\", \"entry_name\", CONST_NAME)",
        );

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_full = PathBuf::from(&manifest_dir).join(&ptx_path);
    let json_full = PathBuf::from(&manifest_dir).join(&json_path);

    let ptx_source = std::fs::read_to_string(&ptx_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_full.display()));
    let json_source = std::fs::read_to_string(&json_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_full.display()));

    let (rewritten, _entry) = perimeter::replace_perimeter(&ptx_source, &json_source, &entry_name)
        .unwrap_or_else(|e| panic!("perimeter replacement failed: {e}"));

    let rewritten_str = rewritten.as_str();
    let const_name = syn::Ident::new(&const_name_str, proc_macro2::Span::call_site());

    let output = quote! {
        const #const_name: &str = #rewritten_str;
    };
    output.into()
}

/// General kernel fusion: fuse two or more PTX kernels via escape perimeter analysis.
///
/// The macro is kernel-agnostic — it doesn't know what the kernels do. It only
/// sees their perimeters (global stores/loads traced to params) and rewires the
/// plumbing so bound data transits through SMEM (or registers) instead of GMEM.
///
/// ```rust,ignore
/// ptx_fusion::fuse!(
///     a = "kernels/rms_norm.ptx",
///     b = "kernels/silu_mul.ptx",
///     bind = { a.output => b.input },
///     name = "fused_norm_silu",
///     const = FUSED_PTX,
/// );
/// // expands to:
/// // const FUSED_PTX: &str = "...fused PTX...";
/// ```
#[proc_macro]
pub fn fuse(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let parsed =
        parse_general_fuse_args(&input_str).unwrap_or_else(|e| panic!("fuse! parse error: {e}"));

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    // Resolve kernel sources: either read PTX from file or mark as intrinsic
    // (name, source) where source is either PTX text or "intrinsic:NAME"
    let mut kernel_sources: Vec<(String, String)> = Vec::new();
    for (name, path) in &parsed.kernels {
        if path.starts_with("intrinsic:") {
            kernel_sources.push((name.clone(), path.clone()));
        } else {
            let full_path = PathBuf::from(&manifest_dir).join(path);
            let ptx = std::fs::read_to_string(&full_path)
                .unwrap_or_else(|e| panic!("fuse!: failed to read {}: {e}", full_path.display()));
            kernel_sources.push((name.clone(), ptx));
        }
    }

    if kernel_sources.len() != 2 {
        panic!(
            "fuse! currently supports exactly 2 kernels, got {}",
            kernel_sources.len()
        );
    }
    if parsed.bindings.len() != 1 {
        panic!(
            "fuse! currently supports exactly 1 binding, got {}",
            parsed.bindings.len()
        );
    }

    let binding = &parsed.bindings[0];
    let producer = kernel_sources
        .iter()
        .find(|(n, _)| *n == binding.producer)
        .unwrap_or_else(|| panic!("fuse!: no kernel named '{}'", binding.producer));
    let consumer = kernel_sources
        .iter()
        .find(|(n, _)| *n == binding.consumer)
        .unwrap_or_else(|| panic!("fuse!: no kernel named '{}'", binding.consumer));

    // Dispatch based on whether either kernel is an intrinsic
    let fused_ptx = if producer.1.starts_with("intrinsic:") {
        // Intrinsic producer → GEMM consumer (prologue injection)
        let intrinsic_name = &producer.1["intrinsic:".len()..];
        let gemm_ptx = &consumer.1;

        // Read perimeter JSON if provided (needed for tile_m and flat-param replacement)
        let json_source = parsed.perimeter.as_ref().map(|p| {
            let json_path = PathBuf::from(&manifest_dir).join(p);
            std::fs::read_to_string(&json_path)
                .unwrap_or_else(|e| panic!("fuse!: failed to read {}: {e}", json_path.display()))
        });

        // Get tile dims from derivations (if available) or from PTX kernel name
        let (tile_m, tile_n) = if let Some(ref json) = json_source {
            let probe = perimeter::parse_derivations(json)
                .unwrap_or_else(|e| panic!("fuse!: failed to parse derivations: {e}"));
            (probe.tile.0 as usize, probe.tile.1 as usize)
        } else {
            // Try to extract from CUTLASS kernel name
            let tm = extract_tile_m_from_ptx(gemm_ptx).unwrap_or_else(|e| panic!("fuse!: {e}"));
            let tn = extract_tile_n_from_ptx(gemm_ptx).unwrap_or_else(|e| panic!("fuse!: {e}"));
            (tm, tn)
        };

        // Build the intrinsic computation
        let computation = match intrinsic_name {
            "rms_norm" => {
                intrinsic_rms_norm::rms_norm_computation(tile_m, tile_n, &parsed.fused_name)
            }
            other => panic!("fuse!: unknown intrinsic '{other}' (available: rms_norm)"),
        };

        // Inject into GEMM prologue
        let after_fusion = fuse_general::replace_a_loads_with_inline_fn(
            gemm_ptx,
            &binding.consumer_port,
            &computation,
        )
        .unwrap_or_else(|e| panic!("fuse!: intrinsic prologue injection failed: {e}"));

        // Apply perimeter replacement if derivations JSON provided
        if let Some(ref json) = json_source {
            let (rewritten, _) =
                perimeter::replace_perimeter(&after_fusion, json, &parsed.fused_name)
                    .unwrap_or_else(|e| panic!("fuse!: perimeter replacement failed: {e}"));
            rewritten
        } else {
            after_fusion
        }
    } else if consumer.1.starts_with("intrinsic:") {
        panic!("fuse!: intrinsic as consumer not yet supported (intrinsics are producers)");
    } else {
        // Both are PTX files — use general pairwise fusion
        let result = fuse_general::fuse_two(
            &producer.1,
            &consumer.1,
            &binding.producer_port,
            &binding.consumer_port,
            &parsed.fused_name,
        )
        .unwrap_or_else(|e| panic!("fuse! fusion failed: {e}"));

        // Apply perimeter replacement if derivations JSON provided
        if let Some(ref json_path) = parsed.perimeter {
            let json_full = PathBuf::from(&manifest_dir).join(json_path);
            let json_source = std::fs::read_to_string(&json_full)
                .unwrap_or_else(|e| panic!("fuse!: failed to read {}: {e}", json_full.display()));
            let (rewritten, _) =
                perimeter::replace_perimeter(&result.ptx, &json_source, &parsed.fused_name)
                    .unwrap_or_else(|e| panic!("fuse!: perimeter replacement failed: {e}"));
            rewritten
        } else {
            result.ptx
        }
    };

    let ptx_str = fused_ptx.as_str();

    // If const name was provided, emit a const declaration (backwards compat).
    // Otherwise, emit the PTX as a bare expression.
    if !parsed.const_name.is_empty() {
        let const_name = syn::Ident::new(&parsed.const_name, proc_macro2::Span::call_site());
        let output = quote! {
            const #const_name: &str = #ptx_str;
        };
        output.into()
    } else {
        let output = quote! { #ptx_str };
        output.into()
    }
}

/// Extract tile_m from a CUTLASS GEMM PTX by parsing the mangled GemmShape in the entry name.
fn extract_tile_m_from_ptx(ptx: &str) -> Result<usize, String> {
    for line in ptx.lines() {
        let t = line.trim();
        if !t.contains(".entry") {
            continue;
        }
        if let Some(pos) = t.find("GemmShapeILi") {
            let after = &t[pos + "GemmShapeILi".len()..];
            if let Some(end) = after.find('E') {
                if let Ok(m) = after[..end].parse::<usize>() {
                    return Ok(m);
                }
            }
        }
    }
    Err("could not find GemmShape in PTX entry name".into())
}

/// Extract tile_n (second value in GemmShapeILiMELiNELiKE) from PTX.
fn extract_tile_n_from_ptx(ptx: &str) -> Result<usize, String> {
    for line in ptx.lines() {
        let t = line.trim();
        if !t.contains(".entry") {
            continue;
        }
        if let Some(pos) = t.find("GemmShapeILi") {
            let after = &t[pos + "GemmShapeILi".len()..];
            // Skip M value: find first 'E', then 'Li', then parse N
            if let Some(first_e) = after.find('E') {
                let rest = &after[first_e + 1..];
                if let Some(li) = rest.find("Li") {
                    let n_start = &rest[li + 2..];
                    if let Some(end) = n_start.find('E') {
                        if let Ok(n) = n_start[..end].parse::<usize>() {
                            return Ok(n);
                        }
                    }
                }
            }
        }
    }
    Err("could not find tile_n in GemmShape".into())
}

/// Parsed arguments from the `fuse!` macro invocation.
struct GeneralFuseArgs {
    /// (name, ptx_path_or_intrinsic) for each kernel.
    /// Path is either a file path ("kernels/foo.ptx") or an intrinsic ("intrinsic:rms_norm").
    kernels: Vec<(String, String)>,
    /// Bindings: producer.port => consumer.port
    bindings: Vec<fuse_general::ParsedBinding>,
    /// Optional derivations JSON for CUTLASS perimeter replacement (flat params).
    perimeter: Option<String>,
    /// Entry name for the fused kernel
    fused_name: String,
    /// Rust const name to emit
    const_name: String,
}

/// Parse the fuse! DSL:
/// ```text
/// fuse!(
///     a = "path/to/a.ptx",
///     b = "path/to/b.ptx",
///     bind = { a.output => b.input },
///     name = "fused_kernel",
///     const = CONST_NAME,
/// )
/// ```
fn parse_general_fuse_args(input: &str) -> Result<GeneralFuseArgs, String> {
    let input = input.trim();

    let mut kernels = Vec::new();
    let mut bindings = Vec::new();
    let mut perimeter = None;
    let mut fused_name = String::new();
    let mut const_name = String::new();

    // Split by top-level commas (respecting braces and quotes)
    let items = split_top_level(input);

    for item in &items {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        if item.starts_with("bind") {
            // bind = { a.output => b.input }
            let brace_start = item.find('{').ok_or("bind: missing '{'")?;
            let brace_end = item.rfind('}').ok_or("bind: missing '}'")?;
            let inner = item[brace_start + 1..brace_end].trim();

            // Parse bindings (comma-separated within braces)
            for binding_str in inner.split(',') {
                let binding_str = binding_str.trim();
                if binding_str.is_empty() {
                    continue;
                }
                let arrow = binding_str.find("=>").ok_or("bind: missing '=>'")?;
                let lhs = binding_str[..arrow].trim();
                let rhs = binding_str[arrow + 2..].trim();

                let (prod, prod_port) = lhs
                    .split_once('.')
                    .ok_or_else(|| format!("bind: expected 'name.port', got '{lhs}'"))?;
                let (cons, cons_port) = rhs
                    .split_once('.')
                    .ok_or_else(|| format!("bind: expected 'name.port', got '{rhs}'"))?;

                bindings.push(fuse_general::ParsedBinding {
                    producer: prod.to_string(),
                    producer_port: prod_port.to_string(),
                    consumer: cons.to_string(),
                    consumer_port: cons_port.to_string(),
                });
            }
        } else if let Some(eq_pos) = item.find('=') {
            let key = item[..eq_pos].trim();
            let val = item[eq_pos + 1..].trim();

            if key == "name" {
                fused_name = val.trim_matches('"').to_string();
            } else if key == "const" {
                const_name = val
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                    .to_string();
            } else if key == "perimeter" {
                perimeter = Some(val.trim_matches('"').to_string());
            } else {
                // Kernel: name = "path.ptx" or name = "intrinsic:rms_norm"
                let path = val.trim_matches('"').to_string();
                kernels.push((key.to_string(), path));
            }
        }
    }

    if kernels.is_empty() {
        return Err("no kernels specified".into());
    }
    if bindings.is_empty() {
        return Err("no bindings specified".into());
    }
    // Auto-generate entry name from kernel names if not provided
    if fused_name.is_empty() {
        fused_name = format!(
            "fused_{}",
            kernels
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join("_")
        );
    }

    Ok(GeneralFuseArgs {
        kernels,
        bindings,
        perimeter,
        fused_name,
        const_name,
    })
}

/// Split a string by top-level commas (not inside braces or quotes).
fn split_top_level(s: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let mut brace_depth = 0;
    let mut in_quotes = false;

    for ch in s.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            '{' if !in_quotes => {
                brace_depth += 1;
                current.push(ch);
            }
            '}' if !in_quotes => {
                brace_depth -= 1;
                current.push(ch);
            }
            ',' if !in_quotes && brace_depth == 0 => {
                items.push(current.clone());
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.trim().is_empty() {
        items.push(current);
    }
    items
}

/// Compile a fusion graph into a launchable `FeriteKernel`.
///
/// Resolves kernel names from `ferrite.toml` (generated by build.rs).
/// Runs fusion transforms at compile time. Emits a `FeriteKernel` struct
/// with PTX + launch metadata.
///
/// ```rust,ignore
/// const NORM_QKV: ptx_fusion::FeriteKernel = ptx_fusion::compile!(
///     a = intrinsic(rms_norm),
///     b = gemm_64x128x32,
///     bind = { a.output => b.param_0 },
/// );
/// ```
#[proc_macro]
pub fn compile(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();

    // Read and parse the ferrite.toml manifest
    let (toml_content, kernel_dir) =
        compile::find_manifest().unwrap_or_else(|e| panic!("compile!: {e}"));
    let manifest = compile::parse_manifest(&toml_content)
        .unwrap_or_else(|e| panic!("compile!: failed to parse ferrite.toml: {e}"));

    // Parse the DSL — same format as fuse! but kernel sources can be
    // manifest names (gemm_64x128x32) or intrinsic(rms_norm)
    let parsed = parse_compile_args(&input_str, &manifest, &kernel_dir)
        .unwrap_or_else(|e| panic!("compile! parse error: {e}"));

    // Run the same fusion logic as fuse!
    let binding = &parsed.bindings[0];
    let producer = &parsed
        .kernels
        .iter()
        .find(|(n, _)| *n == binding.producer)
        .unwrap();
    let consumer = &parsed
        .kernels
        .iter()
        .find(|(n, _)| *n == binding.consumer)
        .unwrap();

    let fused_ptx = if producer.1.source.starts_with("intrinsic:") {
        let intrinsic_name = &producer.1.source["intrinsic:".len()..];
        let gemm_ptx = &consumer.1.ptx_content;
        let tile_m = consumer.1.tile_m;
        let tile_n = consumer.1.tile_n;

        let computation = match intrinsic_name {
            "rms_norm" => intrinsic_rms_norm::rms_norm_computation(
                tile_m as usize,
                tile_n as usize,
                &parsed.fused_name,
            ),
            other => panic!("compile!: unknown intrinsic '{other}'"),
        };

        let after_fusion = fuse_general::replace_a_loads_with_inline_fn(
            gemm_ptx,
            &binding.consumer_port,
            &computation,
        )
        .unwrap_or_else(|e| panic!("compile!: intrinsic fusion failed: {e}"));

        if let Some(ref json) = consumer.1.derivations_content {
            let (rewritten, _) =
                perimeter::replace_perimeter(&after_fusion, json, &parsed.fused_name)
                    .unwrap_or_else(|e| panic!("compile!: perimeter replacement failed: {e}"));
            rewritten
        } else {
            after_fusion
        }
    } else if consumer.1.source.starts_with("intrinsic:") {
        panic!("compile!: intrinsic as consumer not yet supported");
    } else {
        let result = fuse_general::fuse_two(
            &producer.1.ptx_content,
            &consumer.1.ptx_content,
            &binding.producer_port,
            &binding.consumer_port,
            &parsed.fused_name,
        )
        .unwrap_or_else(|e| panic!("compile!: fusion failed: {e}"));

        if let Some(ref json) = consumer.1.derivations_content {
            let (rewritten, _) =
                perimeter::replace_perimeter(&result.ptx, json, &parsed.fused_name)
                    .unwrap_or_else(|e| panic!("compile!: perimeter replacement failed: {e}"));
            rewritten
        } else {
            result.ptx
        }
    };

    // Compute extra param bytes (prefix before flat GEMM params)
    let extra_param_bytes = parsed.extra_param_bytes;
    let tile_m = consumer.1.tile_m;
    let tile_n = consumer.1.tile_n;
    let threads = consumer.1.threads;
    let smem_bytes = consumer.1.smem_bytes;
    let fused_name = &parsed.fused_name;
    let ptx_str = fused_ptx.as_str();

    let output = quote! {
        ptx_fusion::FeriteKernel {
            ptx: #ptx_str,
            entry: #fused_name,
            tile_m: #tile_m,
            tile_n: #tile_n,
            threads: #threads,
            smem_bytes: #smem_bytes,
            extra_param_bytes: #extra_param_bytes,
        }
    };
    output.into()
}

/// Resolved kernel source for compile!.
struct ResolvedKernel {
    source: String,      // "intrinsic:rms_norm" or "manifest:gemm_64x128x32"
    ptx_content: String, // actual PTX text (empty for intrinsics)
    derivations_content: Option<String>,
    tile_m: u32,
    tile_n: u32,
    threads: u32,
    smem_bytes: u32,
}

struct CompileArgs {
    kernels: Vec<(String, ResolvedKernel)>,
    bindings: Vec<fuse_general::ParsedBinding>,
    fused_name: String,
    extra_param_bytes: u32,
}

fn parse_compile_args(
    input: &str,
    manifest: &compile::Manifest,
    kernel_dir: &std::path::Path,
) -> Result<CompileArgs, String> {
    let input = input.trim();
    let items = split_top_level(input);

    let mut kernels = Vec::new();
    let mut bindings = Vec::new();
    let mut fused_name = String::new();
    let mut extra_param_bytes: u32 = 0;

    for item in &items {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        if item.starts_with("bind") {
            let brace_start = item.find('{').ok_or("bind: missing '{'")?;
            let brace_end = item.rfind('}').ok_or("bind: missing '}'")?;
            let inner = item[brace_start + 1..brace_end].trim();

            for binding_str in inner.split(',') {
                let binding_str = binding_str.trim();
                if binding_str.is_empty() {
                    continue;
                }
                let arrow = binding_str.find("=>").ok_or("bind: missing '=>'")?;
                let lhs = binding_str[..arrow].trim();
                let rhs = binding_str[arrow + 2..].trim();
                let (prod, prod_port) = lhs
                    .split_once('.')
                    .ok_or_else(|| format!("bind: expected 'name.port', got '{lhs}'"))?;
                let (cons, cons_port) = rhs
                    .split_once('.')
                    .ok_or_else(|| format!("bind: expected 'name.port', got '{rhs}'"))?;

                bindings.push(fuse_general::ParsedBinding {
                    producer: prod.to_string(),
                    producer_port: prod_port.to_string(),
                    consumer: cons.to_string(),
                    consumer_port: cons_port.to_string(),
                });
            }
        } else if let Some(eq_pos) = item.find('=') {
            let key = item[..eq_pos].trim();
            let val = item[eq_pos + 1..].trim();

            if key == "name" {
                fused_name = val.trim_matches('"').to_string();
            } else if val.starts_with("intrinsic(") || val.starts_with("intrinsic (") {
                // intrinsic(rms_norm)
                let paren_start = val.find('(').unwrap();
                let paren_end = val.rfind(')').ok_or("intrinsic: missing ')'")?;
                let intrinsic_name = val[paren_start + 1..paren_end].trim();

                // rms_norm adds 5 params: weight(u64), eps(f32), hidden(u32), a_ptr(u64), a_stride(u64)
                // = 8 + 4 + 4 + 8 + 8 = 32 bytes
                let extra_bytes = match intrinsic_name {
                    "rms_norm" => 40u32, // 8+4+4+8+8+4 = 36, padded to 40 for align 8
                    _ => 0,
                };
                extra_param_bytes += extra_bytes;

                kernels.push((
                    key.to_string(),
                    ResolvedKernel {
                        source: format!("intrinsic:{intrinsic_name}"),
                        ptx_content: String::new(),
                        derivations_content: None,
                        tile_m: 0,
                        tile_n: 0,
                        threads: 0,
                        smem_bytes: 0,
                    },
                ));
            } else {
                // Manifest kernel name (e.g., gemm_64x128x32)
                let kernel_name = val.trim();
                let mk = manifest.kernels.get(kernel_name).ok_or_else(|| {
                    format!("compile!: unknown kernel '{kernel_name}' (not in ferrite.toml)")
                })?;

                let ptx_path = kernel_dir.join(&mk.ptx_path);
                let ptx_content = std::fs::read_to_string(&ptx_path)
                    .map_err(|e| format!("compile!: failed to read {}: {e}", ptx_path.display()))?;

                let derivations_content = if !mk.derivations_path.is_empty() {
                    let json_path = kernel_dir.join(&mk.derivations_path);
                    Some(std::fs::read_to_string(&json_path).map_err(|e| {
                        format!("compile!: failed to read {}: {e}", json_path.display())
                    })?)
                } else {
                    None
                };

                kernels.push((
                    key.to_string(),
                    ResolvedKernel {
                        source: format!("manifest:{kernel_name}"),
                        ptx_content,
                        derivations_content,
                        tile_m: mk.tile.0,
                        tile_n: mk.tile.1,
                        threads: mk.threads,
                        smem_bytes: mk.smem,
                    },
                ));
            }
        }
    }

    if kernels.is_empty() {
        return Err("no kernels specified".into());
    }
    if bindings.is_empty() {
        return Err("no bindings specified".into());
    }

    if fused_name.is_empty() {
        fused_name = format!(
            "fused_{}",
            kernels
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join("_")
        );
    }

    Ok(CompileArgs {
        kernels,
        bindings,
        fused_name,
        extra_param_bytes,
    })
}

/// Apply identity prologue (passthrough at A-load sites) then flat-param perimeter
/// replacement. Produces a flat-param kernel with cp.async replaced by explicit
/// ld→unpack→repack→st. For GPU correctness testing.
///
/// ```rust,ignore
/// prologue_identity_flat!(
///     "kernels/cutlass.ptx",
///     "kernels/cutlass.derivations.json",
///     "entry_name",
///     CONST_NAME
/// );
/// ```
#[proc_macro]
pub fn prologue_identity_flat(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let (ptx_path, json_path, entry_name, const_name_str) = parse_perimeter_args(&input_str)
        .expect("prologue_identity_flat! expects (\"ptx\", \"json\", \"entry\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_full = PathBuf::from(&manifest_dir).join(&ptx_path);
    let json_full = PathBuf::from(&manifest_dir).join(&json_path);

    let ptx_source = std::fs::read_to_string(&ptx_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_full.display()));
    let json_source = std::fs::read_to_string(&json_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_full.display()));

    // Step 1: Apply identity prologue on original CUTLASS PTX
    let identity = fuse_general::PointwiseComputation {
        instructions: vec![],
        param_loads: vec![],
        prologue: vec![],
        extra_reg_decls: vec![],
        extra_params: vec![],
        per_site: vec![],
        entry_name: None,
        scratch_f32_count: 0,
        scratch_b32_count: 0,
    };
    let after_prologue =
        fuse_general::replace_a_loads_with_inline_fn(&ptx_source, "param_0", &identity)
            .unwrap_or_else(|e| panic!("prologue_identity failed: {e}"));

    // Step 2: Apply perimeter replacement (flat-param) on the prologue result
    let (rewritten, _) = perimeter::replace_perimeter(&after_prologue, &json_source, &entry_name)
        .unwrap_or_else(|e| panic!("perimeter replacement after prologue failed: {e}"));

    let const_name = syn::Ident::new(&const_name_str, proc_macro2::Span::call_site());
    let rewritten_str = rewritten.as_str();
    let output = quote! {
        const #const_name: &str = #rewritten_str;
    };
    output.into()
}

/// Apply rms_norm intrinsic prologue then flat-param perimeter replacement.
/// Produces a flat-param kernel with rms_norm fused into the GEMM's A-load path.
///
/// Extra params: _ferrite_rms_weight (u64), _ferrite_rms_epsilon (f32),
/// _ferrite_rms_hidden (u32) — prepended before the flat ferrite_params[88].
///
/// ```rust,ignore
/// fuse_rms_norm_gemm_flat!(
///     "kernels/cutlass.ptx",
///     "kernels/cutlass.derivations.json",
///     "fused_norm_gemm",
///     CONST_NAME
/// );
/// ```
#[proc_macro]
pub fn fuse_rms_norm_gemm_flat(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let (ptx_path, json_path, entry_name, const_name_str) = parse_perimeter_args(&input_str)
        .expect("fuse_rms_norm_gemm_flat! expects (\"ptx\", \"json\", \"entry\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_full = PathBuf::from(&manifest_dir).join(&ptx_path);
    let json_full = PathBuf::from(&manifest_dir).join(&json_path);

    let ptx_source = std::fs::read_to_string(&ptx_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_full.display()));
    let json_source = std::fs::read_to_string(&json_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_full.display()));

    // Extract tile_m from derivations JSON
    let probe = perimeter::parse_derivations(&json_source)
        .unwrap_or_else(|e| panic!("failed to parse derivations: {e}"));
    let tile_m = probe.tile.0 as usize;
    let tile_n = probe.tile.1 as usize;

    // Step 1: Build rms_norm computation (first-class intrinsic — no hardcoded registers)
    let computation = intrinsic_rms_norm::rms_norm_computation(tile_m, tile_n, &entry_name);

    // Step 2: Inject rms_norm into GEMM via replace_a_loads_with_inline_fn
    let after_rms =
        fuse_general::replace_a_loads_with_inline_fn(&ptx_source, "param_0", &computation)
            .unwrap_or_else(|e| panic!("rms_norm prologue injection failed: {e}"));

    // Step 3: Apply perimeter replacement (flat params) on the result
    let (rewritten, _) = perimeter::replace_perimeter(&after_rms, &json_source, &entry_name)
        .unwrap_or_else(|e| panic!("perimeter replacement after rms_norm failed: {e}"));

    let const_name = syn::Ident::new(&const_name_str, proc_macro2::Span::call_site());
    let rewritten_str = rewritten.as_str();
    let output = quote! {
        const #const_name: &str = #rewritten_str;
    };
    output.into()
}

/// Apply scale-by-2.0 prologue then flat-param perimeter replacement.
/// Each A-matrix element is multiplied by 2.0 inline at the cp.async site.
/// For GPU correctness testing: GEMM(A*2, B) should equal 2 * GEMM(A, B).
#[proc_macro]
pub fn prologue_scale2_flat(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let (ptx_path, json_path, entry_name, const_name_str) = parse_perimeter_args(&input_str)
        .expect("prologue_scale2_flat! expects (\"ptx\", \"json\", \"entry\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_full = PathBuf::from(&manifest_dir).join(&ptx_path);
    let json_full = PathBuf::from(&manifest_dir).join(&json_path);

    let ptx_source = std::fs::read_to_string(&ptx_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_full.display()));
    let json_source = std::fs::read_to_string(&json_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_full.display()));

    // Scale by 2.0: mul.f32 {INPUT}, {INPUT}, 0f40000000 (IEEE 754 for 2.0)
    let scale2 = fuse_general::PointwiseComputation {
        instructions: vec!["mul.f32 {INPUT}, {INPUT}, 0f40000000;".to_string()],
        param_loads: vec![],
        prologue: vec![],
        extra_reg_decls: vec![],
        extra_params: vec![],
        per_site: vec![],
        entry_name: None,
        scratch_f32_count: 0,
        scratch_b32_count: 0,
    };

    let after_prologue =
        fuse_general::replace_a_loads_with_inline_fn(&ptx_source, "param_0", &scale2)
            .unwrap_or_else(|e| panic!("prologue_scale2 failed: {e}"));

    let (rewritten, _) = perimeter::replace_perimeter(&after_prologue, &json_source, &entry_name)
        .unwrap_or_else(|e| panic!("perimeter replacement after prologue failed: {e}"));

    let const_name = syn::Ident::new(&const_name_str, proc_macro2::Span::call_site());
    let rewritten_str = rewritten.as_str();
    let output = quote! {
        const #const_name: &str = #rewritten_str;
    };
    output.into()
}

/// Apply `replace_a_loads_with_inline_fn` with an identity (passthrough) function.
/// Used for GPU correctness testing of the prologue injection mechanism.
///
/// ```rust,ignore
/// prologue_identity!("kernels/cutlass.ptx", "param_0", "test_entry", CONST_NAME);
/// ```
#[proc_macro]
pub fn prologue_identity(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();
    let (ptx_path, hint, entry, const_name_str) = parse_perimeter_args(&input_str)
        .expect("prologue_identity! expects (\"ptx\", \"hint\", \"entry\", CONST_NAME)");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let ptx_full = PathBuf::from(&manifest_dir).join(&ptx_path);
    let ptx_source = std::fs::read_to_string(&ptx_full)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_full.display()));

    let identity = fuse_general::PointwiseComputation {
        instructions: vec![],
        param_loads: vec![],
        prologue: vec![],
        extra_reg_decls: vec![],
        extra_params: vec![],
        per_site: vec![],
        entry_name: None,
        scratch_f32_count: 0,
        scratch_b32_count: 0,
    };

    let mut result = fuse_general::replace_a_loads_with_inline_fn(&ptx_source, &hint, &identity)
        .unwrap_or_else(|e| panic!("prologue_identity failed: {e}"));

    // Rename entry point
    if let Some(start) = result.find(".entry") {
        if let Some(paren) = result[start..].find('(') {
            let entry_start = start + 7; // ".entry "
            let entry_end = start + paren;
            let old_entry = result[entry_start..entry_end].trim().to_string();
            result = result.replacen(&old_entry, &entry, 1);
        }
    }

    let const_name = syn::Ident::new(&const_name_str, proc_macro2::Span::call_site());
    let result_str = result.as_str();
    let output = quote! {
        const #const_name: &str = #result_str;
    };
    output.into()
}

fn parse_perimeter_args(input: &str) -> Result<(String, String, String, String), String> {
    let input = input.trim();
    let mut strings = Vec::new();
    let mut rest = input;

    for _ in 0..3 {
        let start = rest.find('"').ok_or("expected opening quote")?;
        let after = &rest[start + 1..];
        let end = after.find('"').ok_or("expected closing quote")?;
        strings.push(after[..end].to_string());
        rest = &after[end + 1..];
    }

    let rest = rest.trim().trim_start_matches(',').trim();
    let const_name = rest
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .ok_or("expected const name")?
        .to_string();

    if strings.len() != 3 || const_name.is_empty() {
        return Err("expected 3 string args + const name".into());
    }

    Ok((
        strings[0].clone(),
        strings[1].clone(),
        strings[2].clone(),
        const_name,
    ))
}
