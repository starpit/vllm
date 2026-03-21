use proc_macro2::TokenStream;
use quote::quote;
use syn::ItemFn;

use crate::ops::OpGraph;
use crate::parse::FuseAttr;
use crate::strategy::{FusionPlan, KernelKind};

/// Generate the complete output TokenStream for a `#[fuse]` function.
///
/// For the megakernel case (RmsNormGemmSilu), this:
/// 1. Calls `ferrite_ptx::fused::build_fused_rmsnorm_gemm_silu()` at compile time
/// 2. Embeds the PTX as a `const &str`
/// 3. Emits a `ferrite_runtime::JitKernel` with lazy compilation
/// 4. Wraps the original function signature to launch the kernel
pub fn generate(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    graph: &OpGraph,
    plan: &FusionPlan,
) -> syn::Result<TokenStream> {
    // Validate: we expect exactly ONE kernel in the plan for the megakernel case.
    // If we get multiple, that's a strategy failure — we refuse to emit separate launches.
    if plan.kernels.len() != 1 {
        return Err(syn::Error::new_spanned(
            &input_fn.sig.ident,
            format!(
                "Ferrite fusion strategy produced {} kernels instead of 1. \
                 The entire point of Ferrite is the megakernel — ONE kernel, ONE launch. \
                 Found: {:?}",
                plan.kernels.len(),
                plan.kernels
                    .iter()
                    .map(|k| match k {
                        KernelKind::RmsNorm { .. } => "RmsNorm",
                        KernelKind::Gemm { .. } => "Gemm",
                        KernelKind::Silu { .. } => "Silu",
                        KernelKind::GemmSilu { .. } => "GemmSilu",
                        KernelKind::RmsNormGemmSilu { .. } => "RmsNormGemmSilu",
                        KernelKind::RmsNormGemm { .. } => "RmsNormGemm",
                    })
                    .collect::<Vec<_>>(),
            ),
        ));
    }

    match &plan.kernels[0] {
        KernelKind::RmsNormGemmSilu {
            norm_idx,
            gemm_idx,
            silu_idx,
        } => generate_rmsnorm_gemm_silu(attr, input_fn, graph, *norm_idx, *gemm_idx, *silu_idx),
        KernelKind::RmsNormGemm { norm_idx, gemm_idx } => {
            generate_rmsnorm_gemm(attr, input_fn, graph, *norm_idx, *gemm_idx)
        }
        KernelKind::GemmSilu { gemm_idx, silu_idx } => {
            generate_gemm_silu(attr, input_fn, graph, *gemm_idx, *silu_idx)
        }
        KernelKind::Gemm { node_idx } => generate_standalone_gemm(attr, input_fn, graph, *node_idx),
        other => Err(syn::Error::new_spanned(
            &input_fn.sig.ident,
            format!(
                "Unsupported standalone kernel kind: {:?}",
                match other {
                    KernelKind::RmsNorm { .. } => "RmsNorm",
                    KernelKind::Silu { .. } => "Silu",
                    _ => "Unknown",
                }
            ),
        )),
    }
}

/// Generate code for the fused RmsNorm -> GEMM -> SiLU megakernel.
///
/// This is THE key codegen path — produces ONE kernel, ONE launch.
fn generate_rmsnorm_gemm_silu(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    _graph: &OpGraph,
    norm_idx: usize,
    gemm_idx: usize,
    silu_idx: usize,
) -> syn::Result<TokenStream> {
    let arch = &attr.arch;
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_args = &input_fn.sig.inputs;
    let fn_attrs = &input_fn.attrs;
    let _ = (norm_idx, gemm_idx, silu_idx); // used for graph analysis

    // Generate PTX at compile time using 128×128 tiles for the fused megakernel.
    let config = ferrite_ptx::config::GemmConfig {
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 64,
        wn: 64,
        mma_m: 16,
        mma_n: 8,
        mma_k: 16,
        num_stages: 2,
        sm_arch: arch.clone(),
    };

    // We need the hidden_size at compile time for the norm reduction loop.
    // For now, use 4096 as the default (LLaMA-7B hidden size).
    // TODO: make this a parameter in the fuse attribute.
    let hidden_size: u32 = 4096;

    let ptx_string = ferrite_ptx::fused::build_fused_pipeline(&config, hidden_size);

    // Shared memory: GEMM tiles + norm scratch (32) + norm factors (BM*4) + gamma preload (hidden*2)
    let smem_bytes = (config.smem_total() + 32 + config.bm * 4 + hidden_size * 2) as u32;
    let threads = config.threads() as u32;

    // Emit Rust code that:
    // 1. Stores the PTX as a const string
    // 2. Creates a static JitKernel
    // 3. Wraps the original function to launch it
    let kernel_name_str = "fused_rmsnorm_gemm_silu";
    let static_name =
        quote::format_ident!("__FERRITE_KERNEL_{}", fn_name.to_string().to_uppercase());

    Ok(quote! {
        // The compile-time-generated PTX for the fused megakernel.
        const _: () = {
            // Validate the PTX was generated (compile-time check).
            const PTX: &str = #ptx_string;
        };

        static #static_name: ferrite_runtime::JitKernel =
            ferrite_runtime::JitKernel::new(#ptx_string, #kernel_name_str);

        #(#fn_attrs)*
        #fn_vis fn #fn_name(#fn_args) {
            // Grid: (out_features / BN, batch / BM, 1)
            // The caller must pass correctly-sized tensors.
            //
            // Parameters to the PTX kernel:
            //   param_input:  ptr to input tensor (batch x hidden_size, f16)
            //   param_wnorm:  ptr to RMSNorm weight (hidden_size, f16)
            //   param_wgemm:  ptr to GEMM weight matrix (hidden_size x out_features, f16)
            //   param_output: ptr to output tensor (batch x out_features, f32)
            //   param_N:      out_features (u32)
            //   param_K:      hidden_size (u32)
            let grid_x = (out_feat + 127) / 128;
            let grid_y = (batch + 127) / 128;
            let grid = (grid_x, grid_y, 1u32);
            let block = (#threads, 1u32, 1u32);
            let shared_mem = #smem_bytes as u32;

            let mut p_input = input as u64;
            let mut p_wnorm = weight_norm as u64;
            let mut p_wgemm = weight_gemm as u64;
            let mut p_output = output as u64;
            let mut p_n = out_feat;
            let mut p_k = hidden;

            let args: &[*mut std::ffi::c_void] = &[
                &mut p_input as *mut u64 as *mut std::ffi::c_void,
                &mut p_wnorm as *mut u64 as *mut std::ffi::c_void,
                &mut p_wgemm as *mut u64 as *mut std::ffi::c_void,
                &mut p_output as *mut u64 as *mut std::ffi::c_void,
                &mut p_n as *mut u32 as *mut std::ffi::c_void,
                &mut p_k as *mut u32 as *mut std::ffi::c_void,
            ];

            unsafe {
                #static_name.launch(grid, block, shared_mem, args)
                    .expect("fused kernel launch failed");
            }
        }
    })
}

/// Generate code for fused RmsNorm -> GEMM (without SiLU).
fn generate_rmsnorm_gemm(
    _attr: &FuseAttr,
    input_fn: &syn::ItemFn,
    _graph: &OpGraph,
    _norm_idx: usize,
    _gemm_idx: usize,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "RmsNorm -> GEMM fusion without SiLU not yet implemented. Add silu() to your pipeline.",
    ))
}

/// Generate code for fused GEMM -> SiLU.
fn generate_gemm_silu(
    _attr: &FuseAttr,
    input_fn: &syn::ItemFn,
    _graph: &OpGraph,
    _gemm_idx: usize,
    _silu_idx: usize,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "Standalone GEMM+SiLU fusion not yet implemented. Use the full rmsnorm->gemm->silu pipeline.",
    ))
}

/// Generate code for standalone GEMM.
fn generate_standalone_gemm(
    _attr: &FuseAttr,
    input_fn: &syn::ItemFn,
    _graph: &OpGraph,
    _node_idx: usize,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "Standalone GEMM not yet implemented via proc macro. Use ferrite_ptx::gemm::build_gemm() directly.",
    ))
}
