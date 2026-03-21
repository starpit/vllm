use proc_macro2::TokenStream;
use quote::quote;
use syn::ItemFn;

use crate::ops::{OpGraph, OpKind};
use crate::parse::FuseAttr;
use crate::strategy::{FusionPlan, Stage};

/// Generate the complete output TokenStream for a `#[fuse]` function.
///
/// Dispatches based on the number of stages and the composition of each stage.
/// Single-stage plans produce one kernel launch; multi-stage plans produce
/// multiple launches with intermediates through global memory.
pub fn generate(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    graph: &OpGraph,
    plan: &FusionPlan,
) -> syn::Result<TokenStream> {
    if plan.stages.is_empty() {
        return Err(syn::Error::new_spanned(
            &input_fn.sig.ident,
            "Ferrite fusion strategy produced no stages. \
             The function body must contain at least one GEMM operation.",
        ));
    }

    // For now, we only support single-stage plans at the codegen level.
    // Multi-stage support (multiple kernel launches) is the next step.
    if plan.stages.len() != 1 {
        return Err(syn::Error::new_spanned(
            &input_fn.sig.ident,
            format!(
                "Ferrite fusion strategy produced {} stages. \
                 Multi-stage codegen (multiple kernel launches) is not yet implemented. \
                 Stages: {:?}",
                plan.stages.len(),
                plan.stages
                    .iter()
                    .map(|s| describe_stage(s, graph))
                    .collect::<Vec<_>>(),
            ),
        ));
    }

    generate_single_stage(attr, input_fn, graph, &plan.stages[0])
}

/// Generate code for a single-stage fusion plan.
fn generate_single_stage(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    graph: &OpGraph,
    stage: &Stage,
) -> syn::Result<TokenStream> {
    // Classify by prologue/epilogue combination
    let prologue_kind = stage.prologue_transform.map(|id| graph.nodes[id].kind);
    let epilogue_kind = stage.epilogue_transform.map(|id| graph.nodes[id].kind);

    match (prologue_kind, epilogue_kind) {
        (Some(OpKind::RmsNorm), Some(OpKind::Silu)) => {
            // Transform(RmsNorm) + GEMM + Epilogue(SiLU) — the full megakernel
            generate_rmsnorm_gemm_silu(
                attr,
                input_fn,
                graph,
                stage.prologue_transform.unwrap(),
                stage.gemm,
                stage.epilogue_transform.unwrap(),
            )
        }
        (Some(OpKind::RmsNorm), None) => {
            // Transform(RmsNorm) + GEMM — no epilogue
            generate_rmsnorm_gemm(
                attr,
                input_fn,
                graph,
                stage.prologue_transform.unwrap(),
                stage.gemm,
            )
        }
        (None, Some(OpKind::Silu)) => {
            // GEMM + Epilogue(SiLU)
            generate_gemm_silu(
                attr,
                input_fn,
                graph,
                stage.gemm,
                stage.epilogue_transform.unwrap(),
            )
        }
        (None, None) => {
            // Standalone GEMM
            generate_standalone_gemm(attr, input_fn, graph, stage.gemm)
        }
        (prologue, epilogue) => {
            // Future: when new OpKinds are added, they get routed here until
            // their codegen is implemented. The strategy engine already handles
            // them — only codegen needs updating.
            Err(syn::Error::new_spanned(
                &input_fn.sig.ident,
                format!(
                    "Unsupported stage composition: prologue={:?}, epilogue={:?}. \
                     Add codegen support for this combination.",
                    prologue.map(|k| k.name()),
                    epilogue.map(|k| k.name()),
                ),
            ))
        }
    }
}

/// Generate code for the fused RmsNorm -> GEMM -> SiLU megakernel.
///
/// This is THE key codegen path — produces ONE kernel, ONE launch.
fn generate_rmsnorm_gemm_silu(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    _graph: &OpGraph,
    _norm_id: usize,
    _gemm_id: usize,
    _silu_id: usize,
) -> syn::Result<TokenStream> {
    let arch = &attr.arch;
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_args = &input_fn.sig.inputs;
    let fn_attrs = &input_fn.attrs;

    // Generate PTX at compile time using 128x128 tiles for the fused megakernel.
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
    _norm_id: usize,
    _gemm_id: usize,
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
    _gemm_id: usize,
    _silu_id: usize,
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
    _gemm_id: usize,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "Standalone GEMM not yet implemented via proc macro. Use ferrite_ptx::gemm::build_gemm() directly.",
    ))
}

/// Human-readable description of a stage for error messages.
fn describe_stage(stage: &Stage, graph: &OpGraph) -> String {
    let mut parts = Vec::new();
    if let Some(p) = stage.prologue_transform {
        parts.push(format!("Transform({})", graph.nodes[p].kind.name()));
    }
    parts.push("GEMM".to_string());
    if let Some(e) = stage.epilogue_transform {
        parts.push(format!("Epilogue({})", graph.nodes[e].kind.name()));
    }
    parts.join(" -> ")
}
