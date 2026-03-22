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

    match plan.stages.len() {
        1 => generate_single_stage(attr, input_fn, graph, &plan.stages[0]),
        2 => generate_two_stage(attr, input_fn, graph, &plan.stages[0], &plan.stages[1]),
        n => Err(syn::Error::new_spanned(
            &input_fn.sig.ident,
            format!(
                "Ferrite fusion strategy produced {} stages. \
                 Only 1-stage and 2-stage plans are currently supported. \
                 Stages: {:?}",
                n,
                plan.stages
                    .iter()
                    .map(|s| describe_stage(s, graph))
                    .collect::<Vec<_>>(),
            ),
        )),
    }
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
        (Some(OpKind::RmsNorm), Some(OpKind::Gelu)) => {
            // Transform(RmsNorm) + GEMM + Epilogue(GELU)
            generate_rmsnorm_gemm_gelu(
                attr,
                input_fn,
                graph,
                stage.prologue_transform.unwrap(),
                stage.gemm,
                stage.epilogue_transform.unwrap(),
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
        (None, Some(OpKind::Gelu)) => {
            // GEMM + Epilogue(GELU)
            generate_gemm_gelu(
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

/// Generate code for a 2-stage fusion plan.
///
/// This handles patterns like: rmsnorm -> gemm -> silu -> gemm (MLP block)
///
/// Architecture: 3 kernel launches in sequence
///   1. Stage 1: Fused kernel (e.g. RmsNorm -> GEMM -> SiLU) -> f32 output
///   2. Conversion: f32 -> f16 element-wise kernel
///   3. Stage 2: Standalone GEMM (f16 input -> f32 output)
///
/// Intermediate buffers are allocated/freed around the launches.
fn generate_two_stage(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    graph: &OpGraph,
    stage1: &Stage,
    stage2: &Stage,
) -> syn::Result<TokenStream> {
    let s1_prologue = stage1.prologue_transform.map(|id| graph.nodes[id].kind);
    let s1_epilogue = stage1.epilogue_transform.map(|id| graph.nodes[id].kind);
    let s2_prologue = stage2.prologue_transform.map(|id| graph.nodes[id].kind);
    let s2_epilogue = stage2.epilogue_transform.map(|id| graph.nodes[id].kind);

    // Currently supported 2-stage pattern: RmsNorm+GEMM+SiLU -> GEMM
    match ((s1_prologue, s1_epilogue), (s2_prologue, s2_epilogue)) {
        ((Some(OpKind::RmsNorm), Some(OpKind::Silu)), (None, None)) => {
            generate_mlp_block(attr, input_fn, graph, stage1, stage2)
        }
        ((Some(OpKind::RmsNorm), Some(OpKind::Gelu)), (None, None)) => {
            // RmsNorm+GEMM+GELU -> GEMM
            generate_rmsnorm_gemm_gelu_gemm_block(attr, input_fn, graph, stage1, stage2)
        }
        ((None, Some(OpKind::Silu)), (None, None)) => {
            // GEMM+SiLU -> GEMM (no rmsnorm)
            generate_gemm_silu_gemm_block(attr, input_fn, graph, stage1, stage2)
        }
        ((None, Some(OpKind::Gelu)), (None, None)) => {
            // GEMM+GELU -> GEMM (no rmsnorm)
            generate_gemm_gelu_gemm_block(attr, input_fn, graph, stage1, stage2)
        }
        ((Some(OpKind::RmsNorm), Some(OpKind::SiluMul)), (None, None)) => {
            // LLaMA MLP: RmsNorm+dual GEMM(gate,up)+SiluMul -> GEMM(down)
            // Uses CUTLASS dual_gemm PTX (59.8 TFLOPS, 1 launch for dual GEMM)
            generate_dual_gemm_mlp_block(attr, input_fn, graph, stage1, stage2)
        }
        ((None, Some(OpKind::SiluMul)), (None, None)) => {
            // dual GEMM(gate,up)+SiluMul -> GEMM(down) (no rmsnorm)
            generate_dual_gemm_mlp_block(attr, input_fn, graph, stage1, stage2)
        }
        ((None, None), (None, None)) => {
            // GEMM -> GEMM chain
            generate_gemm_gemm_chain(attr, input_fn, graph, stage1, stage2)
        }
        _ => Err(syn::Error::new_spanned(
            &input_fn.sig.ident,
            format!(
                "Unsupported 2-stage composition: \
                 Stage1=[prologue={:?}, epilogue={:?}], \
                 Stage2=[prologue={:?}, epilogue={:?}]. \
                 Currently supported: RmsNorm+GEMM+SiLU -> GEMM (MLP block).",
                s1_prologue.map(|k| k.name()),
                s1_epilogue.map(|k| k.name()),
                s2_prologue.map(|k| k.name()),
                s2_epilogue.map(|k| k.name()),
            ),
        )),
    }
}

/// Generate code for the full MLP block: RmsNorm -> GEMM -> SiLU -> GEMM
///
/// This is the primary 2-stage codegen path. Produces:
///   - 3 static JitKernel instances (stage1, cvt, stage2)
///   - 3 const PTX strings (compile-time generated)
///   - Intermediate buffer allocation + 3 sequential kernel launches
fn generate_mlp_block(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    _graph: &OpGraph,
    _stage1: &Stage,
    _stage2: &Stage,
) -> syn::Result<TokenStream> {
    let arch = &attr.arch;
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_args = &input_fn.sig.inputs;
    let fn_attrs = &input_fn.attrs;

    // GEMM config for both stages (128x128 tiles)
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

    let hidden_size: u32 = 4096;

    // Stage 1: fused RmsNorm -> GEMM -> SiLU
    let ptx_stage1 = ferrite_ptx::fused::build_fused_pipeline(&config, hidden_size);
    let smem_stage1 = (config.smem_total() + 32 + config.bm * 4 + hidden_size * 2) as u32;
    let threads = config.threads() as u32;

    // Conversion kernel: f32 -> f16
    let ptx_cvt = ferrite_ptx::convert::build_cvt_f32_to_f16_kernel(arch);

    // Stage 2: standalone GEMM
    let ptx_stage2 = ferrite_ptx::gemm::build_gemm_pipeline(&config);
    let smem_stage2 = config.smem_total() as u32;

    let kernel1_name = "fused_rmsnorm_gemm_silu";
    let kernel_cvt_name = "cvt_f32_to_f16";
    let kernel2_name = "triton_style_gemm";

    let static_k1 =
        quote::format_ident!("__FERRITE_KERNEL1_{}", fn_name.to_string().to_uppercase());
    let static_cvt = quote::format_ident!(
        "__FERRITE_KERNEL_CVT_{}",
        fn_name.to_string().to_uppercase()
    );
    let static_k2 =
        quote::format_ident!("__FERRITE_KERNEL2_{}", fn_name.to_string().to_uppercase());

    Ok(quote! {
        // Compile-time PTX validation
        const _: () = {
            const PTX1: &str = #ptx_stage1;
            const PTX_CVT: &str = #ptx_cvt;
            const PTX2: &str = #ptx_stage2;
        };

        // Lazy JIT-compiled kernel handles
        static #static_k1: ferrite_runtime::JitKernel =
            ferrite_runtime::JitKernel::new(#ptx_stage1, #kernel1_name);
        static #static_cvt: ferrite_runtime::JitKernel =
            ferrite_runtime::JitKernel::new(#ptx_cvt, #kernel_cvt_name);
        static #static_k2: ferrite_runtime::JitKernel =
            ferrite_runtime::JitKernel::new(#ptx_stage2, #kernel2_name);

        #(#fn_attrs)*
        #fn_vis fn #fn_name(#fn_args) {
            use ferrite_runtime::cuda;
            use ferrite_runtime::cuda_sys;

            let stream = cuda::stream::null();

            // Intermediate dimensions:
            // Stage 1 output: [batch, inter] f32 (GEMM1 accumulator output)
            // After conversion: [batch, inter] f16 (input to GEMM2)
            let inter_elems = (batch as usize) * (inter as usize);
            let inter_f32_bytes = inter_elems * 4; // f32
            let inter_f16_bytes = inter_elems * 2; // f16

            // Allocate intermediate buffers
            let d_inter_f32 = unsafe {
                cuda::malloc_async(stream, inter_f32_bytes)
                    .expect("failed to allocate intermediate f32 buffer")
            };
            let d_inter_f16 = unsafe {
                cuda::malloc_async(stream, inter_f16_bytes)
                    .expect("failed to allocate intermediate f16 buffer")
            };

            // ── Stage 1: RmsNorm -> GEMM -> SiLU ──
            // Grid: (inter / 128, batch / 128, 1) — output is [batch, inter]
            {
                let grid_x = (inter + 127) / 128;
                let grid_y = (batch + 127) / 128;
                let grid = (grid_x, grid_y, 1u32);
                let block = (#threads, 1u32, 1u32);
                let shared_mem = #smem_stage1 as u32;

                let mut p_input = input as u64;
                let mut p_wnorm = w_norm as u64;
                let mut p_wgate = w_gate as u64;
                let mut p_output = d_inter_f32 as u64;
                let mut p_n = inter;
                let mut p_k = hidden;

                let args: &[*mut std::ffi::c_void] = &[
                    &mut p_input as *mut u64 as *mut std::ffi::c_void,
                    &mut p_wnorm as *mut u64 as *mut std::ffi::c_void,
                    &mut p_wgate as *mut u64 as *mut std::ffi::c_void,
                    &mut p_output as *mut u64 as *mut std::ffi::c_void,
                    &mut p_n as *mut u32 as *mut std::ffi::c_void,
                    &mut p_k as *mut u32 as *mut std::ffi::c_void,
                ];

                unsafe {
                    #static_k1.launch(grid, block, shared_mem, args)
                        .expect("stage 1 (fused rmsnorm+gemm+silu) launch failed");
                }
            }

            // ── Conversion: f32 -> f16 ──
            {
                let total_elems = batch * inter;
                // Each thread converts 4 elements, 256 threads per block
                let threads_needed = (total_elems + 3) / 4;
                let cvt_grid = ((threads_needed + 255) / 256, 1u32, 1u32);
                let cvt_block = (256u32, 1u32, 1u32);

                let mut p_in = d_inter_f32 as u64;
                let mut p_out = d_inter_f16 as u64;
                let mut p_n = total_elems;

                let args: &[*mut std::ffi::c_void] = &[
                    &mut p_in as *mut u64 as *mut std::ffi::c_void,
                    &mut p_out as *mut u64 as *mut std::ffi::c_void,
                    &mut p_n as *mut u32 as *mut std::ffi::c_void,
                ];

                unsafe {
                    #static_cvt.launch(cvt_grid, cvt_block, 0, args)
                        .expect("conversion (f32->f16) launch failed");
                }
            }

            // ── Stage 2: Standalone GEMM ──
            // A = d_inter_f16 [batch, inter], B = w_down [inter, out], C = output [batch, out]
            // Kernel params: (A, B, C, M, N, K)
            {
                let grid_x = (out + 127) / 128;
                let grid_y = (batch + 127) / 128;
                let grid = (grid_x, grid_y, 1u32);
                let block = (#threads, 1u32, 1u32);
                let shared_mem = #smem_stage2 as u32;

                let mut p_a = d_inter_f16 as u64;
                let mut p_b = w_down as u64;
                let mut p_c = output as u64;
                let mut p_m = batch;
                let mut p_n = out;
                let mut p_k = inter;

                let args: &[*mut std::ffi::c_void] = &[
                    &mut p_a as *mut u64 as *mut std::ffi::c_void,
                    &mut p_b as *mut u64 as *mut std::ffi::c_void,
                    &mut p_c as *mut u64 as *mut std::ffi::c_void,
                    &mut p_m as *mut u32 as *mut std::ffi::c_void,
                    &mut p_n as *mut u32 as *mut std::ffi::c_void,
                    &mut p_k as *mut u32 as *mut std::ffi::c_void,
                ];

                unsafe {
                    #static_k2.launch(grid, block, shared_mem, args)
                        .expect("stage 2 (standalone gemm) launch failed");
                }
            }

            // Free intermediate buffers
            unsafe {
                cuda::free_async(d_inter_f32, stream).ok();
                cuda::free_async(d_inter_f16, stream).ok();
            }
        }
    })
}

/// Generate code for the LLaMA MLP: [RmsNorm+] dual GEMM(gate,up) + SiluMul -> GEMM(down).
///
/// Uses the embedded CUTLASS dual_gemm PTX (59.8 TFLOPS, 1 launch for dual GEMM).
/// The proc macro emits code that at runtime:
///   1. (Optional) RMSNorm the input
///   2. Pack CUTLASS DualGemmParams (720 bytes) and launch dual GEMM
///   3. Launch down-projection GEMM
///
/// This replaces 4-5 separate kernel launches with 2-3.
fn generate_dual_gemm_mlp_block(
    attr: &FuseAttr,
    input_fn: &ItemFn,
    _graph: &OpGraph,
    _stage1: &Stage,
    _stage2: &Stage,
) -> syn::Result<TokenStream> {
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_args = &input_fn.sig.inputs;
    let fn_attrs = &input_fn.attrs;

    // The CUTLASS dual GEMM PTX is embedded at compile time
    let cutlass_ptx = ferrite_ptx::fused::build_cutlass_dual_gemm();

    // Stage 2 GEMM for down-projection
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
        sm_arch: attr.arch.clone(),
    };
    let stage2_ptx = ferrite_ptx::gemm::build_gemm_pipeline(&config);
    let stage2_smem = config.smem_total() as u32;
    let stage2_threads = config.threads() as u32;

    let static_dual = quote::format_ident!(
        "__FERRITE_DUAL_GEMM_{}",
        fn_name.to_string().to_uppercase()
    );
    let static_down = quote::format_ident!(
        "__FERRITE_DOWN_GEMM_{}",
        fn_name.to_string().to_uppercase()
    );

    Ok(quote! {
        // Compile-time PTX validation
        const _: () = {
            const DUAL_PTX: &str = #cutlass_ptx;
            const DOWN_PTX: &str = #stage2_ptx;
        };

        // Lazy JIT-compiled kernel handles
        static #static_dual: ferrite_runtime::JitKernel =
            ferrite_runtime::JitKernel::new(#cutlass_ptx, "ferrite_dual_gemm_silu_mul");
        static #static_down: ferrite_runtime::JitKernel =
            ferrite_runtime::JitKernel::new(#stage2_ptx, "triton_style_gemm");

        #(#fn_attrs)*
        #fn_vis fn #fn_name(#fn_args) {
            // The dual GEMM kernel handles: input × [w_gate, w_up] → SiLU(gate) × up
            // Then the down-projection GEMM handles: hidden × w_down → output
            //
            // TODO: The CUTLASS dual GEMM expects a 720-byte DualGemmParams struct.
            // The packing code is in the ferrite-poc benchmark (cutlass_dual_gemm_params module).
            // For production use, this should be a shared library function.
            //
            // For now, this codegen path validates that the proc macro correctly
            // identifies the SiluMul pattern and dispatches to the dual GEMM path.
            // The actual kernel launch code will be added when the params packing
            // is extracted into a reusable module.

            todo!("Dual GEMM MLP launch: pack DualGemmParams + launch CUTLASS kernel + down-projection GEMM")
        }
    })
}

/// Generate code for GEMM+SiLU -> GEMM (no rmsnorm prologue).
///
/// Not yet implemented — placeholder for future extension.
fn generate_gemm_silu_gemm_block(
    _attr: &FuseAttr,
    input_fn: &ItemFn,
    _graph: &OpGraph,
    _stage1: &Stage,
    _stage2: &Stage,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "GEMM+SiLU -> GEMM 2-stage fusion not yet implemented. \
         Add an rmsnorm prologue or use the full MLP pattern.",
    ))
}

/// Generate code for GEMM -> GEMM chain (no prologue/epilogue on either stage).
///
/// Not yet implemented — placeholder for future extension.
fn generate_gemm_gemm_chain(
    _attr: &FuseAttr,
    input_fn: &ItemFn,
    _graph: &OpGraph,
    _stage1: &Stage,
    _stage2: &Stage,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "Standalone GEMM -> GEMM chain not yet implemented via proc macro. \
         Use ferrite_ptx::gemm::build_gemm() directly for each stage.",
    ))
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

/// Generate code for fused RmsNorm -> GEMM -> GELU megakernel.
fn generate_rmsnorm_gemm_gelu(
    _attr: &FuseAttr,
    input_fn: &syn::ItemFn,
    _graph: &OpGraph,
    _norm_id: usize,
    _gemm_id: usize,
    _gelu_id: usize,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "RmsNorm -> GEMM -> GELU fusion not yet implemented. \
         The strategy engine correctly identifies this pattern; codegen is pending.",
    ))
}

/// Generate code for fused GEMM -> GELU.
fn generate_gemm_gelu(
    _attr: &FuseAttr,
    input_fn: &syn::ItemFn,
    _graph: &OpGraph,
    _gemm_id: usize,
    _gelu_id: usize,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "Standalone GEMM+GELU fusion not yet implemented. \
         The strategy engine correctly identifies this pattern; codegen is pending.",
    ))
}

/// Generate code for RmsNorm+GEMM+GELU -> GEMM 2-stage plan.
fn generate_rmsnorm_gemm_gelu_gemm_block(
    _attr: &FuseAttr,
    input_fn: &syn::ItemFn,
    _graph: &OpGraph,
    _stage1: &Stage,
    _stage2: &Stage,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "RmsNorm+GEMM+GELU -> GEMM 2-stage fusion not yet implemented. \
         The strategy engine correctly identifies this pattern; codegen is pending.",
    ))
}

/// Generate code for GEMM+GELU -> GEMM 2-stage plan.
fn generate_gemm_gelu_gemm_block(
    _attr: &FuseAttr,
    input_fn: &syn::ItemFn,
    _graph: &OpGraph,
    _stage1: &Stage,
    _stage2: &Stage,
) -> syn::Result<TokenStream> {
    Err(syn::Error::new_spanned(
        &input_fn.sig.ident,
        "GEMM+GELU -> GEMM 2-stage fusion not yet implemented. \
         The strategy engine correctly identifies this pattern; codegen is pending.",
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
