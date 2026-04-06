// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]
//! Core logic for the `megakernel!` proc-macro.
//!
//! This is a regular library crate (not proc-macro) so it can be used by both:
//! - The `vllm-tk-macros` proc-macro crate (for compile-time expansion)
//! - The `vllm-tk-static` build.rs (for generating CUDA source to compile)

#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::useless_vec
)]
pub mod cpu_golden;
pub mod cuda_codegen;
pub mod dag;
pub mod diagram;
pub mod parse;
pub mod verify;

/// Generate the complete CUDA source for a LLaMA-like megakernel from DSL source.
///
/// This is the entry point for build.rs: pass the same DSL text that `megakernel!`
/// would parse, get back the CUDA source string.
pub fn generate_cuda_from_dsl(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;

    let dag = parse::build_dag(&def)?;

    let errors = verify::verify(&dag);
    if !errors.is_empty() {
        let mut msg = String::from("verification failed:\n");
        for e in &errors {
            msg.push_str(&format!("  {e}\n"));
        }
        return Err(msg);
    }

    Ok(cuda_codegen::generate_static_kernel(&dag))
}

/// Generate a standalone CUDA test kernel for a single TK op.
///
/// `dsl` is the same DSL text as `generate_cuda_from_dsl`.
/// `op_name` is one of the TK op names (e.g. "attn_norm", "gate_silu").
/// Returns CUDA source with a `test_{op_name}_launch(...)` C entry point.
pub fn generate_single_op_kernel(dsl: &str, op_name: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    cuda_codegen::generate_single_op_kernel(&dag, op_name)
}

/// List all op names that can be tested with `generate_single_op_kernel`.
pub fn available_op_names() -> Vec<&'static str> {
    cuda_codegen::available_op_names()
}

/// Generate an inline RMSNorm kernel (no KVM protocol).
///
/// Uses TK tile primitives + group::sync only. No semaphores, no pages, no warp roles.
pub fn generate_inline_rmsnorm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_inline_rmsnorm_kernel(&dag))
}

/// Generate an inline GEMM kernel (no KVM protocol).
///
/// Double-buffered K-loop with 8 cooperative warps using TK tile primitives.
pub fn generate_inline_gemm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_inline_gemm_kernel(&dag))
}

/// Generate a fused MLP block kernel (no KVM protocol).
pub fn generate_fused_mlp_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_mlp_kernel(&dag))
}

/// Generate a fused RMSNorm → GEMM kernel (no KVM protocol, shmem inter-op passing).
pub fn generate_fused_rmsnorm_gemm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_fused_rmsnorm_gemm_kernel(&dag))
}

/// Generate an inline attention decode kernel (no KVM protocol).
///
/// Each of 8 warps handles 1 KV head with GQA_RATIO query heads.
pub fn generate_inline_attention_decode_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_inline_attention_decode_kernel(&dag))
}

/// Generate a fused single-layer kernel (no KVM protocol).
///
/// Chains: attn_norm → QKV GEMM → [skip attention] → o_proj+residual → MLP block.
pub fn generate_fused_layer_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_layer_kernel(&dag))
}

/// Generate a fused full-layer kernel WITH attention decode (no KVM protocol).
pub fn generate_fused_full_layer_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_full_layer_kernel(&dag))
}

/// Generate a fused multi-layer kernel WITH attention decode (no KVM protocol).
pub fn generate_fused_multi_layer_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_multi_layer_kernel(&dag))
}

/// Generate a fused multi-SM kernel that distributes GEMM tiles across CTAs.
///
/// Uses cross-CTA barriers (atomicAdd + spin) for synchronization between phases.
/// RMSNorm and attention run on CTA 0; GEMMs are partitioned across all CTAs.
pub fn generate_fused_multi_sm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_multi_sm_kernel(&dag))
}

/// Generate a debug variant of the decode kernel that syncs and writes a
/// marker to a debug buffer after each op. Useful for identifying which op
/// crashes in the full megakernel.
pub fn generate_debug_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_debug_kernel(&dag))
}
