// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]
//! Core logic for the `megakernel!` proc-macro.
//!
//! This is a regular library crate (not proc-macro) so it can be used by both:
//! - The `vllm-tk-macros` proc-macro crate (for compile-time expansion)
//! - The `vllm-tk-static` build.rs (for generating CUDA source to compile)

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
