// SPDX-License-Identifier: Apache-2.0
//! `#[forward]` attribute macro — compile a DSL forward pass into
//! specialized Rust + CUDA per model architecture.
//!
//! This crate is the proc-macro shell. Compiler passes (parsing,
//! classification, shape inference, CFG, unroll, solve, schedule,
//! codegen) live in internal modules here so they can be
//! unit-tested in isolation.

use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

mod ast;
mod cfg;
mod classified;
mod classify;
mod config;
mod parse;
mod shape;

/// Attribute macro entry point.
///
/// Attach to an empty carrier fn naming the architecture:
///
/// ```ignore
/// #[forward]
/// fn llama() {
///     hidden_states = embed(input_ids, embed_tokens);
///     for layer in 0..num_hidden_layers { ... }
///     ...
/// }
/// ```
///
/// Current state (up through Phase 1): parses the carrier fn's
/// body into our internal [`ast::Ast`] and then emits an empty fn
/// of the same signature. Later phases plug classification,
/// shape inference, codegen, and CUDA emission in.
#[proc_macro_attribute]
pub fn forward(_args: TokenStream, item: TokenStream) -> TokenStream {
    let carrier = parse_macro_input!(item as ItemFn);

    // Parse the body into our AST, then classify every free
    // variable reference. Errors propagate as compile errors at
    // the macro call site.
    let ast = match parse::parse_block(&carrier.block) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let _classified = match classify::classify(&ast) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    let sig = &carrier.sig;
    let vis = &carrier.vis;
    quote! {
        #vis #sig {
            // Phase 1: body parsed into AST but not yet lowered.
            // Emitting an empty fn; Phase 5+ produces real code.
        }
    }
    .into()
}
