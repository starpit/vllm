// SPDX-License-Identifier: Apache-2.0
//! `#[forward]` attribute macro — compile a DSL forward pass into
//! specialized Rust + CUDA per model architecture.
//!
//! Phase 0: scaffolding only. The macro accepts the attribute,
//! reads and discards the carrier fn's body, and emits an empty fn
//! with the same signature. Later phases plug real compilation in.

use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

/// Attribute macro entry point.
///
/// Attach to an empty carrier fn naming the architecture:
///
/// ```ignore
/// #[forward]
/// fn llama() {
///     // DSL body will go here in a later phase.
/// }
/// ```
///
/// In Phase 0 the macro does no real work — it parses and discards
/// the body, emitting the carrier fn unchanged but with an empty
/// body so downstream crates can link against it.
#[proc_macro_attribute]
pub fn forward(_args: TokenStream, item: TokenStream) -> TokenStream {
    let carrier = parse_macro_input!(item as ItemFn);
    let sig = &carrier.sig;
    let vis = &carrier.vis;
    quote! {
        #vis #sig {
            // Phase 0: body intentionally empty. Real codegen lands
            // in a later phase.
        }
    }
    .into()
}
