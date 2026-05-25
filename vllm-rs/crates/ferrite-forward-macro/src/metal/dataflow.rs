// SPDX-License-Identifier: Apache-2.0
//! Metal-specific helper: emit a per-bucket `static [bool]` slice
//! carrying the compile-time barrier-before flags. The flags are
//! computed by the generic FUF-level analyzer in
//! `interpreter_codegen::compute_barrier_before` — this module
//! only handles the rendering side.

use proc_macro2::TokenStream;
use quote::quote;

/// Emit a `static <ident>: &[bool] = &[…];` slice. Length matches
/// the corresponding `emit_bucket_static_slice` instruction static
/// (pre-loop-compression — one flag per `OpInstance`).
pub fn emit_bucket_barriers_static(static_ident: &syn::Ident, barriers: &[bool]) -> TokenStream {
    let elements = barriers.iter().map(|b| {
        if *b {
            quote! { true }
        } else {
            quote! { false }
        }
    });
    quote! {
        #[cfg(feature = "metal")]
        static #static_ident: &[bool] = &[ #(#elements),* ];
    }
}
