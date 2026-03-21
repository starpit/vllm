extern crate proc_macro;

mod codegen;
mod ops;
mod parse;
mod strategy;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// The `#[fuse]` attribute macro for Ferrite kernel fusion.
///
/// Parses a function body containing `rmsnorm`, `gemm`, and `silu` calls,
/// builds an operation graph, evaluates fusion strategies, generates PTX
/// at compile time via `ferrite-ptx`, and emits Rust code that JIT-compiles
/// and launches the fused kernels at runtime.
///
/// # Example
///
/// ```ignore
/// #[ferrite_macros::fuse(arch = "sm_89")]
/// fn mlp_block(
///     x: DevicePtr, w_gate: DevicePtr, w_down: DevicePtr, norm_w: DevicePtr,
///     m: u32, n: u32, k: u32,
/// ) -> DevicePtr {
///     let n = rmsnorm(x, norm_w);
///     let g = gemm(n, w_gate);
///     let h = silu(g);
///     gemm(h, w_down)
/// }
/// ```
#[proc_macro_attribute]
pub fn fuse(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(attr as parse::FuseAttr);
    let input_fn = parse_macro_input!(item as syn::ItemFn);

    match fuse_impl(attr_args, input_fn) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error().into(),
    }
}

fn fuse_impl(attr: parse::FuseAttr, input_fn: syn::ItemFn) -> syn::Result<TokenStream> {
    // Step 1: Parse function body into OpGraph
    let graph = parse::parse_fn_body(&input_fn)?;

    // Step 2: Evaluate fusion strategy
    let plan = strategy::evaluate_strategy(&graph);

    // Step 3: Generate code
    let output = codegen::generate(&attr, &input_fn, &graph, &plan)?;

    Ok(output.into())
}
