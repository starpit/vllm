// SPDX-License-Identifier: Apache-2.0
//! FUF codegen — generates Rust code from a solved Fully Unrolled Forward.
//!
//! Walks tiles in topological order. Each tile gets a unique variable
//! name (`t{id}`). The codegen looks up the solver-picked impl and
//! the tile's deps/weight_name to emit the correct FFI call.
//!
//! No hardcoded patterns, no booleans, no LLaMA-specific assumptions.
//! The tile graph IS the IR.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::lowering::assignment::Assignment;
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::tile_graph::{TileGraph, TileId, TileKind, TileNode};

/// Generate a variable name for a tile's output.
fn tile_var(id: TileId) -> proc_macro2::Ident {
    format_ident!("t{}", id.0)
}

/// Generate a field access expression for a weight.
/// Converts DSL names like "input_layernorm" to `layer.input_layernorm`
/// and "self_attn.o_proj" to `layer.self_attn_o_proj`.
fn weight_field(weight_name: &str, layer_idx: &TokenStream) -> TokenStream {
    // Per-layer weights: "foo[layer]" → model.layers[layer_idx].foo
    // Global weights: "foo" → model.foo
    let field_name = weight_name.replace('.', "_");
    let field = format_ident!("{}", field_name);
    quote! { model.layers[#layer_idx].#field }
}

/// Generate a field access for a global (non-per-layer) weight.
fn global_weight_field(weight_name: &str) -> TokenStream {
    let field_name = weight_name.replace('.', "_");
    let field = format_ident!("{}", field_name);
    quote! { model.#field }
}

/// Emit the code for one tile based on the solver-picked impl.
fn emit_tile(
    node: &TileNode,
    assignment: &Assignment,
    tile_graph: &TileGraph,
    library: &ImplementationLibrary,
) -> Option<TokenStream> {
    let sg = assignment.cover.get(&node.id)?;
    let impl_id = assignment.impls.get(sg)?;
    let imp = &library.entries[impl_id.0 as usize];
    let imp_name = imp.name();

    let out = tile_var(node.id);
    let layer_idx_lit = {
        let l = node.layer as usize;
        quote! { #l }
    };

    // Helper: get the variable for a dependency tile.
    let dep_var = |idx: usize| -> TokenStream {
        let dep_id = node.deps[idx];
        let v = tile_var(dep_id);
        quote! { #v }
    };

    // Noop tiles (QkvSplit, KvCacheWrite — logically free passthroughs).
    if imp_name == "qkv_split_free" || imp_name == "kv_cache_write" {
        return None;
    }

    let weight_name = node.weight_name.as_deref();

    match node.kind {
        TileKind::Embed => {
            let w = weight_name
                .map(global_weight_field)
                .unwrap_or_else(|| quote! { model.embed_tokens });
            Some(quote! {
                let #out = kernels::embedding_gather(
                    #w.weight,
                    *input_ids,
                    &mut device.caching,
                    device.compute_stream,
                );
            })
        }

        TileKind::RmsNorm => {
            let input = dep_var(0);
            let w = weight_name
                .map(|n| weight_field(n, &layer_idx_lit))
                .unwrap_or_else(|| quote! { model.norm });
            // Runtime check: if there's a residual to fuse,
            // use fused_add_rms_norm_inplace. Otherwise standalone.
            // The "residual" is tracked as the previous hidden_states
            // before the norm replaced it.
            // TODO: this still uses the fused pattern implicitly.
            // For now, emit standalone norm — the fused pattern
            // requires residual tracking that the FUF codegen
            // will handle via liveness analysis.
            Some(quote! {
                let #out = {
                    let __hs: GpuTensor = *#input;
                    kernels::rms_norm(
                        __hs,
                        #w.weight,
                        #w.eps,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            })
        }

        TileKind::GemmQ
        | TileKind::GemmK
        | TileKind::GemmV
        | TileKind::GemmOProj
        | TileKind::GemmGate
        | TileKind::GemmUp
        | TileKind::GemmDown
        | TileKind::GemmLmHead => {
            let input = dep_var(0);
            let w = if node.kind == TileKind::GemmLmHead {
                weight_name
                    .map(global_weight_field)
                    .unwrap_or_else(|| quote! { model.lm_head })
            } else {
                weight_name
                    .map(|n| weight_field(n, &layer_idx_lit))
                    .unwrap_or_else(|| quote! { model.lm_head })
            };

            // Check impl name to determine if it's cuBLAS or CUTLASS.
            if imp_name.starts_with("cublas") {
                Some(quote! {
                    let #out = #w.forward(
                        #input.view(),
                        &mut device.cublas,
                        &mut device.caching,
                        device.compute_stream,
                    );
                })
            } else if imp_name.starts_with("cutlass_gemv") {
                Some(quote! {
                    let #out = {
                        let __act: GpuTensor = *#input;
                        let __m = __act.dim(0) as i32;
                        let __k = __act.dim(1) as i32;
                        let __w = #w.dense_weight();
                        let __n = __w.dim(0) as i32;
                        let __out = device.caching.alloc_tensor(
                            &[__m as usize, __n as usize], __act.dtype(),
                        );
                        let __rc = cutlass_gemv_launch(
                            __out.as_mut_ptr::<u16>(),
                            __act.as_ptr::<u16>(),
                            __w.as_ptr::<u16>(),
                            __m, __n, __k,
                            1.0f32, 0.0f32,
                            device.compute_stream as u64,
                        );
                        debug_assert_eq!(__rc, 0, "CUTLASS GEMV failed");
                        __out
                    };
                })
            } else if imp_name.starts_with("cutlass_") {
                // CUTLASS GEMM — extract tile config from impl name.
                let launch_fn = format_ident!("{}_launch", imp_name);
                Some(quote! {
                    let #out = {
                        let __act: GpuTensor = *#input;
                        let __m = __act.dim(0) as i32;
                        let __k = __act.dim(1) as i32;
                        let __w = #w.dense_weight();
                        let __n = __w.dim(0) as i32;
                        let __out = device.caching.alloc_tensor(
                            &[__m as usize, __n as usize], __act.dtype(),
                        );
                        let __rc = #launch_fn(
                            __out.as_mut_ptr::<u16>(),
                            __act.as_ptr::<u16>(),
                            __w.as_ptr::<u16>(),
                            __m, __n, __k,
                            1.0f32, 0.0f32,
                            device.compute_stream as u64,
                        );
                        debug_assert_eq!(__rc, 0, "CUTLASS GEMM failed");
                        __out
                    };
                })
            } else {
                // Unknown GEMM impl — fallback to cuBLAS.
                Some(quote! {
                    let #out = #w.forward(
                        #input.view(),
                        &mut device.cublas,
                        &mut device.caching,
                        device.compute_stream,
                    );
                })
            }
        }

        TileKind::Attention => {
            let q = dep_var(0);
            Some(quote! {
                let #out = attention_helpers::attention_decode_from_cache(
                    #q.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    model.dims.scale,
                    0.0, -1,
                    kv_cache,
                    #layer_idx_lit,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    std::ptr::null(), 0, false,
                );
            })
        }

        TileKind::QkvSplit | TileKind::Rope | TileKind::KvCacheWrite => {
            // These are handled by the fused rope_append impl.
            // Skip — the fused impl emits all the code.
            None
        }

        TileKind::GateUpConcat => {
            // Logical concat — no actual kernel. The silu_mul impl
            // handles reading both gate and up outputs.
            None
        }

        TileKind::SiluMul => {
            // Find the GateUpConcat dep, then its two deps (gate, up).
            let concat_id = node.deps[0];
            let concat = &tile_graph.nodes[concat_id.0 as usize];
            let gate = tile_var(concat.deps[0]);
            let up = tile_var(concat.deps[1]);
            Some(quote! {
                let #out = {
                    // Concatenate gate + up, then silu_and_mul.
                    let __concat = kernels::concat_dim1(
                        *#gate, *#up,
                        &mut device.caching,
                        device.compute_stream,
                    );
                    let __activated = kernels::silu_and_mul_fused(
                        __concat.as_gpu_tensor(),
                        model.dims.intermediate_size,
                        &mut device.caching,
                        device.compute_stream,
                    );
                    drop(__concat);
                    __activated
                };
            })
        }

        TileKind::ResidualAdd => {
            // Element-wise add of two tensors.
            if node.deps.len() >= 2 {
                let a = dep_var(0);
                let b = dep_var(1);
                Some(quote! {
                    let #out = {
                        // In-place add: a += b, return a.
                        let __a_gpu: GpuTensor = *#a;
                        let __b_gpu: GpuTensor = *#b;
                        kernels::add_inplace(__a_gpu, __b_gpu, device.compute_stream);
                        #a
                    };
                })
            } else {
                None
            }
        }

        TileKind::BiasAdd => {
            let input = dep_var(0);
            Some(quote! {
                let #out = {
                    // Bias add is handled by the fused GEMM impl.
                    // Standalone bias: add bias vector to each row.
                    // TODO: emit standalone bias_add kernel call.
                    #input
                };
            })
        }
    }
}

/// Generate the full forward function from a solved FUF.
pub fn generate_fuf_forward(
    fuf: &TileGraph,
    assignment: &Assignment,
    library: &ImplementationLibrary,
) -> TokenStream {
    let mut stmts = Vec::new();

    for node in fuf.iter_topo() {
        if let Some(ts) = emit_tile(node, assignment, fuf, library) {
            stmts.push(ts);
        }
    }

    // The last tile should be the lm_head output.
    let last_tile = fuf
        .nodes
        .iter()
        .rev()
        .find(|n| n.kind == TileKind::GemmLmHead)
        .map(|n| tile_var(n.id))
        .unwrap_or_else(|| format_ident!("t0"));

    quote! {
        #(#stmts)*
        #last_tile
    }
}
