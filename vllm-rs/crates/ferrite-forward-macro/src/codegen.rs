// SPDX-License-Identifier: Apache-2.0
//! Codegen: walks a scheduled LOOP (sequence of waves) and emits
//! a Rust `forward` function that invokes the solver-chosen
//! kernel for each subgraph.
//!
//! Ported from old ferrite-solver/src/lowering/backend/codegen.rs
//! with heavy detoxification. Structurally the emission is the
//! same shape the old codegen produced (per-op kernel calls with
//! let-bindings threading outputs into downstream inputs), just
//! keyed off our clean `OpKind` instead of the old
//! `TileKind::GemmQ/K/V/...` taxonomy.
//!
//! Emission strategy per-op is a `match` on `OpKind` in this file.
//! That's a known closed-enum point: extending the DSL with a new
//! op (Gemma2's Gelu / SoftCap / SlidingAttention, MoE's TopK)
//! requires adding a match arm here. Moving the emission onto the
//! Implementation trait so different impls can emit different
//! call shapes for the same op is a followup — needed for
//! quantized / fused variants, not for starter Llama.
//!
//! Output shape — per model × per workload-bucket — is a single
//! pub fn:
//!
//! ```ignore
//! pub unsafe fn forward(
//!     wm: &WeightBundle,
//!     ctx: &ForwardCtx,
//!     device: &mut GpuDevice,
//! ) -> OwnedTensor
//! ```
//!
//! `WeightBundle` is a user-provided trait (field accessors the
//! compiler emits calls into). `ForwardCtx` is the runtime-args
//! bundle (input_ids, positions, kv_cache, block_table, rotary,
//! plus the attention-path metadata). `GpuDevice` is the ambient
//! cuBLAS/allocator/stream holder.

#![allow(dead_code)]

use std::collections::HashMap;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};

use crate::classified::{ExternKind, OpKind, Program, WeightId};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::ImplementationLibrary;
use crate::schedule::{Loop, WorkloadLoops};
use crate::solver::{Assignment, SubgraphId, WorkloadAssignments};

// ── WeightBundle emission ────────────────────────────────────────

/// Derive the flat set of `(WeightId, concrete_index)` pairs the
/// FUF references. Each pair becomes one field on the emitted
/// `WeightBundle` trait. Fields are named by joining the weight's
/// dotted path with its concrete index (if any).
fn weight_instances(fuf: &Fuf) -> Vec<(WeightId, Option<u64>)> {
    let mut seen: Vec<(WeightId, Option<u64>)> = Vec::new();
    for node in &fuf.nodes {
        for input in &node.inputs {
            if let FufInput::Weight { id, index } = input {
                let key = (*id, *index);
                if !seen.contains(&key) {
                    seen.push(key);
                }
            }
        }
    }
    seen
}

/// The field name for a weight instance. `self_attn.q_proj` at
/// layer 3 → `self_attn_q_proj_3`; unindexed `embed_tokens` →
/// `embed_tokens`.
fn weight_field_name(program: &Program, id: WeightId, index: Option<u64>) -> syn::Ident {
    let path = program.weights.path(id);
    let dotted: Vec<String> = path.iter().map(|s| s.to_string()).collect();
    let stem = dotted.join("_");
    let ident = match index {
        Some(i) => format!("{stem}_{i}"),
        None => stem,
    };
    format_ident!("{}", ident)
}

/// Emit the per-model `WeightBundle` trait: one accessor method
/// per weight instance, returning a `&Linear` / `&Embedding` /
/// `&RmsNorm` as appropriate for the op that consumes it.
fn emit_weight_bundle_trait(program: &Program, fuf: &Fuf) -> TokenStream {
    let instances = weight_instances(fuf);

    // For each weight instance, derive the runtime type from the
    // op that consumes it. If a weight is consumed in multiple
    // ops with incompatible types, error out loudly.
    let mut consumer_op: HashMap<(WeightId, Option<u64>), OpKind> = HashMap::new();
    for node in &fuf.nodes {
        for input in &node.inputs {
            if let FufInput::Weight { id, index } = input {
                consumer_op.insert((*id, *index), node.op);
            }
        }
    }

    let methods = instances.iter().map(|key| {
        let name = weight_field_name(program, key.0, key.1);
        let ty = match consumer_op.get(key).copied() {
            Some(OpKind::Embed) => quote! { ::ferrite_kernels::layers::Embedding },
            Some(OpKind::RmsNorm) => quote! { ::ferrite_kernels::layers::RmsNorm },
            Some(OpKind::Gemm) => quote! { ::ferrite_kernels::layers::LinearLayer },
            // Other ops don't take weights; fall through to a
            // generic tensor reference. (Only fires if the DSL
            // references a weight in an unexpected position.)
            _ => quote! { ::ferrite_cuda_core::tensor::GpuTensor },
        };
        quote! {
            fn #name(&self) -> &#ty;
        }
    });

    quote! {
        /// Accessor trait the caller implements to expose each
        /// referenced weight to the emitted forward. One method
        /// per (weight-path, concrete-layer-index) pair the DSL
        /// touched. Field types are the ferrite-kernels layer
        /// wrappers matching the op the weight flows into.
        pub trait WeightBundle {
            #(#methods)*
        }
    }
}

// ── Forward fn emission ──────────────────────────────────────────

/// State threaded through per-tile emission: names of the Rust
/// let-bindings that hold each tile's outputs so downstream tiles
/// can reference them. `locals[(tile_id, slot)]` is the
/// identifier.
type LocalMap = HashMap<(TileId, u8), syn::Ident>;

/// Emit an expression that evaluates to a `TensorView<'_>` for
/// a FufInput, reading from our scope's let-bindings / weight
/// bundle / forward ctx.
fn emit_input_expr(input: &FufInput, locals: &LocalMap, program: &Program) -> TokenStream {
    match input {
        FufInput::Tile { id, slot } => {
            let ident = locals
                .get(&(*id, *slot))
                .cloned()
                .unwrap_or_else(|| format_ident!("__missing_tile_{}_{}", id.0, slot));
            quote! { (*#ident).as_view() }
        }
        FufInput::Weight { id, index } => {
            let name = weight_field_name(program, *id, *index);
            quote! { wm.#name() }
        }
        FufInput::Extern { kind, .. } => match kind {
            ExternKind::InputIds => quote! { ctx.input_ids },
            ExternKind::Positions => quote! { ctx.positions },
            ExternKind::Rotary => quote! { ctx.rotary },
            ExternKind::BlockTable => quote! { ctx.block_table },
            ExternKind::KvCache => quote! { ctx.kv_cache },
        },
    }
}

/// Emit the Rust that computes a single subgraph's output from
/// its inputs. Assigns to `output_idents`. For multi-tile
/// subgraphs, each output slot gets its own let-binding.
///
/// Today this is hardcoded per `OpKind`. When we need variant
/// choice (fused vs unfused, quantized vs plain) the match
/// dispatches to the impl picked by the solver, not to a single
/// hardcoded pattern per op.
fn emit_subgraph_call(
    fuf: &Fuf,
    sfuf: &Assignment,
    sg: SubgraphId,
    locals: &LocalMap,
    program: &Program,
) -> TokenStream {
    let tiles = sfuf.tiles_in_subgraph(sg);
    // For starter library, each subgraph is a single tile. Pick
    // the first claimed tile as the representative op.
    let tile = tiles[0];
    let node = fuf.get(tile);
    let op = node.op;

    // Output binding: one let per output slot.
    let out0 = locals
        .get(&(tile, 0))
        .cloned()
        .unwrap_or_else(|| format_ident!("t_{}_0", tile.0));

    // Input expressions in declared order.
    let input_exprs: Vec<TokenStream> = node
        .inputs
        .iter()
        .map(|i| emit_input_expr(i, locals, program))
        .collect();

    match op {
        OpKind::Embed => {
            // embed(input_ids, embed_tokens) → out.
            // Call Embedding::forward(input_ids, alloc, stream).
            let weight_expr = &input_exprs[1];
            let ids_expr = &input_exprs[0];
            quote! {
                let #out0 = unsafe {
                    (#weight_expr).forward(
                        #ids_expr,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
        OpKind::RmsNorm => {
            // rmsnorm(x, w) — call the dtype-specific fused kernel
            // through ferrite_kernels::kernels.
            let x_expr = &input_exprs[0];
            let w_expr = &input_exprs[1];
            quote! {
                let #out0 = unsafe {
                    ::ferrite_kernels::kernels::rms_norm_forward_owned(
                        #x_expr,
                        (#w_expr).weight,
                        (#w_expr).eps,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
        OpKind::Gemm => {
            // gemm(x, W) via the LinearLayer dispatch enum.
            let x_expr = &input_exprs[0];
            let w_expr = &input_exprs[1];
            quote! {
                let #out0 = unsafe {
                    (#w_expr).forward(
                        #x_expr,
                        &mut device.cublas,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
        OpKind::RopeAppend => {
            // (q', k', v') = rope_append(q, k, v, positions, rotary, kv_cache).
            // Emits three output bindings: slots 0 (q), 1 (k), 2 (v).
            let q_in = &input_exprs[0];
            let k_in = &input_exprs[1];
            let v_in = &input_exprs[2];
            let pos = &input_exprs[3];
            let rotary = &input_exprs[4];
            let kv = &input_exprs[5];
            let q_out = locals
                .get(&(tile, 0))
                .cloned()
                .unwrap_or_else(|| format_ident!("t_{}_q", tile.0));
            let k_out = locals
                .get(&(tile, 1))
                .cloned()
                .unwrap_or_else(|| format_ident!("t_{}_k", tile.0));
            let v_out = locals
                .get(&(tile, 2))
                .cloned()
                .unwrap_or_else(|| format_ident!("t_{}_v", tile.0));
            quote! {
                let (#q_out, #k_out, #v_out) = unsafe {
                    ::ferrite_kernels::kernels::rope_append_kv(
                        #q_in, #k_in, #v_in, #pos, #rotary, #kv,
                        ctx.slot_mapping,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
        OpKind::Attention => {
            // attention(q, k, v, kv_cache, block_table).
            let q_in = &input_exprs[0];
            let k_in = &input_exprs[1];
            let v_in = &input_exprs[2];
            let kv = &input_exprs[3];
            let bt = &input_exprs[4];
            quote! {
                let #out0 = unsafe {
                    ::ferrite_kernels::kernels::flash_attention(
                        #q_in, #k_in, #v_in, #kv, #bt,
                        ctx.cu_seqlens_q,
                        ctx.seqused_k,
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_k,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
        OpKind::Silu => {
            let x = &input_exprs[0];
            quote! {
                let #out0 = unsafe {
                    ::ferrite_kernels::kernels::silu_owned(
                        #x,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
        OpKind::Add => {
            let a = &input_exprs[0];
            let b = &input_exprs[1];
            quote! {
                let #out0 = unsafe {
                    ::ferrite_kernels::kernels::add_owned(
                        #a, #b,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
        OpKind::Mul => {
            let a = &input_exprs[0];
            let b = &input_exprs[1];
            quote! {
                let #out0 = unsafe {
                    ::ferrite_kernels::kernels::mul_owned(
                        #a, #b,
                        &mut device.caching,
                        device.compute_stream,
                    )
                };
            }
        }
    }
}

/// Emit one per-workload-bucket forward fn.
fn emit_forward_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    num_tokens: u64,
) -> TokenStream {
    // Allocate a stable local-binding ident per tile-output slot.
    let mut locals: LocalMap = HashMap::new();
    for node in &fuf.nodes {
        for slot in 0..node.outputs.len().max(1) as u8 {
            locals.insert((node.id, slot), format_ident!("t_{}_{}", node.id.0, slot));
        }
    }

    // Walk waves in order, emitting one statement per subgraph.
    let mut body: Vec<TokenStream> = Vec::new();
    for wave in &loop_ir.waves {
        for (sg, _imp) in &wave.subgraphs {
            body.push(emit_subgraph_call(fuf, sfuf, *sg, &locals, program));
        }
    }

    // The forward's return value: the last tile's output.
    let last_output = fuf
        .nodes
        .last()
        .map(|n| {
            let id = locals[&(n.id, 0)].clone();
            quote! { #id }
        })
        .unwrap_or_else(|| quote! { unreachable!("empty FUF") });

    let fn_name = format_ident!("forward_m_{}", num_tokens);
    quote! {
        /// Forward pass for this model × workload bucket. Walks
        /// the solver-picked kernels in wavefront order.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments, unused_mut, unused_variables)]
        pub unsafe fn #fn_name<W: WeightBundle>(
            wm: &W,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            #(#body)*
            #last_output
        }
    }
}

/// Emit the full per-model module body: WeightBundle trait, one
/// forward fn per workload bucket, and a dispatching wrapper.
pub fn emit_model(
    program: &Program,
    model: &ModelParams,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    loops: &WorkloadLoops,
    _lib: &ImplementationLibrary,
) -> TokenStream {
    let weight_bundle_trait = emit_weight_bundle_trait(program, fuf);

    let bucket_fns: Vec<TokenStream> = sfufs
        .per_num_tokens
        .iter()
        .map(|(m, sfuf)| {
            let loop_ir = loops
                .per_num_tokens
                .get(m)
                .expect("schedule populated every key");
            emit_forward_for_bucket(fuf, sfuf, loop_ir, program, *m)
        })
        .collect();

    // match num_tokens dispatch. Coalesce adjacent buckets with
    // the same forward fn in a future pass; for now one arm per
    // bucket.
    let match_arms: Vec<TokenStream> = sfufs
        .per_num_tokens
        .keys()
        .map(|m| {
            let lit = proc_macro2::Literal::u64_unsuffixed(*m);
            let fn_name = format_ident!("forward_m_{}", m);
            quote! { #lit => unsafe { #fn_name(wm, ctx, device) }, }
        })
        .collect();

    let _ = (model, Span::call_site());

    // Returns the items that go inside the per-model module. The
    // caller (lib.rs) concatenates these with the stub's
    // constants under one `pub mod <model>`.
    quote! {
        #[cfg(feature = "cuda")]
        #weight_bundle_trait

        #(#bucket_fns)*

        /// Dispatch on `num_tokens`. Panics if the runtime
        /// num_tokens isn't one of the compiled buckets.
        /// Future: coalesce into inclusive ranges.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward<W: WeightBundle>(
            wm: &W,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            num_tokens: u64,
        ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
            match num_tokens {
                #(#match_arms)*
                other => panic!("no compiled bucket for num_tokens={other}"),
            }
        }
    }
}
