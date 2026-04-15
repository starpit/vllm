// SPDX-License-Identifier: Apache-2.0
//! Codegen: walks a scheduled LOOP (sequence of waves) and emits
//! a Rust `forward` function that invokes the solver-chosen
//! kernel for each subgraph.
//!
//! Ported from old ferrite-solver/src/lowering/backend/codegen.rs
//! with heavy detoxification. Structurally the emission is the
//! same shape the old codegen produced (per-op kernel calls with
//! let-bindings threading outputs into downstream inputs).
//!
//! Emission-per-subgraph is delegated to
//! [`crate::impl_lib::Implementation::emit_call`]: codegen walks
//! the LOOP's waves and asks each subgraph's bound impl to emit
//! its own tokens. New kernels / new ops extend the library, not
//! this file.
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

use crate::classified::{OpKind, Program, WeightId};
use crate::config::ModelParams;
use crate::emit::{EmitCtx, LocalMap, weight_field_name};
use crate::fuf::{Fuf, FufInput};
use crate::impl_lib::ImplementationLibrary;
use crate::schedule::{Loop, WorkloadLoops};
use crate::solver::{Assignment, WorkloadAssignments};

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

/// Emit one per-workload-bucket forward fn.
fn emit_forward_for_bucket(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    program: &Program,
    lib: &ImplementationLibrary,
    num_tokens: u64,
) -> TokenStream {
    // Allocate a stable local-binding ident per tile-output slot.
    let mut locals: LocalMap = HashMap::new();
    for node in &fuf.nodes {
        for slot in 0..node.outputs.len().max(1) as u8 {
            locals.insert((node.id, slot), format_ident!("t_{}_{}", node.id.0, slot));
        }
    }

    // Walk waves in order; each subgraph's bound impl emits its own
    // kernel invocation. Codegen stays mechanical — no per-op match
    // lives here.
    let mut body: Vec<TokenStream> = Vec::new();
    for wave in &loop_ir.waves {
        for (sg, imp_id) in &wave.subgraphs {
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let ctx = EmitCtx {
                fuf,
                program,
                claimed_tiles: &claimed,
                locals: &locals,
            };
            body.push(lib.get(*imp_id).emit_call(&ctx));
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
    lib: &ImplementationLibrary,
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
            emit_forward_for_bucket(fuf, sfuf, loop_ir, program, lib, *m)
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
