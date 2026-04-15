// SPDX-License-Identifier: Apache-2.0
//! Emission context for [`Implementation::emit_call`].
//!
//! Each impl in the library owns its codegen. When the solver binds
//! a subgraph to an impl, codegen asks the impl to emit the Rust
//! tokens that invoke its kernel. The impl reads shapes, input
//! expressions, and output idents off [`EmitCtx`] — it never
//! reaches into the FUF directly to discover downstream bindings.
//!
//! This is the extension point that keeps new kernels from
//! requiring compiler edits: a new impl is a new struct with a new
//! `emit_call` body. Codegen's outer loop is the same shape for
//! every impl.

#![allow(dead_code)]

use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::classified::{ExternKind, Program, WeightId};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};

/// State threaded through per-subgraph emission: names of the Rust
/// let-bindings that hold each tile's outputs so downstream tiles
/// can reference them. `locals[(tile_id, slot)]` is the identifier.
pub type LocalMap = HashMap<(TileId, u8), syn::Ident>;

/// Context passed to each [`crate::impl_lib::Implementation::emit_call`].
///
/// Carries enough to emit the let-binding(s) for one subgraph's
/// output, reading inputs from already-bound locals / the
/// `WeightBundle` / the `ForwardCtx`.
pub struct EmitCtx<'a> {
    pub fuf: &'a Fuf,
    pub program: &'a Program,
    /// Model being emitted — gives impls read-access to bounds
    /// (intermediate_size, hidden_size, head_dim, …) when shaping
    /// kernel arguments.
    pub model: &'a ModelParams,
    /// Tiles this subgraph claims, in topological order (the order
    /// the DP committed them).
    pub claimed_tiles: &'a [TileId],
    /// Map from (tile, slot) → ident of the let-binding holding
    /// that value in the enclosing fn's scope.
    pub locals: &'a LocalMap,
}

impl<'a> EmitCtx<'a> {
    /// The first claimed tile. For singleton subgraphs, this is
    /// "the" tile.
    pub fn primary(&self) -> TileId {
        self.claimed_tiles[0]
    }

    /// FUF node for a claimed tile.
    pub fn node(&self, tile: TileId) -> &FufNode {
        self.fuf.get(tile)
    }

    /// Rust expression invoking a [`crate::impl_lib::WeightAccessor`]
    /// by name on the enclosing `wm: &impl WeightBundle`. Impls that
    /// override `required_weights` reference their declared
    /// accessors through this rather than constructing field idents
    /// ad hoc.
    pub fn weight_accessor(&self, name: &syn::Ident) -> TokenStream {
        // Field access on the emitted `Weights` struct. No trait
        // indirection — the compiler generates both the struct
        // definition and the `Weights::load` method, so the caller
        // never writes weight-bookkeeping code.
        quote! { wm.#name }
    }

    /// Read a model-wide integer bound (e.g. `intermediate_size`,
    /// `head_dim`). Panics if the key is missing — callers know the
    /// keys they need by DSL / conventions.
    pub fn bound(&self, key: &str) -> u64 {
        *self
            .model
            .bounds
            .get(key)
            .unwrap_or_else(|| panic!("model has no bound `{key}` in config.json"))
    }

    /// Read a model-wide float scalar (e.g. `query_pre_attn_scalar`,
    /// `attn_logit_softcapping`). Returns `None` if the key is
    /// absent; callers pick an architecture-appropriate default in
    /// that case. Unlike [`bound`](Self::bound) this never panics —
    /// most architectures carry no floats at all, and Impls should
    /// treat presence as opt-in parameterization.
    pub fn scalar(&self, key: &str) -> Option<f64> {
        self.model.scalars.get(key).copied()
    }

    /// Rust expression that evaluates to the Nth input of `tile`.
    ///
    /// Reads from the local binding for tile-sourced inputs, from
    /// the `wm: &impl WeightBundle` binding for weight-sourced
    /// inputs, and from the `ctx: &ForwardCtx` binding for extern
    /// inputs.
    pub fn input_expr(&self, tile: TileId, slot: usize) -> TokenStream {
        let node = self.fuf.get(tile);
        let input = &node.inputs[slot];
        self.emit_input(input)
    }

    /// Like `input_expr` but returns one expression per input in
    /// declared order.
    pub fn all_input_exprs(&self, tile: TileId) -> Vec<TokenStream> {
        let node = self.fuf.get(tile);
        node.inputs.iter().map(|i| self.emit_input(i)).collect()
    }

    /// The raw let-binding ident of the producer tile for `tile`'s
    /// input slot — `None` if the slot is a Weight or Extern.
    ///
    /// Fusion impls that need to alias an upstream OwnedTensor (e.g.
    /// a kernel that mutates `residual` in place and wants a
    /// downstream TensorView borrowed off the same storage) use this
    /// to reach past `input_expr`'s `(*_).as_view()` wrapper.
    pub fn input_tile_ident(&self, tile: TileId, slot: usize) -> Option<syn::Ident> {
        let node = self.fuf.get(tile);
        match node.inputs.get(slot)? {
            FufInput::Tile { id, slot } => self.locals.get(&(*id, *slot)).cloned(),
            _ => None,
        }
    }

    fn emit_input(&self, input: &FufInput) -> TokenStream {
        match input {
            FufInput::Tile { id, slot } => {
                let ident = self
                    .locals
                    .get(&(*id, *slot))
                    .cloned()
                    .unwrap_or_else(|| format_ident!("__missing_tile_{}_{}", id.0, slot));
                quote! { (*#ident).as_view() }
            }
            FufInput::Weight { id, index } => {
                let name = weight_field_name(self.program, *id, *index);
                quote! { wm.#name }
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

    /// The let-binding ident holding `(tile, slot)`'s output.
    pub fn output_ident(&self, tile: TileId, slot: u8) -> syn::Ident {
        self.locals
            .get(&(tile, slot))
            .cloned()
            .unwrap_or_else(|| format_ident!("t_{}_{}", tile.0, slot))
    }
}

/// The field name for a weight instance. `self_attn.q_proj` at
/// layer 3 → `self_attn_q_proj_3`; unindexed `embed_tokens` →
/// `embed_tokens`. Shared with `WeightBundle` trait emission so
/// the accessor names match what `emit_call` calls.
pub fn weight_field_name(program: &Program, id: WeightId, index: Option<u64>) -> syn::Ident {
    let path = program.weights.path(id);
    let dotted: Vec<String> = path.iter().map(|s| s.to_string()).collect();
    let stem = dotted.join("_");
    let ident = match index {
        Some(i) => format!("{stem}_{i}"),
        None => stem,
    };
    format_ident!("{}", ident)
}
