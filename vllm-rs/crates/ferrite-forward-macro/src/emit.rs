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

/// Emission mode for a subgraph.
///
/// `Concrete` (today's default) inlines the kernel call directly
/// into the enclosing forward fn, referencing `t_<id>_<slot>`
/// locals, `wm.<field>` weights, and `ctx.*` externs.
///
/// `Abstract` emits the SAME kernel call body but with input /
/// weight references replaced by fn-parameter idents (`input_0`,
/// `w_0`, …). The result is a reusable fragment body that can be
/// wrapped in a `fn __frag_<hash>(...)` and called from many
/// subgraphs whose emit is structurally identical. Extern (`ctx.*`)
/// and scalar inputs are emitted verbatim in both modes — externs
/// are always the same bindings, scalars are part of the fragment's
/// identity.
#[derive(Clone, Debug)]
pub enum EmitMode {
    Concrete,
    /// In this mode, `input_expr` / `weight_accessor` / friends
    /// return abstract parameter names instead of concrete
    /// `wm.`/`locals[..]` expressions. The accompanying maps tell
    /// `emit_input` which abstract ident to substitute for each
    /// boundary Tile / Weight.
    Abstract {
        /// Boundary tile inputs passed as `TensorView<'_>` — the
        /// kernel only needs to READ them. Maps
        /// (producer_tile, slot) → fragment fn-param ident. At the
        /// call site the concrete local is wrapped as
        /// `(*#local).as_view()`.
        tile_params: HashMap<(TileId, u8), syn::Ident>,
        /// Boundary tile inputs passed as `OwnedTensor` BY VALUE
        /// — the impl's `consumes_input_tiles()` declared it
        /// would move the upstream owner into its own output
        /// (e.g. in-place kernels like add_rmsnorm, scale_inplace,
        /// tanh_softcap_inplace). At the call site the concrete
        /// local is passed by move. `input_tile_ident()` returns
        /// the param ident directly; `input_expr()` wraps with
        /// `(*…).as_view()` for the rare impl that also reads the
        /// consumed value through the view path.
        consumed_params: HashMap<(TileId, u8), syn::Ident>,
        /// Weight inputs → fragment fn-param ident. Populated for
        /// every `(WeightId, Option<u64>)` that appears as a source
        /// weight of any accessor. Fragment params are
        /// `&WeightStruct` (e.g. `w_0: &RmsNorm`), matching what
        /// today's `wm.<field>` would resolve to at the call site.
        weight_params: HashMap<(crate::classified::WeightId, Option<u64>), syn::Ident>,
        /// Explicit `weight_accessor(name)` override — impls call
        /// this with the declared accessor name (e.g. the fused
        /// `self_attn_qkv_0` accessor, whose source_weights include
        /// three separate q/k/v ids). The accessor's name is the
        /// Weights-struct field, NOT derivable from any single
        /// source-weight ident via `weight_field_name`, so we keep
        /// a parallel by-name map.
        weight_params_by_name: HashMap<String, syn::Ident>,
    },
}

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
    /// Concrete (inline today's forward fn) vs Abstract (emit a
    /// reusable fragment body whose inputs/weights are fn params).
    /// Defaults to Concrete when `EmitCtx` is built via the old
    /// field-literal style.
    pub mode: EmitMode,
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
    ///
    /// In [`EmitMode::Abstract`] mode the supplied `name` is the
    /// concrete per-model field (e.g. `self_attn_q_proj_3`); we look
    /// up the fragment's fn-param ident for the (WeightId, index)
    /// pair that backs that field. Callers of `weight_accessor` that
    /// build the ident ad-hoc from a path still resolve correctly
    /// because the path→(WeightId,index) mapping is unique.
    pub fn weight_accessor(&self, name: &syn::Ident) -> TokenStream {
        // In Abstract mode, redirect to the fragment's by-name
        // override if this accessor name was declared as a fragment
        // param (matches every impl — fused or singleton — because
        // `weight_params_by_name` is populated from
        // `imp.required_weights().name`).
        if let EmitMode::Abstract {
            weight_params_by_name,
            ..
        } = &self.mode
            && let Some(param) = weight_params_by_name.get(&name.to_string())
        {
            return quote! { #param };
        }
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
            FufInput::Tile { id, slot } => {
                if let EmitMode::Abstract {
                    tile_params,
                    consumed_params,
                    ..
                } = &self.mode
                {
                    if let Some(param) = tile_params.get(&(*id, *slot)) {
                        return Some(param.clone());
                    }
                    if let Some(param) = consumed_params.get(&(*id, *slot)) {
                        return Some(param.clone());
                    }
                }
                self.locals.get(&(*id, *slot)).cloned()
            }
            _ => None,
        }
    }

    fn emit_input(&self, input: &FufInput) -> TokenStream {
        match input {
            FufInput::Tile { id, slot } => {
                if let EmitMode::Abstract {
                    tile_params,
                    consumed_params,
                    ..
                } = &self.mode
                {
                    // Boundary tile, viewed: fn-param ident is
                    // already `TensorView<'_>` at the boundary.
                    if let Some(param) = tile_params.get(&(*id, *slot)) {
                        return quote! { #param };
                    }
                    // Boundary tile, consumed: fn-param is
                    // `OwnedTensor` (by value). `input_expr` callers
                    // want a `TensorView`, so wrap with the same
                    // `(*x).as_view()` pattern the concrete path
                    // uses. Impls that need the raw owner
                    // (in-place kernels) use `input_tile_ident`
                    // instead — that path returns the param ident
                    // unwrapped.
                    if let Some(param) = consumed_params.get(&(*id, *slot)) {
                        return quote! { (*#param).as_view() };
                    }
                    // Internal tile input (producer IS a claimed
                    // tile earlier in the subgraph): emit the
                    // abstract output ident matching what
                    // `output_ident(producer, slot)` returned for
                    // that producer — the fragment's let-binding
                    // inside this same body.
                    if let Some(pos) = self.claimed_tiles.iter().position(|&t| t == *id) {
                        let ident = format_ident!("__out_{}_{}", pos, slot);
                        return quote! { (*#ident).as_view() };
                    }
                    // Fall-through: missing claimed producer. Emit
                    // a debug marker so the build fails loudly
                    // rather than silently dropping the edge.
                    return quote! { __missing_claimed_producer };
                }
                let ident = self
                    .locals
                    .get(&(*id, *slot))
                    .cloned()
                    .unwrap_or_else(|| format_ident!("__missing_tile_{}_{}", id.0, slot));
                quote! { (*#ident).as_view() }
            }
            FufInput::Weight { id, index, .. } => {
                if let EmitMode::Abstract { weight_params, .. } = &self.mode
                    && let Some(param) = weight_params.get(&(*id, *index))
                {
                    return quote! { #param };
                }
                let name = weight_field_name(self.program, *id, *index);
                quote! { wm.#name }
            }
            FufInput::Extern { kind, .. } => match kind {
                ExternKind::InputIds => quote! { ctx.input_ids },
                ExternKind::Positions => quote! { ctx.positions },
                ExternKind::Rotary => quote! { ctx.rotary },
                ExternKind::RotaryLocal => quote! { wm.rotary_local },
                ExternKind::BlockTable => quote! { ctx.block_table },
                ExternKind::KvCache => quote! { ctx.kv_cache },
            },
            FufInput::Scalar(v) => {
                // Emit as an `f32` literal — the only call sites
                // that consume a Scalar today are kernel-param
                // positions that expect `f32`. Wider types can be
                // handled by specialised emit paths when they show
                // up.
                let v = *v as f32;
                quote! { #v }
            }
        }
    }

    /// The let-binding ident holding `(tile, slot)`'s output.
    ///
    /// In [`EmitMode::Abstract`] mode this returns
    /// `__out_<claimed_pos>_<slot>` so two structurally identical
    /// emit bodies stringify identically regardless of the
    /// underlying tile id. `claimed_pos` is the tile's index inside
    /// the subgraph's `claimed_tiles` (position-based = stable
    /// across subgraphs with the same impl pick). The extra pos
    /// dimension prevents internal shadowing in multi-tile fused
    /// subgraphs (e.g. `FusedGateUpSiluMulImpl` claims 4 tiles
    /// whose individual outputs chain — each needs its own
    /// binding).
    pub fn output_ident(&self, tile: TileId, slot: u8) -> syn::Ident {
        if matches!(self.mode, EmitMode::Abstract { .. }) {
            let pos = self
                .claimed_tiles
                .iter()
                .position(|&t| t == tile)
                .unwrap_or(0);
            return format_ident!("__out_{}_{}", pos, slot);
        }
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
    let stem = program.weights.path(id).join("_");
    let ident = match index {
        Some(i) => format!("{stem}_{i}"),
        None => stem,
    };
    format_ident!("{}", ident)
}
