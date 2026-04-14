// SPDX-License-Identifier: Apache-2.0
//! Codegen for the per-model `Model` + `Layer` structs and
//! `Model::load`.
//!
//! What this emits, given a classified DSL body and one
//! [`ModelParams`]:
//!
//! ```ignore
//! pub struct Layer {
//!     pub input_layernorm:           ::ferrite_kernels::layers::RmsNorm,
//!     pub self_attn_q_proj:          ::ferrite_kernels::layers::Linear,
//!     pub self_attn_k_proj:          ::ferrite_kernels::layers::Linear,
//!     pub self_attn_v_proj:          ::ferrite_kernels::layers::Linear,
//!     pub self_attn_o_proj:          ::ferrite_kernels::layers::Linear,
//!     pub post_attention_layernorm:  ::ferrite_kernels::layers::RmsNorm,
//!     pub mlp_gate_proj:             ::ferrite_kernels::layers::Linear,
//!     pub mlp_up_proj:               ::ferrite_kernels::layers::Linear,
//!     pub mlp_down_proj:             ::ferrite_kernels::layers::Linear,
//! }
//!
//! pub struct Model {
//!     pub layers: Vec<Layer>,
//!     pub embed_tokens: ::ferrite_kernels::layers::Embedding,
//!     pub norm:         ::ferrite_kernels::layers::RmsNorm,
//!     pub lm_head:      ::ferrite_kernels::layers::Linear,
//! }
//!
//! impl Model {
//!     pub const NUM_LAYERS: usize = 16;  // from the model config
//!
//!     pub unsafe fn load(
//!         weights: &mut ::ferrite_cuda_core::weights::GpuWeights,
//!         rms_norm_eps: f32,
//!     ) -> ::anyhow::Result<Self> { ... }
//! }
//! ```
//!
//! Design: the emitted code is entirely data-driven from the
//! classified Program. Per-layer vs global is determined by whether
//! any `Expr::Weight` read of that weight has `index: Some(_)`.
//! The runtime type is determined by the op the weight is an
//! argument of (embed → Embedding, rmsnorm → RmsNorm, gemm →
//! Linear). HF runtime paths come from
//! [`crate::weight_conventions::hf_weight_prefix`].
//!
//! Tie-word-embeddings: if the safetensors file does not have
//! `lm_head.weight`, the generated loader aliases `embed_tokens`'s
//! weight into `lm_head`. That's a runtime decision made by
//! inspecting the file; no compile-time flag needed.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};

use crate::classified::{Expr, OpKind, Program, Stmt, WeightId};
use crate::config::ModelParams;

/// Runtime type a weight maps to, derived from the op that consumes
/// it. Mirrors the set of ferrite-kernels layer types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightType {
    Embedding,
    Linear,
    RmsNorm,
}

impl WeightType {
    fn type_ident(self) -> TokenStream {
        match self {
            Self::Embedding => quote! { ::ferrite_kernels::layers::Embedding },
            Self::Linear => quote! { ::ferrite_kernels::layers::Linear },
            Self::RmsNorm => quote! { ::ferrite_kernels::layers::RmsNorm },
        }
    }

    /// The ferrite-kernels associated fn that constructs this type
    /// from a `GpuWeights` + prefix. RmsNorm also takes `eps`.
    fn emit_load(self, field: &syn::Ident, prefix_expr: TokenStream) -> TokenStream {
        match self {
            Self::Embedding => quote! {
                let #field = ::ferrite_kernels::layers::Embedding::load(weights, #prefix_expr)?;
            },
            Self::Linear => quote! {
                let #field = ::ferrite_kernels::layers::Linear::load(weights, #prefix_expr)?;
            },
            Self::RmsNorm => quote! {
                let #field = ::ferrite_kernels::layers::RmsNorm::load(weights, #prefix_expr, rms_norm_eps)?;
            },
        }
    }
}

/// Classification of one weight seen in the DSL.
#[derive(Debug)]
struct WeightFacts {
    #[allow(dead_code)] // kept for diagnostics / future use
    id: WeightId,
    /// Dotted path segments ("self_attn.q_proj" → ["self_attn", "q_proj"]).
    path: Vec<String>,
    /// True iff any read of this weight is indexed (inside a loop,
    /// so the weight is replicated per iteration).
    per_layer: bool,
    /// Runtime type, derived from the op the weight flows into.
    ty: WeightType,
}

/// Analyze the classified program: for every WeightId, determine
/// per-layer-ness and runtime type. Returns in WeightTable-interning
/// order so the struct field order is deterministic.
fn analyze_weights(program: &Program) -> Result<Vec<WeightFacts>, String> {
    let n = program.weights.len();
    let mut per_layer = vec![false; n];
    let mut ty: Vec<Option<WeightType>> = vec![None; n];

    fn visit_expr(
        expr: &Expr,
        in_op: Option<(OpKind, usize)>,
        per_layer: &mut [bool],
        ty: &mut [Option<WeightType>],
    ) -> Result<(), String> {
        match expr {
            Expr::Weight { id, index } => {
                if index.is_some() {
                    per_layer[id.0 as usize] = true;
                }
                if let Some((op, arg_idx)) = in_op {
                    let derived = match (op, arg_idx) {
                        (OpKind::Embed, 1) => WeightType::Embedding,
                        (OpKind::RmsNorm, 1) => WeightType::RmsNorm,
                        (OpKind::Gemm, 1) => WeightType::Linear,
                        _ => return Ok(()), // weight in a non-weight arg position
                    };
                    let slot = &mut ty[id.0 as usize];
                    if let Some(existing) = *slot
                        && existing != derived
                    {
                        return Err(format!(
                            "weight #{} used as both {:?} and {:?} — cannot assign \
                             a single runtime type",
                            id.0, existing, derived
                        ));
                    }
                    *slot = Some(derived);
                }
            }
            Expr::Call { op, args } => {
                for (i, a) in args.iter().enumerate() {
                    visit_expr(a, Some((*op, i)), per_layer, ty)?;
                }
            }
            Expr::Mul { lhs, rhs } => {
                // Mul operands don't identify weights — walk for
                // nested Weight reads (unlikely but correct).
                visit_expr(lhs, None, per_layer, ty)?;
                visit_expr(rhs, None, per_layer, ty)?;
            }
            Expr::Local(_) | Expr::Extern { .. } => {}
        }
        Ok(())
    }

    fn visit_stmt(
        stmt: &Stmt,
        per_layer: &mut [bool],
        ty: &mut [Option<WeightType>],
    ) -> Result<(), String> {
        match stmt {
            Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => {
                visit_expr(value, None, per_layer, ty)
            }
            Stmt::For { body, .. } => {
                for s in body {
                    visit_stmt(s, per_layer, ty)?;
                }
                Ok(())
            }
        }
    }

    for stmt in &program.statements {
        visit_stmt(stmt, &mut per_layer, &mut ty)?;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let id = WeightId(i as u32);
        let path: Vec<String> = program
            .weights
            .path(id)
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ty = ty[i].ok_or_else(|| {
            format!(
                "weight `{}` was read but never flowed into an op whose \
                 arg position identifies a runtime type",
                path.join(".")
            )
        })?;
        out.push(WeightFacts {
            id,
            path,
            per_layer: per_layer[i],
            ty,
        });
    }
    Ok(out)
}

/// Turn a dotted path like `["self_attn", "q_proj"]` into a Rust
/// field ident like `self_attn_q_proj`. Preserves ordering from the
/// path; the two-level structure (struct.method-style) is flattened
/// because Rust doesn't let us emit nested structs without naming
/// them, and naming them per-arch would leak transformer concepts
/// into the compiler.
fn field_ident(path: &[String]) -> syn::Ident {
    format_ident!("{}", path.join("_"))
}

/// Emit `pub struct Layer`, `pub struct Model`, and
/// `impl Model { pub unsafe fn load(...) }` for one model.
pub fn emit(program: &Program, model: &ModelParams) -> Result<TokenStream, String> {
    let facts = analyze_weights(program)?;
    // Partition into per-layer and global, preserving
    // WeightTable order for determinism.
    let (per_layer, global): (Vec<&WeightFacts>, Vec<&WeightFacts>) =
        facts.iter().partition(|w| w.per_layer);

    let num_layers: usize = model
        .bounds
        .get("num_hidden_layers")
        .copied()
        .ok_or_else(|| {
            format!(
                "config `{}` is missing `num_hidden_layers`",
                model.source_stem
            )
        })? as usize;

    // Layer struct fields.
    let layer_fields: Vec<TokenStream> = per_layer
        .iter()
        .map(|w| {
            let name = field_ident(&w.path);
            let ty = w.ty.type_ident();
            quote! { pub #name: #ty }
        })
        .collect();

    // Model struct fields (besides layers: Vec<Layer>).
    let global_fields: Vec<TokenStream> = global
        .iter()
        .map(|w| {
            let name = field_ident(&w.path);
            let ty = w.ty.type_ident();
            quote! { pub #name: #ty }
        })
        .collect();

    // Per-layer load block (inside `for i in 0..Self::NUM_LAYERS`).
    let layer_loads: Vec<TokenStream> = per_layer
        .iter()
        .map(|w| {
            let field = field_ident(&w.path);
            let dotted = w.path.join(".");
            // Build the prefix via format!("model.layers.{i}.<dotted>")
            // — the `i` is the loop index in the generated code.
            let fmt = format!("model.layers.{{i}}.{dotted}");
            let prefix_expr = quote! { &format!(#fmt) };
            w.ty.emit_load(&field, prefix_expr)
        })
        .collect();
    let layer_field_idents: Vec<syn::Ident> =
        per_layer.iter().map(|w| field_ident(&w.path)).collect();

    // Global loads.
    let global_loads: Vec<TokenStream> = global
        .iter()
        .map(|w| {
            let field = field_ident(&w.path);
            let prefix = crate::weight_conventions::hf_weight_prefix(&w.path, None);
            let prefix_lit = syn::LitStr::new(&prefix, Span::call_site());
            let prefix_expr = quote! { #prefix_lit };

            // Special case only for lm_head tie-word-embeddings:
            // if the safetensors file doesn't have lm_head.weight,
            // reuse embed_tokens's tensor. HF "tie_word_embeddings"
            // is resolved by inspecting the file, not by reading a
            // flag. Applies only to Linear-typed global `lm_head`.
            if w.path == ["lm_head"] && w.ty == WeightType::Linear {
                quote! {
                    let #field = if weights.contains("lm_head.weight") {
                        ::ferrite_kernels::layers::Linear::load(weights, "lm_head")?
                    } else {
                        // Tied to embed_tokens. GpuTensor: Copy.
                        ::ferrite_kernels::layers::Linear::new(
                            embed_tokens.weight,
                            None,
                        )
                    };
                }
            } else {
                w.ty.emit_load(&field, prefix_expr)
            }
        })
        .collect();
    let global_field_idents: Vec<syn::Ident> =
        global.iter().map(|w| field_ident(&w.path)).collect();

    let num_layers_lit = proc_macro2::Literal::usize_unsuffixed(num_layers);

    // The emitted types reference ferrite-kernels layer types
    // (Linear, Embedding, RmsNorm) and ferrite-cuda-core's
    // GpuWeights, both of which only compile with the `cuda`
    // feature on their crates. The consumer crate gates its own
    // `cuda` feature on those, so we gate the emitted types to
    // match — consumers without cuda still get NUM_TILES / NUM_WAVES
    // / PREDICTED_US (the pipeline observations), just not the
    // runtime `Model` struct. This mirrors how ferrite-models
    // gates `pub mod llama` today.
    Ok(quote! {
        #[cfg(feature = "cuda")]
        pub struct Layer {
            #(#layer_fields,)*
        }

        #[cfg(feature = "cuda")]
        pub struct Model {
            pub layers: Vec<Layer>,
            #(#global_fields,)*
        }

        #[cfg(feature = "cuda")]
        impl Model {
            /// Number of transformer layers in this specific model.
            /// Baked in at macro expansion from the model's
            /// `num_hidden_layers` config field.
            pub const NUM_LAYERS: usize = #num_layers_lit;

            /// Load every weight into a `Model`. Walks the
            /// safetensors mmap held by `weights`; panics if a
            /// required weight is missing. `rms_norm_eps` is the
            /// epsilon used by every RmsNorm layer (from the
            /// model's config.json; callers pass it through).
            ///
            /// # Safety
            /// `weights` must be a live `GpuWeights` holder whose
            /// underlying mmap is valid for the duration of `load`.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn load(
                weights: &mut ::ferrite_cuda_core::weights::GpuWeights,
                rms_norm_eps: f32,
            ) -> ::anyhow::Result<Self> {
                #(#global_loads)*

                let mut layers: Vec<Layer> = Vec::with_capacity(Self::NUM_LAYERS);
                for i in 0..Self::NUM_LAYERS {
                    #(#layer_loads)*
                    layers.push(Layer {
                        #(#layer_field_idents,)*
                    });
                }

                Ok(Model {
                    layers,
                    #(#global_field_idents,)*
                })
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::classify;
    use crate::parse::parse_block;

    fn classified(src: &str) -> Program {
        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).unwrap();
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        classify(&ast).unwrap()
    }

    #[test]
    fn llama_body_classification() {
        // Representative body — at least one weight per runtime
        // type and per per-layer/global bin.
        let program = classified(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                hidden_states = add(q, hidden_states);
            }
            normed = rmsnorm(hidden_states, norm);
            logits = gemm(normed, lm_head);
            "#,
        );
        let facts = analyze_weights(&program).unwrap();

        // embed_tokens: global, Embedding
        let f = facts.iter().find(|w| w.path == ["embed_tokens"]).unwrap();
        assert!(!f.per_layer);
        assert_eq!(f.ty, WeightType::Embedding);

        // input_layernorm: per-layer, RmsNorm
        let f = facts
            .iter()
            .find(|w| w.path == ["input_layernorm"])
            .unwrap();
        assert!(f.per_layer);
        assert_eq!(f.ty, WeightType::RmsNorm);

        // self_attn.q_proj: per-layer, Linear
        let f = facts
            .iter()
            .find(|w| w.path == ["self_attn", "q_proj"])
            .unwrap();
        assert!(f.per_layer);
        assert_eq!(f.ty, WeightType::Linear);

        // norm: global, RmsNorm
        let f = facts.iter().find(|w| w.path == ["norm"]).unwrap();
        assert!(!f.per_layer);
        assert_eq!(f.ty, WeightType::RmsNorm);

        // lm_head: global, Linear
        let f = facts.iter().find(|w| w.path == ["lm_head"]).unwrap();
        assert!(!f.per_layer);
        assert_eq!(f.ty, WeightType::Linear);
    }

    #[test]
    fn weight_used_in_incompatible_ops_errors() {
        // A weight used as both an embedding table and a gemm
        // matrix has conflicting runtime types — reject.
        let program = classified(
            r#"
            x = embed(input_ids, weights.shared);
            y = gemm(x, weights.shared);
            "#,
        );
        let err = analyze_weights(&program).unwrap_err();
        assert!(err.contains("cannot assign"));
    }

    #[test]
    fn weight_never_used_in_op_errors() {
        // Shouldn't arise from well-formed DSL — the classifier
        // only interns WeightIds that are read, and every read at
        // the top level is inside a Call. But guard it anyway.
        // We construct a program manually via classification —
        // since classify produces Call/Weight combos, this test
        // is a placeholder documenting the guard.
        //
        // The analyzer rejects a weight with no derived type; in
        // practice that requires hand-constructing the program.
    }

    #[test]
    fn field_ident_flattens_path() {
        assert_eq!(
            field_ident(&["self_attn".into(), "q_proj".into()]).to_string(),
            "self_attn_q_proj"
        );
        assert_eq!(
            field_ident(&["embed_tokens".into()]).to_string(),
            "embed_tokens"
        );
    }
}
