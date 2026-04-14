// SPDX-License-Identifier: Apache-2.0
//! DSL parser for the `forward!` macro.
//!
//! Syntax:
//!
//! ```ignore
//! forward! {
//!     // Body: the model's forward pass structure (required, always first).
//!     for layer in 0..NL {
//!         let normed = rmsnorm(hidden_states, input_layernorm[layer]);
//!         let q = gemm(normed, self_attn.q_proj[layer]);
//!         let k = gemm(normed, self_attn.k_proj[layer]);
//!         let v = gemm(normed, self_attn.v_proj[layer]);
//!         ...
//!     }
//!
//!     // Optional fields after the body:
//!     models: [
//!         { layers: 28, hidden: 3072, intermediate: 8192, heads: 24, kv_heads: 8, head_dim: 128 },
//!     ],
//!     target: l4_sm89,
//!     workloads: [1..4096],
//! }
//! ```
//!
//! If `models:` is omitted, defaults to runtime (solver at startup).
//! If `target:` is omitted, defaults to runtime (GPU detection at startup).
//! If `workloads:` is omitted, uses the default grid.

use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitInt, Token, braced};

use crate::lowering::tile_graph::ModelDims;

/// A binding that's either resolved at compile time or deferred to runtime.
#[derive(Clone, Debug)]
pub enum Binding<T> {
    Static(T),
    Runtime,
}

impl<T> Binding<T> {
    pub fn is_static(&self) -> bool {
        matches!(self, Binding::Static(_))
    }

    pub fn is_runtime(&self) -> bool {
        matches!(self, Binding::Runtime)
    }

    pub fn as_static(&self) -> Option<&T> {
        match self {
            Binding::Static(v) => Some(v),
            Binding::Runtime => None,
        }
    }
}

/// One inline model specification.
#[derive(Clone, Debug)]
pub struct InlineModel {
    pub num_layers: u16,
    pub dims: ModelDims,
}

/// Known target GPU identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetId {
    L4Sm89,
    L40sSm89,
    A100Sm80,
    H100Sm90,
}

impl TargetId {
    pub fn name(&self) -> &'static str {
        match self {
            TargetId::L4Sm89 => "l4_sm89",
            TargetId::L40sSm89 => "l40s_sm89",
            TargetId::A100Sm80 => "a100_sm80",
            TargetId::H100Sm90 => "h100_sm90",
        }
    }
}

/// Workload range specification.
#[derive(Clone, Debug)]
pub struct WorkloadRange {
    pub min_tokens: u32,
    pub max_tokens: u32,
}

/// Parsed `forward!` DSL.
#[derive(Clone, Debug)]
pub struct ForwardDef {
    /// Control-flow graph for the DSL body. The pipeline runs
    /// `fuf::build_fuf(&cfg, dims)` on this to produce the
    /// solver's `TileGraph`, and `fuf::extract_weight_fields(&cfg)`
    /// for the generated `Layer`/`Model` struct fields.
    pub cfg: crate::cfg::Cfg,
    /// Model dimensions to compile in. Multiple = multi-model binary.
    /// Runtime = solver runs at startup with dims from loaded weights.
    pub models: Binding<Vec<InlineModel>>,
    /// Target GPU.
    pub target: Binding<TargetId>,
    /// Workload grid. None = use default.
    pub workloads: Option<WorkloadRange>,
}

impl ForwardDef {
    pub fn is_fully_specialized(&self) -> bool {
        self.models.is_static() && self.target.is_static()
    }
}

impl Parse for ForwardDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // ── Step 1: Parse the body as megakernel DSL ──
        //
        // The body is everything up to the first `models:`, `target:`,
        // or `workloads:` key. We wrap it in a synthetic `kernel __forward<> { ... }`
        // so the existing MegakernelDef parser can handle it.
        //
        // Collect all tokens until we see `ident :` at the statement level
        // where ident is one of our known keys.
        let mut body_tokens = proc_macro2::TokenStream::new();
        while !input.is_empty() {
            // Peek: is this a key-value field?
            if is_field_key(input) {
                break;
            }
            // Consume one token tree (statement, block, etc.)
            let tt: proc_macro2::TokenTree = input.parse()?;
            body_tokens.extend(std::iter::once(tt));
        }

        // Wrap in `kernel __forward<NL=1, HD=1, ID=1, HDM=1, NAH=1, NKH=1, VS=1> { ... }`
        // with dummy params. NL will be overridden per-model.
        let wrapped: proc_macro2::TokenStream = format!(
            "kernel __forward<NL=1, HD=1, ID=1, HDM=1, NAH=1, NKH=1, VS=1> {{ {} }}",
            body_tokens
        )
        .parse()
        .map_err(|e| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("body parse error: {e}"),
            )
        })?;

        let def: crate::parse::MegakernelDef = syn::parse2(wrapped)?;
        // AST → CFG. Consumed by the FUF codegen path via
        // `fuf::build_fuf(&cfg, dims)`. Cheap to build; we do it
        // eagerly so downstream callers don't have to hold onto
        // the `MegakernelDef`.
        let cfg = crate::cfg::build_cfg(&def);

        // ── Step 2: Parse optional key-value fields ──
        let mut models: Option<Binding<Vec<InlineModel>>> = None;
        let mut target: Option<Binding<TargetId>> = None;
        let mut workloads: Option<WorkloadRange> = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "models" => {
                    if input.peek(Ident) {
                        let val: Ident = input.parse()?;
                        if val == "runtime" {
                            models = Some(Binding::Runtime);
                        } else {
                            return Err(syn::Error::new(
                                val.span(),
                                format!("expected `runtime` or `[...]`, got `{val}`"),
                            ));
                        }
                    } else {
                        // Parse [ { ... }, { ... }, ... ]
                        let bracket_content;
                        syn::bracketed!(bracket_content in input);
                        let mut model_list = Vec::new();
                        while !bracket_content.is_empty() {
                            let model_content;
                            braced!(model_content in bracket_content);
                            model_list.push(parse_inline_model(&model_content)?);
                            let _ = bracket_content.parse::<Token![,]>();
                        }
                        models = Some(Binding::Static(model_list));
                    }
                }
                "target" => {
                    let val: Ident = input.parse()?;
                    let val_str = val.to_string();
                    target = Some(if val_str == "runtime" {
                        Binding::Runtime
                    } else {
                        Binding::Static(parse_target_id(&val_str)?)
                    });
                }
                "workloads" => {
                    let bracket_content;
                    syn::bracketed!(bracket_content in input);
                    let min: LitInt = bracket_content.parse()?;
                    bracket_content.parse::<Token![..]>()?;
                    let max: LitInt = bracket_content.parse()?;
                    workloads = Some(WorkloadRange {
                        min_tokens: min.base10_parse()?,
                        max_tokens: max.base10_parse()?,
                    });
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown field `{other}`. expected: models, target, workloads"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }

        Ok(ForwardDef {
            cfg,
            models: models.unwrap_or(Binding::Runtime),
            target: target.unwrap_or(Binding::Runtime),
            workloads,
        })
    }
}

/// Check if the next tokens look like a field key (`models:`, `target:`, `workloads:`).
fn is_field_key(input: ParseStream) -> bool {
    let fork = input.fork();
    if let Ok(ident) = fork.parse::<Ident>() {
        let name = ident.to_string();
        return fork.peek(Token![:])
            && (name == "models" || name == "target" || name == "workloads");
    }
    false
}

fn parse_inline_model(input: ParseStream) -> syn::Result<InlineModel> {
    let mut layers: Option<u16> = None;
    let mut hidden: Option<u32> = None;
    let mut intermediate: Option<u32> = None;
    let mut heads: Option<u32> = None;
    let mut kv_heads: Option<u32> = None;
    let mut head_dim: Option<u32> = None;
    let mut vocab: Option<u32> = None;

    while !input.is_empty() {
        let key: Ident = input.parse()?;
        input.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "layers" => {
                let val: LitInt = input.parse()?;
                layers = Some(val.base10_parse()?);
            }
            "hidden" => {
                let val: LitInt = input.parse()?;
                hidden = Some(val.base10_parse()?);
            }
            "intermediate" => {
                let val: LitInt = input.parse()?;
                intermediate = Some(val.base10_parse()?);
            }
            "heads" => {
                let val: LitInt = input.parse()?;
                heads = Some(val.base10_parse()?);
            }
            "kv_heads" => {
                let val: LitInt = input.parse()?;
                kv_heads = Some(val.base10_parse()?);
            }
            "head_dim" => {
                let val: LitInt = input.parse()?;
                head_dim = Some(val.base10_parse()?);
            }
            "vocab" => {
                let val: LitInt = input.parse()?;
                vocab = Some(val.base10_parse()?);
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown model field `{other}`"),
                ));
            }
        }
        let _ = input.parse::<Token![,]>();
    }

    Ok(InlineModel {
        num_layers: layers.ok_or_else(|| syn::Error::new(input.span(), "missing `layers`"))?,
        dims: ModelDims {
            hidden_size: hidden.ok_or_else(|| syn::Error::new(input.span(), "missing `hidden`"))?,
            intermediate_size: intermediate
                .ok_or_else(|| syn::Error::new(input.span(), "missing `intermediate`"))?,
            num_attention_heads: heads
                .ok_or_else(|| syn::Error::new(input.span(), "missing `heads`"))?,
            num_kv_heads: kv_heads
                .ok_or_else(|| syn::Error::new(input.span(), "missing `kv_heads`"))?,
            head_dim: head_dim
                .ok_or_else(|| syn::Error::new(input.span(), "missing `head_dim`"))?,
            // Default to Llama 3 vocab if not specified — the solver still
            // compiles without this; it's only used by the lm_head cost lookup.
            vocab_size: vocab.unwrap_or(128256),
        },
    })
}

fn parse_target_id(ident: &str) -> syn::Result<TargetId> {
    match ident {
        "l4_sm89" => Ok(TargetId::L4Sm89),
        "l40s_sm89" => Ok(TargetId::L40sSm89),
        "a100_sm80" => Ok(TargetId::A100Sm80),
        "h100_sm90" => Ok(TargetId::H100Sm90),
        other => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("unknown target: `{other}`. known: l4_sm89, a100_sm80, h100_sm90"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_body_only() {
        let tokens: proc_macro2::TokenStream = r#"
            for layer in 0..NL {
                let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                let attn = attention(q, k, v, kv_cache[layer], block_table);
                let oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                let up = gemm(normed2, mlp.up_proj[layer]);
                let down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }
        "#
        .parse()
        .unwrap();

        let def: ForwardDef = syn::parse2(tokens).unwrap();
        assert!(def.models.is_runtime());
        assert!(def.target.is_runtime());
        assert!(def.workloads.is_none());
        // Body parsed into a non-empty CFG with the DSL's block structure.
        assert!(!def.cfg.blocks.is_empty());
        // There must be at least one instruction in some block (the
        // DSL wasn't completely empty).
        assert!(def.cfg.blocks.iter().any(|b| !b.instrs.is_empty()));
    }

    #[test]
    fn parse_body_with_models_and_target() {
        let tokens: proc_macro2::TokenStream = r#"
            for layer in 0..NL {
                let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                let attn = attention(q, k, v, kv_cache[layer], block_table);
                let oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                let up = gemm(normed2, mlp.up_proj[layer]);
                let down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }

            models: [
                { layers: 28, hidden: 3072, intermediate: 8192, heads: 24, kv_heads: 8, head_dim: 128 },
            ],
            target: l4_sm89,
            workloads: [1..4096],
        "#
        .parse()
        .unwrap();

        let def: ForwardDef = syn::parse2(tokens).unwrap();
        assert!(def.is_fully_specialized());
        let models = def.models.as_static().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].num_layers, 28);
        assert_eq!(models[0].dims.hidden_size, 3072);
        assert_eq!(def.target.as_static().unwrap(), &TargetId::L4Sm89);
        assert_eq!(def.workloads.as_ref().unwrap().max_tokens, 4096);
    }

    #[test]
    fn parse_multiple_models() {
        let tokens: proc_macro2::TokenStream = r#"
            for layer in 0..NL {
                let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                let attn = attention(q, k, v, kv_cache[layer], block_table);
                let oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                let up = gemm(normed2, mlp.up_proj[layer]);
                let down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }

            models: [
                { layers: 16, hidden: 2048, intermediate: 8192, heads: 32, kv_heads: 8, head_dim: 64 },
                { layers: 28, hidden: 3072, intermediate: 8192, heads: 24, kv_heads: 8, head_dim: 128 },
            ],
            target: l4_sm89,
        "#
        .parse()
        .unwrap();

        let def: ForwardDef = syn::parse2(tokens).unwrap();
        let models = def.models.as_static().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].dims.hidden_size, 2048);
        assert_eq!(models[1].dims.hidden_size, 3072);
    }

    #[test]
    fn parse_runtime_models() {
        let tokens: proc_macro2::TokenStream = r#"
            for layer in 0..NL {
                let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                let attn = attention(q, k, v, kv_cache[layer], block_table);
                let oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                let up = gemm(normed2, mlp.up_proj[layer]);
                let down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }

            models: runtime,
            target: l4_sm89,
        "#
        .parse()
        .unwrap();

        let def: ForwardDef = syn::parse2(tokens).unwrap();
        assert!(def.models.is_runtime());
        assert!(def.target.is_static());
    }
}
