// SPDX-License-Identifier: Apache-2.0
//! DSL parser for the `compile!` macro.
//!
//! Parses the binding-mode DSL:
//!
//! ```ignore
//! compile! {
//!     model: llama_3_2_1b,
//!     target: l4_sm89,
//!     workloads: [1..1024],
//! }
//! ```
//!
//! Each field is either a concrete identifier (resolved at proc-macro
//! time) or the keyword `runtime` (deferred to model load).

use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitInt, Token, braced};

/// A binding that's either resolved at compile time or deferred to runtime.
#[derive(Clone, Debug)]
pub enum Binding<T> {
    /// Concrete value, resolved at proc-macro time.
    Static(T),
    /// Deferred to runtime — the proc macro emits code that resolves
    /// this at model load.
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

/// Known model identifiers. Each maps to a specific TileGraph
/// configuration (num_layers, hidden_size, etc.).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelId {
    Llama3_2_1B,
    Llama3_2_3B,
    Llama3_1_8B,
}

impl ModelId {
    pub fn num_layers(&self) -> u16 {
        match self {
            ModelId::Llama3_2_1B => 16,
            ModelId::Llama3_2_3B => 28,
            ModelId::Llama3_1_8B => 32,
        }
    }

    pub fn dims(&self) -> crate::lowering::tile_graph::ModelDims {
        use crate::lowering::tile_graph::ModelDims;
        match self {
            ModelId::Llama3_2_1B => ModelDims {
                hidden_size: 2048,
                intermediate_size: 8192,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dim: 64,
            },
            ModelId::Llama3_2_3B => ModelDims {
                hidden_size: 3072,
                intermediate_size: 8192,
                num_attention_heads: 24,
                num_kv_heads: 8,
                head_dim: 128,
            },
            ModelId::Llama3_1_8B => ModelDims {
                hidden_size: 4096,
                intermediate_size: 14336,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ModelId::Llama3_2_1B => "llama_3_2_1b",
            ModelId::Llama3_2_3B => "llama_3_2_3b",
            ModelId::Llama3_1_8B => "llama_3_1_8b",
        }
    }
}

/// Known target GPU identifiers. Each maps to a TargetProfile +
/// ImplementationLibrary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetId {
    L4Sm89,
    A100Sm80,
    H100Sm90,
}

impl TargetId {
    pub fn name(&self) -> &'static str {
        match self {
            TargetId::L4Sm89 => "l4_sm89",
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

/// Parsed `compile!` DSL.
#[derive(Clone, Debug)]
pub struct CompileDef {
    pub model: Binding<ModelId>,
    pub target: Binding<TargetId>,
    pub workloads: Binding<WorkloadRange>,
}

impl CompileDef {
    /// Whether all bindings are static (fully specialized mode).
    pub fn is_fully_specialized(&self) -> bool {
        self.model.is_static() && self.target.is_static() && self.workloads.is_static()
    }

    /// Whether only the model is static (GPU-specialized mode).
    pub fn is_gpu_specialized(&self) -> bool {
        self.model.is_static() && self.target.is_runtime()
    }

    /// Whether everything is runtime (fully dynamic mode).
    pub fn is_fully_dynamic(&self) -> bool {
        self.model.is_runtime() && self.target.is_runtime()
    }
}

fn parse_model_id(ident: &str) -> syn::Result<ModelId> {
    match ident {
        "llama_3_2_1b" => Ok(ModelId::Llama3_2_1B),
        "llama_3_2_3b" => Ok(ModelId::Llama3_2_3B),
        "llama_3_1_8b" => Ok(ModelId::Llama3_1_8B),
        other => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("unknown model: `{other}`. known: llama_3_2_1b, llama_3_2_3b, llama_3_1_8b"),
        )),
    }
}

fn parse_target_id(ident: &str) -> syn::Result<TargetId> {
    match ident {
        "l4_sm89" => Ok(TargetId::L4Sm89),
        "a100_sm80" => Ok(TargetId::A100Sm80),
        "h100_sm90" => Ok(TargetId::H100Sm90),
        other => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("unknown target: `{other}`. known: l4_sm89, a100_sm80, h100_sm90"),
        )),
    }
}

impl Parse for CompileDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        // Optional outer braces (compile! { ... } has them stripped by
        // the proc_macro entry, but if present, consume them).
        let has_braces = input.peek(syn::token::Brace);
        let inner = if has_braces {
            braced!(content in input);
            &content
        } else {
            input
        };

        let mut model: Option<Binding<ModelId>> = None;
        let mut target: Option<Binding<TargetId>> = None;
        let mut workloads: Option<Binding<WorkloadRange>> = None;

        while !inner.is_empty() {
            let key: Ident = inner.parse()?;
            inner.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "model" => {
                    let value: Ident = inner.parse()?;
                    let val_str = value.to_string();
                    model = Some(if val_str == "runtime" {
                        Binding::Runtime
                    } else {
                        Binding::Static(parse_model_id(&val_str)?)
                    });
                }
                "target" => {
                    let value: Ident = inner.parse()?;
                    let val_str = value.to_string();
                    target = Some(if val_str == "runtime" {
                        Binding::Runtime
                    } else {
                        Binding::Static(parse_target_id(&val_str)?)
                    });
                }
                "workloads" => {
                    let value: Option<String> =
                        inner.fork().parse().ok().map(|i: Ident| i.to_string());
                    if value.as_deref() == Some("runtime") {
                        let _: Ident = inner.parse()?;
                        workloads = Some(Binding::Runtime);
                    } else {
                        // Parse [min..max]
                        let bracket_content;
                        syn::bracketed!(bracket_content in inner);
                        let min: LitInt = bracket_content.parse()?;
                        bracket_content.parse::<Token![..]>()?;
                        let max: LitInt = bracket_content.parse()?;
                        workloads = Some(Binding::Static(WorkloadRange {
                            min_tokens: min.base10_parse()?,
                            max_tokens: max.base10_parse()?,
                        }));
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown field `{other}`. expected: model, target, workloads"),
                    ));
                }
            }

            // Consume optional trailing comma.
            let _ = inner.parse::<Token![,]>();
        }

        Ok(CompileDef {
            model: model.unwrap_or(Binding::Runtime),
            target: target.unwrap_or(Binding::Runtime),
            workloads: workloads.unwrap_or(Binding::Runtime),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fully_specialized() {
        let tokens: proc_macro2::TokenStream = "
            model: llama_3_2_1b,
            target: l4_sm89,
            workloads: [1..1024],
        "
        .parse()
        .unwrap();

        let def: CompileDef = syn::parse2(tokens).unwrap();
        assert!(def.is_fully_specialized());
        assert_eq!(def.model.as_static().unwrap(), &ModelId::Llama3_2_1B);
        assert_eq!(def.target.as_static().unwrap(), &TargetId::L4Sm89);
        let wl = def.workloads.as_static().unwrap();
        assert_eq!(wl.min_tokens, 1);
        assert_eq!(wl.max_tokens, 1024);
    }

    #[test]
    fn parse_gpu_specialized() {
        let tokens: proc_macro2::TokenStream = "
            model: llama_3_2_1b,
            target: runtime,
        "
        .parse()
        .unwrap();

        let def: CompileDef = syn::parse2(tokens).unwrap();
        assert!(def.is_gpu_specialized());
        assert!(def.target.is_runtime());
        assert!(def.workloads.is_runtime());
    }

    #[test]
    fn parse_fully_dynamic() {
        let tokens: proc_macro2::TokenStream = "
            model: runtime,
            target: runtime,
        "
        .parse()
        .unwrap();

        let def: CompileDef = syn::parse2(tokens).unwrap();
        assert!(def.is_fully_dynamic());
    }
}
