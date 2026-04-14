// SPDX-License-Identifier: Apache-2.0
//! Parser for the megakernel! DSL.
//!
//! Syntax:
//! ```ignore
//! megakernel! {
//!     kernel name<PARAM=val, ...> {
//!         for layer in 0..NL {
//!             let x = rmsnorm(input, weights[layer]);
//!             let y = gemm(x, w[layer]);
//!             ...
//!         }
//!         let z = rmsnorm(x, w);
//!         out = gemm(z, w);
//!     }
//! }
//! ```

use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Ident, LitInt, Result, Token, braced, token};

/// A compiled kernel variant with its own dimension overrides.
pub struct VariantDef {
    pub name: Ident,
    pub params: Vec<(Ident, usize)>,
}

/// Top-level: `kernel name<params> { body }` with optional `variants { ... }`.
pub struct MegakernelDef {
    pub name: Ident,
    pub params: Vec<(Ident, usize)>,
    pub body: Vec<Stmt>,
    pub variants: Vec<VariantDef>,
}

/// A statement in the kernel body.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let x = op(args...);`
    Let(LetStmt),
    /// `let (a, b, c) = op(args...);`  (destructuring)
    LetTuple(LetTupleStmt),
    /// `x = op(args...);`  (assignment to existing var, e.g. hidden_states)
    Assign(AssignStmt),
    /// `for layer in 0..NL { ... }`
    ForLoop(ForLoopStmt),
}

#[derive(Clone, Debug)]
pub struct LetStmt {
    pub name: Ident,
    pub call: OpCall,
}

#[derive(Clone, Debug)]
pub struct LetTupleStmt {
    pub names: Vec<Ident>,
    pub call: OpCall,
}

#[derive(Clone, Debug)]
pub struct AssignStmt {
    pub target: Ident,
    pub call: OpCall,
}

#[derive(Clone, Debug)]
pub struct ForLoopStmt {
    pub var: Ident,
    pub range_end: Ident, // e.g. NL
    pub body: Vec<Stmt>,
}

/// An op call: `op_name(arg1, arg2, ...)` or `op_name(arg1 * arg2, ...)`
#[derive(Clone, Debug)]
pub struct OpCall {
    pub op: Ident,
    pub args: Vec<Arg>,
}

/// An argument to an op call.
#[derive(Clone, Debug)]
pub enum Arg {
    /// Variable reference: `x`, `x[layer]`, or `self_attn.q_proj[layer]`.
    /// The name may contain dots (HF weight path segments).
    /// The optional `Ident` is the symbolic loop variable (pre-unroll).
    Var(String, Option<Ident>),
    /// Post-unroll: concrete indexed reference. `VarAt("foo", 3)`
    /// represents element 3 of the `foo` array.
    VarAt(String, usize),
    /// Nested call: `silu(gemm(x, w[layer]))`
    Call(OpCall),
    /// Binary expression: `a * b`
    Mul(Box<Arg>, Box<Arg>),
}

// ── syn Parse impls ──────────────────────────────────────────────────────

impl Parse for MegakernelDef {
    fn parse(input: ParseStream) -> Result<Self> {
        // `kernel`
        let kernel_kw: Ident = input.parse()?;
        if kernel_kw != "kernel" {
            return Err(syn::Error::new(kernel_kw.span(), "expected `kernel`"));
        }

        // name
        let name: Ident = input.parse()?;

        // <PARAM=val, ...>
        input.parse::<Token![<]>()?;
        let mut params = Vec::new();
        loop {
            if input.peek(Token![>]) {
                break;
            }
            let pname: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let pval: LitInt = input.parse()?;
            params.push((pname, pval.base10_parse::<usize>()?));
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        input.parse::<Token![>]>()?;

        // { body }
        let content;
        braced!(content in input);
        let body = parse_stmts(&content)?;

        // Optional: `variants { Name: { PARAM=val, ... }, ... }`
        let variants = if input.peek(Ident) {
            let kw: Ident = input.parse()?;
            if kw != "variants" {
                return Err(syn::Error::new(
                    kw.span(),
                    "expected `variants` or end of input",
                ));
            }
            let vcontent;
            braced!(vcontent in input);
            let mut variants = Vec::new();
            while !vcontent.is_empty() {
                let vname: Ident = vcontent.parse()?;
                vcontent.parse::<Token![:]>()?;
                let pcontent;
                braced!(pcontent in vcontent);
                let mut vparams = Vec::new();
                loop {
                    if pcontent.is_empty() {
                        break;
                    }
                    let pname: Ident = pcontent.parse()?;
                    pcontent.parse::<Token![=]>()?;
                    let pval: LitInt = pcontent.parse()?;
                    vparams.push((pname, pval.base10_parse::<usize>()?));
                    if pcontent.peek(Token![,]) {
                        pcontent.parse::<Token![,]>()?;
                    }
                }
                variants.push(VariantDef {
                    name: vname,
                    params: vparams,
                });
                if vcontent.peek(Token![,]) {
                    vcontent.parse::<Token![,]>()?;
                }
            }
            variants
        } else {
            Vec::new()
        };

        Ok(MegakernelDef {
            name,
            params,
            body,
            variants,
        })
    }
}

fn parse_stmts(input: ParseStream) -> Result<Vec<Stmt>> {
    let mut stmts = Vec::new();
    while !input.is_empty() {
        stmts.push(parse_stmt(input)?);
    }
    Ok(stmts)
}

fn parse_stmt(input: ParseStream) -> Result<Stmt> {
    if input.peek(Token![for]) {
        return parse_for_loop(input).map(Stmt::ForLoop);
    }
    if input.peek(Token![let]) {
        return parse_let(input);
    }
    // Assignment: ident = ...
    parse_assign(input).map(Stmt::Assign)
}

fn parse_let(input: ParseStream) -> Result<Stmt> {
    input.parse::<Token![let]>()?;

    // Check for tuple destructuring: let (a, b, c) = ...
    if input.peek(token::Paren) {
        let content;
        syn::parenthesized!(content in input);
        let names: Punctuated<Ident, Token![,]> =
            content.parse_terminated(Ident::parse, Token![,])?;
        input.parse::<Token![=]>()?;
        let call = parse_op_call(input)?;
        input.parse::<Token![;]>()?;
        return Ok(Stmt::LetTuple(LetTupleStmt {
            names: names.into_iter().collect(),
            call,
        }));
    }

    let name: Ident = input.parse()?;
    input.parse::<Token![=]>()?;
    let call = parse_op_call(input)?;
    input.parse::<Token![;]>()?;
    Ok(Stmt::Let(LetStmt { name, call }))
}

fn parse_assign(input: ParseStream) -> Result<AssignStmt> {
    let target: Ident = input.parse()?;
    input.parse::<Token![=]>()?;
    let call = parse_op_call(input)?;
    input.parse::<Token![;]>()?;
    Ok(AssignStmt { target, call })
}

fn parse_for_loop(input: ParseStream) -> Result<ForLoopStmt> {
    input.parse::<Token![for]>()?;
    let var: Ident = input.parse()?;
    // `in` is a keyword in Rust, so parse it as Token![in]
    input.parse::<Token![in]>()?;
    // 0..NL
    let _zero: LitInt = input.parse()?;
    input.parse::<Token![..]>()?;
    let range_end: Ident = input.parse()?;

    let content;
    braced!(content in input);
    let body = parse_stmts(&content)?;

    Ok(ForLoopStmt {
        var,
        range_end,
        body,
    })
}

fn parse_op_call(input: ParseStream) -> Result<OpCall> {
    let op: Ident = input.parse()?;
    let content;
    syn::parenthesized!(content in input);
    let mut args = Vec::new();
    while !content.is_empty() {
        args.push(parse_arg(&content)?);
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }
    Ok(OpCall { op, args })
}

fn parse_arg(input: ParseStream) -> Result<Arg> {
    // Could be: var, var[idx], nested_call(args), or expr * expr
    let first = parse_primary_arg(input)?;

    // Check for `*` (binary mul)
    if input.peek(Token![*]) {
        input.parse::<Token![*]>()?;
        let second = parse_primary_arg(input)?;
        return Ok(Arg::Mul(Box::new(first), Box::new(second)));
    }

    Ok(first)
}

fn parse_primary_arg(input: ParseStream) -> Result<Arg> {
    let ident: Ident = input.parse()?;

    // Nested call?
    if input.peek(token::Paren) {
        let content;
        syn::parenthesized!(content in input);
        let mut args = Vec::new();
        while !content.is_empty() {
            args.push(parse_arg(&content)?);
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
        return Ok(Arg::Call(OpCall { op: ident, args }));
    }

    // Dotted path? self_attn.q_proj or self_attn.q_proj[layer]
    // Join with '.' to form the full HF weight path segment.
    let mut name = ident.to_string();
    while input.peek(Token![.]) && !input.peek2(Token![.]) {
        input.parse::<Token![.]>()?;
        let next: Ident = input.parse()?;
        name.push('.');
        name.push_str(&next.to_string());
    }

    // Indexed? var[layer] or self_attn.q_proj[layer]
    if input.peek(token::Bracket) {
        let content;
        syn::bracketed!(content in input);
        let idx: Ident = content.parse()?;
        return Ok(Arg::Var(name, Some(idx)));
    }

    Ok(Arg::Var(name, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::TokenStream;

    #[test]
    fn parse_minimal_kernel() {
        let input: TokenStream = quote::quote! {
            kernel test_kernel<HD=2048, NL=16> {
                let x = rmsnorm(hidden_states, norm_w);
            }
        };

        let def: MegakernelDef = syn::parse2(input).expect("parse failed");
        assert_eq!(def.name.to_string(), "test_kernel");
        assert_eq!(def.params.len(), 2);
        assert_eq!(def.body.len(), 1);
    }

    #[test]
    fn parse_full_llama() {
        let input: TokenStream = quote::quote! {
            kernel llama_sm89<NL=16, HD=2048, ID=5632, HDM=64, NAH=32, NKH=8, VS=128256> {
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
                let normed = rmsnorm(hidden_states, lm_head_norm);
                logits = gemm(normed, lm_head);
            }
        };

        let def: MegakernelDef = syn::parse2(input).expect("parse failed");
        assert_eq!(def.name.to_string(), "llama_sm89");
        assert_eq!(def.params.len(), 7);
        // Body: 1 for-loop + 2 post-loop stmts
        assert_eq!(def.body.len(), 3);
    }

    #[test]
    fn parse_and_build_tile_graph() {
        // Regression: the full LLaMA DSL parses and lowers through
        // the canonical pipeline (parse → cfg → unroll → fuf) to a
        // tile graph with the expected number of per-layer tiles.
        let input: TokenStream = quote::quote! {
            kernel llama_sm89<NL=16, HD=2048, ID=5632, HDM=64, NAH=32, NKH=8, VS=128256> {
                hidden_states = embed(input_ids, embed_tokens);
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
                let final_norm = rmsnorm(hidden_states, norm);
                logits = gemm(final_norm, lm_head);
            }
        };

        let def: MegakernelDef = syn::parse2(input).expect("parse failed");
        let cfg = crate::cfg::build_cfg(&def);
        let tg = crate::fuf::build_fuf(&cfg, crate::lowering::tile_graph::ModelDims::LLAMA_3_2_1B)
            .expect("fuf build failed");

        // 1 embed + 17 × 16 layers + 2 post-loop = 275 tiles.
        assert_eq!(tg.num_layers, 16);
        assert_eq!(tg.nodes.len(), 1 + 17 * 16 + 2);
    }
}
