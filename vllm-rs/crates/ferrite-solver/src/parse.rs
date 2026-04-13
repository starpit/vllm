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

use crate::dag::*;
use std::collections::HashMap;
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

pub struct LetStmt {
    pub name: Ident,
    pub call: OpCall,
}

pub struct LetTupleStmt {
    pub names: Vec<Ident>,
    pub call: OpCall,
}

pub struct AssignStmt {
    pub target: Ident,
    pub call: OpCall,
}

pub struct ForLoopStmt {
    pub var: Ident,
    pub range_end: Ident, // e.g. NL
    pub body: Vec<Stmt>,
}

/// An op call: `op_name(arg1, arg2, ...)` or `op_name(arg1 * arg2, ...)`
pub struct OpCall {
    pub op: Ident,
    pub args: Vec<Arg>,
}

/// An argument to an op call.
pub enum Arg {
    /// Variable reference: `x`, `x[layer]`, or `self_attn.q_proj[layer]`.
    /// The name may contain dots (HF weight path segments).
    Var(String, Option<Ident>),
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

// ── DSL → DAG conversion ─────────────────────────────────────────────────

/// Convert a parsed MegakernelDef into a typed ModelDag.
/// This is where buffer declarations are inferred from op signatures.
pub fn build_dag(def: &MegakernelDef) -> std::result::Result<ModelDag, String> {
    let params: HashMap<String, usize> = def
        .params
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    let mut dag = ModelDag::new(def.name.to_string(), params);
    let mut ctx = BuildCtx::new(&dag.params);

    // Process the body
    for stmt in &def.body {
        process_stmt(&mut dag, &mut ctx, stmt, false)?;
    }

    Ok(dag)
}

struct BuildCtx {
    /// Maps variable names to buffer IDs and their shapes.
    vars: HashMap<String, (BufferId, TensorShape)>,
    /// Dimension params for resolving shapes.
    params: HashMap<String, usize>,
    /// Counter for generating unique intermediate buffer names.
    anon_counter: usize,
}

impl BuildCtx {
    fn new(params: &HashMap<String, usize>) -> Self {
        Self {
            vars: HashMap::new(),
            params: params.clone(),
            anon_counter: 0,
        }
    }

    fn fresh_name(&mut self, hint: &str) -> String {
        self.anon_counter += 1;
        format!("__{hint}_{}", self.anon_counter)
    }

    fn bs(&self) -> Dim {
        Dim::Param("BS".into())
    }
    fn hd(&self) -> Dim {
        Dim::Param("HD".into())
    }
    fn id(&self) -> Dim {
        Dim::Param("ID".into())
    }
    fn vs(&self) -> Dim {
        Dim::Param("VS".into())
    }

    fn dim_param(&self, name: &str) -> Dim {
        Dim::Param(name.into())
    }
}

fn process_stmt(
    dag: &mut ModelDag,
    ctx: &mut BuildCtx,
    stmt: &Stmt,
    in_loop: bool,
) -> std::result::Result<(), String> {
    match stmt {
        Stmt::Let(s) => {
            let (buf_id, shape) = process_call(dag, ctx, &s.call, in_loop)?;
            ctx.vars.insert(s.name.to_string(), (buf_id, shape));
            Ok(())
        }
        Stmt::LetTuple(s) => {
            // Only rope_append produces a tuple.
            let (buf_id, _shape) = process_call(dag, ctx, &s.call, in_loop)?;
            // The call already registered q, k, v buffers. Map names.
            // We need a convention: rope_append returns (q, k, v).
            // The q/k/v buffer IDs are generated inside process_call.
            // We'll look them up by the generated names.
            let base = buf_id.0.trim_start_matches("__rope_").to_string();
            for (i, name) in s.names.iter().enumerate() {
                let suffix = match i {
                    0 => "q",
                    1 => "k",
                    2 => "v",
                    _ => return Err("rope_append returns exactly 3 values".into()),
                };
                let child_id = BufferId(format!("{base}_{suffix}"));
                if let Some(buf) = dag.buffers.get(&child_id) {
                    ctx.vars
                        .insert(name.to_string(), (child_id, buf.shape.clone()));
                }
            }
            Ok(())
        }
        Stmt::Assign(s) => {
            let (buf_id, shape) = process_call(dag, ctx, &s.call, in_loop)?;
            ctx.vars.insert(s.target.to_string(), (buf_id, shape));
            Ok(())
        }
        Stmt::ForLoop(f) => {
            for stmt in &f.body {
                process_stmt(dag, ctx, stmt, true)?;
            }
            Ok(())
        }
    }
}

/// Process an op call, register buffers and ops, return the output buffer ID and shape.
fn process_call(
    dag: &mut ModelDag,
    ctx: &mut BuildCtx,
    call: &OpCall,
    in_loop: bool,
) -> std::result::Result<(BufferId, TensorShape), String> {
    let op_name = call.op.to_string();

    match op_name.as_str() {
        "embed" => {
            // embed(input_ids, embed_tokens) -> output[BS, HD]
            //
            // Produces the initial `hidden_states` activation at the
            // top of the forward pass. `embed_tokens` is a weight
            // (Embedding type); `input_ids` is a runtime input.
            let (ids_id, _) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let (w_id, _) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let out_shape = TensorShape::matrix(ctx.bs(), ctx.hd());
            let out_name = ctx.fresh_name("embed");
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: out_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::Embed {
                    input_ids: ids_id,
                    weights: w_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, out_shape))
        }
        "rmsnorm" => {
            // rmsnorm(input, weights) -> output[BS, D] where D = input's last dim
            let (input_id, input_shape) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let (weights_id, _) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let out_name = ctx.fresh_name("rmsnorm");
            let out_shape = input_shape.clone();
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: out_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::RmsNorm {
                    input: input_id,
                    weights: weights_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, out_shape))
        }
        "gemm" => {
            // gemm(A, B) -> output[A_rows, B_rows] (A @ B^T)
            let (a_id, a_shape) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let (b_id, b_shape) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let out_rows = a_shape
                .dims
                .get(a_shape.dims.len().wrapping_sub(2))
                .cloned()
                .unwrap_or(ctx.bs());
            let out_cols = b_shape.dims.first().cloned().unwrap_or(ctx.hd());
            let out_shape = TensorShape::matrix(out_rows, out_cols);
            let out_name = ctx.fresh_name("gemm");
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: out_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::Gemm {
                    a: a_id,
                    b: b_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, out_shape))
        }
        "gemm_add" => {
            // gemm_add(A, B, residual) -> output (same shape as residual)
            let (a_id, _a_shape) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let (b_id, _b_shape) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let (res_id, res_shape) = resolve_arg(dag, ctx, &call.args[2], in_loop)?;
            let out_shape = res_shape.clone();
            let out_name = ctx.fresh_name("gemm_add");
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: out_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::GemmAdd {
                    a: a_id,
                    b: b_id,
                    residual: res_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, out_shape))
        }
        "rope_append" => {
            // rope_append(q, k, v, positions, rotary, kv_cache) -> (q, k, v)
            //
            // Takes separate Q, K, V projections (unfused). The solver
            // may elect to fuse the upstream GEMMs, but the DSL
            // describes the logical math with separate projections.
            // `rotary` is the pre-computed cos/sin cache — a global
            // weight (not per-layer), classified as Rust type
            // `RotaryCache` in the field extractor.
            let (q_in_id, _) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let (k_in_id, _) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let (v_in_id, _) = resolve_arg(dag, ctx, &call.args[2], in_loop)?;
            let (pos_id, _) = resolve_arg(dag, ctx, &call.args[3], in_loop)?;
            let (rot_id, _) = resolve_arg(dag, ctx, &call.args[4], in_loop)?;
            let (kv_id, _) = resolve_arg(dag, ctx, &call.args[5], in_loop)?;

            let base = ctx.fresh_name("rope");
            let q_id = BufferId(format!("{base}_q"));
            let k_id = BufferId(format!("{base}_k"));
            let v_id = BufferId(format!("{base}_v"));

            // Q shape: [BS, NAH * HDM]
            let q_shape = TensorShape::matrix(ctx.bs(), ctx.hd());
            // K, V are opaque (go into cache)
            let kv_shape = TensorShape {
                dims: vec![Dim::Lit(1)],
            };

            for (id, shape) in [
                (q_id.clone(), q_shape.clone()),
                (k_id.clone(), kv_shape.clone()),
                (v_id.clone(), kv_shape.clone()),
            ] {
                dag.add_buffer(Buffer {
                    id,
                    kind: BufferKind::Activation,
                    shape,
                    producer: None,
                    consumers: vec![],
                    per_layer: in_loop,
                    is_input: false,
                });
            }

            dag.add_op(
                OpKind::RopeAppend {
                    q_in: q_in_id,
                    k_in: k_in_id,
                    v_in: v_in_id,
                    positions: pos_id,
                    rotary: rot_id,
                    kv_cache: kv_id,
                    q_out: q_id.clone(),
                    k_out: k_id,
                    v_out: v_id,
                },
                in_loop,
            );

            // Return the "rope" group ID — LetTuple will unpack q/k/v
            let group_id = BufferId(format!("__rope_{base}"));
            Ok((group_id, q_shape))
        }
        "attention" | "attention_decode" => {
            // attention(q, k, v, kv_cache, block_table) -> output[BS, HD]
            // The solver picks decode vs prefill based on seq_len.
            let (q_id, _) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            // k and v args are ignored (they're in the kv_cache)
            let (_k_id, _) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let (_v_id, _) = resolve_arg(dag, ctx, &call.args[2], in_loop)?;
            let (kv_id, _) = resolve_arg(dag, ctx, &call.args[3], in_loop)?;
            let (bt_id, _) = resolve_arg(dag, ctx, &call.args[4], in_loop)?;

            let out_shape = TensorShape::matrix(ctx.bs(), ctx.hd());
            let out_name = ctx.fresh_name("attn");
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: out_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::AttentionDecode {
                    q: q_id,
                    kv_cache: kv_id,
                    block_table: bt_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, out_shape))
        }
        "add" => {
            // add(a, b) -> output (same shape as a)
            let (a_id, a_shape) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let (b_id, _b_shape) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let out_name = ctx.fresh_name("add");
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: a_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::Add {
                    a: a_id,
                    b: b_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, a_shape))
        }
        "silu" => {
            // silu(x) -> output (same shape)
            // But x can be a nested call: silu(gemm(a, b))
            let (input_id, input_shape) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let out_name = ctx.fresh_name("silu");
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: input_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::Silu {
                    input: input_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, input_shape))
        }
        "bias_add" => {
            // bias_add(input, bias_weights) -> output (same shape as input)
            let (input_id, input_shape) = resolve_arg(dag, ctx, &call.args[0], in_loop)?;
            let (bias_id, _) = resolve_arg(dag, ctx, &call.args[1], in_loop)?;
            let out_name = ctx.fresh_name("bias_add");
            let out_shape = input_shape.clone();
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: out_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::BiasAdd {
                    input: input_id,
                    bias: bias_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, out_shape))
        }
        other => Err(format!("unknown op: {other}")),
    }
}

/// Resolve an argument to a (BufferId, TensorShape).
fn resolve_arg(
    dag: &mut ModelDag,
    ctx: &mut BuildCtx,
    arg: &Arg,
    in_loop: bool,
) -> std::result::Result<(BufferId, TensorShape), String> {
    match arg {
        Arg::Var(name, idx) => {
            let name = name.clone();
            // Check if it's a known variable
            if let Some((id, shape)) = ctx.vars.get(&name) {
                return Ok((id.clone(), shape.clone()));
            }
            // Otherwise it's an external input — infer its type from name convention.
            let (kind, shape) = infer_external_buffer(&name, ctx);
            let buf_id = BufferId(name.clone());
            if !dag.buffers.contains_key(&buf_id) {
                dag.add_buffer(Buffer {
                    id: buf_id.clone(),
                    kind,
                    shape: shape.clone(),
                    producer: None,
                    consumers: vec![],
                    // `per_layer` is solely determined by whether the
                    // DSL reference carries a `[layer]` index. A
                    // global weight (e.g. `rotary`) referenced from
                    // inside the layer loop is NOT per-layer.
                    per_layer: idx.is_some(),
                    is_input: true, // external buffer — provided by caller
                });
            }
            Ok((buf_id, shape))
        }
        Arg::Call(call) => process_call(dag, ctx, call, in_loop),
        Arg::Mul(a, b) => {
            let (a_id, a_shape) = resolve_arg(dag, ctx, a, in_loop)?;
            let (b_id, _b_shape) = resolve_arg(dag, ctx, b, in_loop)?;
            let out_name = ctx.fresh_name("mul");
            let out_id = BufferId(out_name);
            dag.add_buffer(Buffer {
                id: out_id.clone(),
                kind: BufferKind::Activation,
                shape: a_shape.clone(),
                producer: None,
                consumers: vec![],
                per_layer: in_loop,
                is_input: false,
            });
            dag.add_op(
                OpKind::Mul {
                    a: a_id,
                    b: b_id,
                    output: out_id.clone(),
                },
                in_loop,
            );
            Ok((out_id, a_shape))
        }
    }
}

/// Infer buffer kind and shape from naming conventions.
fn infer_external_buffer(name: &str, ctx: &BuildCtx) -> (BufferKind, TensorShape) {
    match name {
        "hidden_states" => (
            BufferKind::Activation,
            TensorShape::matrix(ctx.bs(), ctx.hd()),
        ),
        "input_ids" => (
            // Token IDs fed to the initial `embed(...)` op. Runtime
            // input, not a weight.
            BufferKind::Metadata,
            TensorShape {
                dims: vec![ctx.bs()],
            },
        ),
        "embed_tokens" => (
            // Embedding table — a global weight of Rust type `Embedding`.
            BufferKind::Weight,
            TensorShape::matrix(ctx.vs(), ctx.hd()),
        ),
        "rotary" => (
            // Pre-computed cos/sin cache — a global weight (Rust type
            // `RotaryCache`). Shape is runtime (depends on
            // max_position_embeddings), so use a 1-element placeholder.
            BufferKind::Weight,
            TensorShape {
                dims: vec![Dim::Lit(1)],
            },
        ),
        "positions" => (
            BufferKind::Metadata,
            TensorShape {
                dims: vec![ctx.bs()],
            },
        ),
        "kv_cache" => (
            BufferKind::KvCache,
            TensorShape {
                dims: vec![Dim::Lit(1)],
            },
        ),
        "block_table" => (
            BufferKind::Metadata,
            TensorShape {
                dims: vec![Dim::Lit(1)],
            },
        ),
        // Layernorms: input_layernorm, post_attention_layernorm, norm, *_norm
        n if n.contains("layernorm") || n.ends_with("_norm") || n == "norm" => (
            BufferKind::Weight,
            TensorShape {
                dims: vec![ctx.hd()],
            },
        ),
        "lm_head" | "lm_head_weights" => {
            (BufferKind::Weight, TensorShape::matrix(ctx.vs(), ctx.hd()))
        }
        // Bias vectors: self_attn.q_proj.bias, self_attn.k_proj.bias, etc.
        // Shape is [out_features] — a 1-D weight vector.
        n if n.ends_with(".bias") => {
            // Infer the bias dimension from the parent projection name.
            let parent = &n[..n.len() - ".bias".len()];
            let nah = ctx.params.get("NAH").copied().unwrap_or(32);
            let nkh = ctx.params.get("NKH").copied().unwrap_or(8);
            let hdm = ctx.params.get("HDM").copied().unwrap_or(64);
            let dim = if parent.contains("q_proj") {
                Dim::Lit(nah * hdm)
            } else if parent.contains("k_proj") || parent.contains("v_proj") {
                Dim::Lit(nkh * hdm)
            } else if parent.contains("gate_proj") || parent.contains("up_proj") {
                ctx.id()
            } else {
                // o_proj, down_proj, and anything else: hidden dim.
                ctx.hd()
            };
            (BufferKind::Weight, TensorShape { dims: vec![dim] })
        }
        // Q/K/V projections: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj
        n if n.contains("q_proj") || n.contains("k_proj") || n.contains("v_proj") => {
            let nah = ctx.params.get("NAH").copied().unwrap_or(32);
            let nkh = ctx.params.get("NKH").copied().unwrap_or(8);
            let hdm = ctx.params.get("HDM").copied().unwrap_or(64);
            let out_dim = if n.contains("q_proj") {
                nah * hdm
            } else {
                nkh * hdm
            };
            (
                BufferKind::Weight,
                TensorShape::matrix(Dim::Lit(out_dim), ctx.hd()),
            )
        }
        // O projection: self_attn.o_proj
        n if n.contains("o_proj") => (BufferKind::Weight, TensorShape::matrix(ctx.hd(), ctx.hd())),
        // MLP: gate_proj, up_proj → [intermediate, hidden]
        n if n.contains("gate_proj") || n.contains("up_proj") => {
            (BufferKind::Weight, TensorShape::matrix(ctx.id(), ctx.hd()))
        }
        // MLP: down_proj → [hidden, intermediate]
        n if n.contains("down_proj") => {
            (BufferKind::Weight, TensorShape::matrix(ctx.hd(), ctx.id()))
        }
        // Legacy names (backward compat with old DSL bodies)
        n if n.contains("gate") || n.contains("up") => {
            (BufferKind::Weight, TensorShape::matrix(ctx.id(), ctx.hd()))
        }
        n if n.contains("down") => (BufferKind::Weight, TensorShape::matrix(ctx.hd(), ctx.id())),
        n if n.contains("qkv") => {
            let nah = ctx.params.get("NAH").copied().unwrap_or(32);
            let nkh = ctx.params.get("NKH").copied().unwrap_or(8);
            let hdm = ctx.params.get("HDM").copied().unwrap_or(64);
            let qkv_dim = (nah + 2 * nkh) * hdm;
            (
                BufferKind::Weight,
                TensorShape::matrix(Dim::Lit(qkv_dim), ctx.hd()),
            )
        }
        n if n.contains("proj") => (BufferKind::Weight, TensorShape::matrix(ctx.hd(), ctx.hd())),
        // Default: assume activation [BS, HD]
        _ => (
            BufferKind::Activation,
            TensorShape::matrix(ctx.bs(), ctx.hd()),
        ),
    }
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
    fn parse_and_build_dag() {
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
        let dag = build_dag(&def).expect("DAG build failed");

        // Should have ops per the LLaMA pipeline:
        // rmsnorm, gemm(qkv), rope_append, attention, gemm(o_proj), add,
        // rmsnorm, gemm(gate)+silu, gemm(up), mul, gemm(down), add,
        // rmsnorm(lm_head), gemm(lm_head)
        // Note: silu(gemm(...)) creates 2 ops (gemm + silu)
        assert!(
            dag.ops.len() >= 12,
            "expected >= 12 ops, got {}",
            dag.ops.len()
        );
    }
}
