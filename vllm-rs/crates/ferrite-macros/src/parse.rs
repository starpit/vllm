use crate::ops::{Edge, InputPort, OpGraph, OpKind, OpNode, PARAM, ParamInfo};
use syn::{self, Expr, ItemFn, Stmt, spanned::Spanned};

/// Parsed `#[fuse(arch = "sm_89")]` attribute.
#[derive(Debug)]
pub struct FuseAttr {
    pub arch: String,
}

impl syn::parse::Parse for FuseAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Parse `arch = "sm_89"`
        let ident: syn::Ident = input.parse()?;
        if ident != "arch" {
            return Err(syn::Error::new(ident.span(), "expected `arch`"));
        }
        let _eq: syn::Token![=] = input.parse()?;
        let lit: syn::LitStr = input.parse()?;
        Ok(FuseAttr { arch: lit.value() })
    }
}

/// Intermediate parsed call before DAG construction.
struct ParsedCall {
    func_name: String,
    args: Vec<String>,
    result_name: Option<String>,
    span: proc_macro2::Span,
}

/// Parse a function body into an OpGraph (real DAG with edges).
///
/// Supported patterns:
/// - `let x = op(args...);`
/// - `op(args...)` as trailing expression (return value)
pub fn parse_fn_body(func: &ItemFn) -> syn::Result<OpGraph> {
    let mut graph = OpGraph::new();

    // Extract parameter names and types
    for arg in &func.sig.inputs {
        match arg {
            syn::FnArg::Typed(pat_ty) => {
                let name = pat_to_string(&pat_ty.pat)?;
                let ty = type_to_string(&pat_ty.ty);
                graph.params.push(ParamInfo { name, ty });
            }
            syn::FnArg::Receiver(_) => {
                return Err(syn::Error::new(arg.span(), "self parameters not supported"));
            }
        }
    }

    // First pass: collect all calls
    let mut calls = Vec::new();
    let block = &func.block;
    for stmt in block.stmts.iter() {
        match stmt {
            Stmt::Local(local) => {
                let result_name = pat_to_string(&local.pat)?;
                let init = local.init.as_ref().ok_or_else(|| {
                    syn::Error::new(local.span(), "let binding must have initializer")
                })?;
                let (func_name, args) = parse_call_expr(&init.expr)?;
                calls.push(ParsedCall {
                    func_name,
                    args,
                    result_name: Some(result_name),
                    span: init.expr.span(),
                });
            }
            Stmt::Expr(expr, _semi) => {
                let (func_name, args) = parse_call_expr(expr)?;
                calls.push(ParsedCall {
                    func_name,
                    args,
                    result_name: None,
                    span: expr.span(),
                });
            }
            _ => {
                return Err(syn::Error::new(
                    stmt.span(),
                    "unsupported statement in fuse function",
                ));
            }
        }
    }

    // Second pass: build DAG nodes with edges
    for (idx, call) in calls.iter().enumerate() {
        let (kind, inputs) = match call.func_name.as_str() {
            "rmsnorm" => {
                if call.args.len() != 2 {
                    return Err(syn::Error::new(call.span, "rmsnorm expects 2 arguments"));
                }
                let inputs = vec![
                    resolve_edge(&graph, &call.args[0], InputPort::Primary, call.span)?,
                    resolve_edge(&graph, &call.args[1], InputPort::Weight, call.span)?,
                ];
                (OpKind::RmsNorm, inputs)
            }
            "gemm" => {
                if call.args.len() != 2 {
                    return Err(syn::Error::new(call.span, "gemm expects 2 arguments"));
                }
                let inputs = vec![
                    resolve_edge(&graph, &call.args[0], InputPort::Primary, call.span)?,
                    resolve_edge(&graph, &call.args[1], InputPort::Weight, call.span)?,
                ];
                (OpKind::Gemm, inputs)
            }
            "silu" => {
                if call.args.len() != 1 {
                    return Err(syn::Error::new(call.span, "silu expects 1 argument"));
                }
                let inputs = vec![resolve_edge(
                    &graph,
                    &call.args[0],
                    InputPort::Primary,
                    call.span,
                )?];
                (OpKind::Silu, inputs)
            }
            other => {
                return Err(syn::Error::new(call.span, format!("unknown op: {other}")));
            }
        };

        let class = kind.class();
        graph.nodes.push(OpNode {
            id: idx,
            kind,
            class,
            inputs,
            result_name: call.result_name.clone(),
        });
    }

    // Set output to the last node
    if !graph.nodes.is_empty() {
        graph.output = graph.nodes.len() - 1;
    }

    Ok(graph)
}

/// Resolve a variable name to an Edge: either from a producer node or a function parameter.
fn resolve_edge(
    graph: &OpGraph,
    name: &str,
    port: InputPort,
    span: proc_macro2::Span,
) -> syn::Result<Edge> {
    // Check if it's produced by an earlier node
    if let Some(src) = graph.producer_of(name) {
        return Ok(Edge {
            src,
            port,
            src_name: name.to_string(),
        });
    }

    // Check if it's a function parameter
    if graph.is_param(name) {
        return Ok(Edge {
            src: PARAM,
            port,
            src_name: name.to_string(),
        });
    }

    Err(syn::Error::new(
        span,
        format!("undefined variable: `{name}` — not a function parameter or previous result"),
    ))
}

/// Parse a function call expression like `gemm(a, b)` into (func_name, args).
fn parse_call_expr(expr: &Expr) -> syn::Result<(String, Vec<String>)> {
    match expr {
        Expr::Call(call) => {
            let func_name = expr_to_ident(&call.func)?;
            let args: Vec<String> = call
                .args
                .iter()
                .map(|a| expr_to_string(a))
                .collect::<syn::Result<_>>()?;
            Ok((func_name, args))
        }
        _ => Err(syn::Error::new(
            expr.span(),
            "expected a function call expression",
        )),
    }
}

fn expr_to_ident(expr: &Expr) -> syn::Result<String> {
    match expr {
        Expr::Path(p) => Ok(p
            .path
            .segments
            .last()
            .ok_or_else(|| syn::Error::new(p.span(), "empty path"))?
            .ident
            .to_string()),
        _ => Err(syn::Error::new(expr.span(), "expected identifier")),
    }
}

fn expr_to_string(expr: &Expr) -> syn::Result<String> {
    match expr {
        Expr::Path(p) => Ok(p
            .path
            .segments
            .last()
            .ok_or_else(|| syn::Error::new(p.span(), "empty path"))?
            .ident
            .to_string()),
        _ => Err(syn::Error::new(expr.span(), "expected identifier")),
    }
}

fn pat_to_string(pat: &syn::Pat) -> syn::Result<String> {
    match pat {
        syn::Pat::Ident(pi) => Ok(pi.ident.to_string()),
        _ => Err(syn::Error::new(pat.span(), "expected identifier pattern")),
    }
}

fn type_to_string(ty: &syn::Type) -> String {
    quote::quote!(#ty).to_string()
}
