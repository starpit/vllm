use crate::ops::{Op, OpGraph, OpNode};
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

/// Parse a function body into an OpGraph.
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
                graph.params.push((name, ty));
            }
            syn::FnArg::Receiver(_) => {
                return Err(syn::Error::new(arg.span(), "self parameters not supported"));
            }
        }
    }

    let block = &func.block;
    for (i, stmt) in block.stmts.iter().enumerate() {
        let is_last = i == block.stmts.len() - 1;

        match stmt {
            Stmt::Local(local) => {
                // `let x = op(args...);`
                let result_name = pat_to_string(&local.pat)?;
                let init = local.init.as_ref().ok_or_else(|| {
                    syn::Error::new(local.span(), "let binding must have initializer")
                })?;
                let op = parse_call_expr(&init.expr)?;
                let index = graph.nodes.len();
                graph.nodes.push(OpNode {
                    result_name: Some(result_name),
                    op,
                    index,
                });
            }
            Stmt::Expr(expr, _semi) => {
                // Trailing expression or expression statement
                let op = parse_call_expr(expr)?;
                let index = graph.nodes.len();
                graph.nodes.push(OpNode {
                    result_name: if is_last { None } else { None },
                    op,
                    index,
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

    Ok(graph)
}

/// Parse a function call expression like `gemm(a, b)` into an Op.
fn parse_call_expr(expr: &Expr) -> syn::Result<Op> {
    match expr {
        Expr::Call(call) => {
            let func_name = expr_to_ident(&call.func)?;
            let args: Vec<String> = call
                .args
                .iter()
                .map(|a| expr_to_string(a))
                .collect::<syn::Result<_>>()?;

            match func_name.as_str() {
                "rmsnorm" => {
                    if args.len() != 2 {
                        return Err(syn::Error::new(call.span(), "rmsnorm expects 2 arguments"));
                    }
                    Ok(Op::RmsNorm {
                        input: args[0].clone(),
                        weight: args[1].clone(),
                    })
                }
                "gemm" => {
                    if args.len() != 2 {
                        return Err(syn::Error::new(call.span(), "gemm expects 2 arguments"));
                    }
                    Ok(Op::Gemm {
                        a: args[0].clone(),
                        b: args[1].clone(),
                    })
                }
                "silu" => {
                    if args.len() != 1 {
                        return Err(syn::Error::new(call.span(), "silu expects 1 argument"));
                    }
                    Ok(Op::Silu {
                        input: args[0].clone(),
                    })
                }
                other => Err(syn::Error::new(call.span(), format!("unknown op: {other}"))),
            }
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
