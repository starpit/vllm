// SPDX-License-Identifier: Apache-2.0
//! `syn::Block` → [`Ast`].
//!
//! Walks the carrier fn's body, recognizing the DSL's statement
//! and expression shapes:
//!
//!   - `name = expr;` → [`Stmt::Assign`]
//!   - `(a, b, c) = expr;` → [`Stmt::AssignTuple`]
//!   - `for ivar in start..end { ... }` → [`Stmt::For`]
//!   - expression forms: bare idents, dotted paths, indexing,
//!     calls, and multiplication (`gate * up`).
//!
//! The parser is intentionally strict: anything outside this
//! vocabulary is an error, because the DSL is a compiler front-end
//! and surprise-Rust-constructs below the parser is exactly the
//! kind of leakage we're trying to prevent.
//!
//! Identifiers are preserved verbatim. No classification, no
//! substitution, no shape reasoning happens here.
use syn::spanned::Spanned;
use syn::{BinOp, Block, Expr as SynExpr, ExprLit, Lit, Pat, Stmt as SynStmt};

use crate::ast::{Ast, BoolExpr, BoundExpr, Expr, Stmt};

pub type ParseResult<T> = Result<T, syn::Error>;

/// Parse the body of a `#[forward]` carrier fn.
pub fn parse_block(block: &Block) -> ParseResult<Ast> {
    let statements = parse_stmts(&block.stmts)?;
    Ok(Ast { statements })
}

fn parse_stmts(stmts: &[SynStmt]) -> ParseResult<Vec<Stmt>> {
    stmts.iter().map(parse_stmt).collect()
}

fn parse_stmt(stmt: &SynStmt) -> ParseResult<Stmt> {
    match stmt {
        // A bare expression-with-semicolon or a trailing expression
        // (the common forms of `name = expr;` and `for ... {}`).
        SynStmt::Expr(expr, _) => parse_stmt_expr(expr),

        // `let x = ...;` is not part of the DSL (see PLAN.md:
        // "Drop `let` entirely"). Reject loudly so we never grow a
        // `let`-tolerance that then drags scoping into the IR.
        SynStmt::Local(local) => Err(syn::Error::new(
            local.span(),
            "`let` is not part of the DSL; write `name = expr;` instead",
        )),

        SynStmt::Item(item) => Err(syn::Error::new(
            item.span(),
            "items (fn/struct/mod/...) are not allowed inside the forward! body",
        )),

        SynStmt::Macro(m) => Err(syn::Error::new(
            m.span(),
            "macros are not allowed inside the forward! body",
        )),
    }
}

fn parse_stmt_expr(expr: &SynExpr) -> ParseResult<Stmt> {
    match expr {
        SynExpr::Assign(a) => parse_assign(a),
        SynExpr::ForLoop(f) => parse_for(f),
        SynExpr::If(i) => parse_if(i),
        other => Err(syn::Error::new(
            other.span(),
            "expected `name = expr;`, `(a, b, c) = expr;`, \
             `for x in 0..N { ... }`, or `if <pred> { ... } else { ... }`",
        )),
    }
}

fn parse_assign(a: &syn::ExprAssign) -> ParseResult<Stmt> {
    let value = parse_expr(&a.right)?;
    match &*a.left {
        SynExpr::Path(p) if p.path.get_ident().is_some() => Ok(Stmt::Assign {
            target: p.path.get_ident().unwrap().clone(),
            value,
        }),
        SynExpr::Tuple(tuple) => {
            let mut targets = Vec::with_capacity(tuple.elems.len());
            for elem in &tuple.elems {
                match elem {
                    SynExpr::Path(p) if p.path.get_ident().is_some() => {
                        targets.push(p.path.get_ident().unwrap().clone());
                    }
                    other => {
                        return Err(syn::Error::new(
                            other.span(),
                            "tuple destructuring targets must be plain identifiers",
                        ));
                    }
                }
            }
            Ok(Stmt::AssignTuple { targets, value })
        }
        other => Err(syn::Error::new(
            other.span(),
            "assignment target must be an identifier or a tuple of identifiers",
        )),
    }
}

fn parse_for(f: &syn::ExprForLoop) -> ParseResult<Stmt> {
    let ivar = match &*f.pat {
        Pat::Ident(p) => p.ident.clone(),
        other => {
            return Err(syn::Error::new(
                other.span(),
                "for-loop variable must be a plain identifier",
            ));
        }
    };
    let (start, end) = parse_range(&f.expr)?;
    let body = parse_stmts(&f.body.stmts)?;
    Ok(Stmt::For {
        ivar,
        start,
        end,
        body,
    })
}

fn parse_if(i: &syn::ExprIf) -> ParseResult<Stmt> {
    let cond = parse_bool_expr(&i.cond)?;
    let then_body = parse_stmts(&i.then_branch.stmts)?;
    let else_body = match i.else_branch.as_ref() {
        Some((_, else_expr)) => match &**else_expr {
            SynExpr::Block(b) => parse_stmts(&b.block.stmts)?,
            SynExpr::If(_) => {
                return Err(syn::Error::new(
                    else_expr.span(),
                    "`else if` chains are not supported; nest an `if` inside `else { ... }` instead",
                ));
            }
            _ => {
                return Err(syn::Error::new(
                    else_expr.span(),
                    "`else` must be a block `else { ... }`",
                ));
            }
        },
        None => {
            return Err(syn::Error::new(
                i.span(),
                "`if` must have an `else` arm; both arms must bind the same set of names",
            ));
        }
    };
    Ok(Stmt::If {
        cond,
        then_body,
        else_body,
    })
}

/// Parse a boolean predicate for an `if` condition. Accepts only
/// two shapes: `ivar % <bound> == <bound>` or `ivar < <bound>`,
/// where `ivar` is a plain identifier (must refer to an enclosing
/// loop induction variable; classify enforces this) and `<bound>`
/// is an integer literal or a bare identifier naming a per-model
/// bound (e.g. `sliding_window_pattern`).
fn parse_bool_expr(expr: &SynExpr) -> ParseResult<BoolExpr> {
    let expr = unwrap_parens(expr);
    match expr {
        SynExpr::Binary(b) => match b.op {
            BinOp::Eq(_) => {
                // Expect left = `ivar % <bound>`, right = `<bound>`.
                let (ivar, divisor) = match unwrap_parens(&b.left) {
                    SynExpr::Binary(inner) if matches!(inner.op, BinOp::Rem(_)) => {
                        let ivar = parse_ivar(&inner.left)?;
                        let divisor = parse_bound(unwrap_parens(&inner.right))?;
                        (ivar, divisor)
                    }
                    other => {
                        return Err(syn::Error::new(
                            other.span(),
                            "left of `==` in an `if` condition must be `ivar % <bound>`",
                        ));
                    }
                };
                let remainder = parse_bound(unwrap_parens(&b.right))?;
                Ok(BoolExpr::Modulo {
                    ivar,
                    divisor,
                    remainder,
                })
            }
            BinOp::Ne(_) => {
                let (ivar, divisor) = match unwrap_parens(&b.left) {
                    SynExpr::Binary(inner) if matches!(inner.op, BinOp::Rem(_)) => {
                        let ivar = parse_ivar(&inner.left)?;
                        let divisor = parse_bound(unwrap_parens(&inner.right))?;
                        (ivar, divisor)
                    }
                    other => {
                        return Err(syn::Error::new(
                            other.span(),
                            "left of `!=` in an `if` condition must be `ivar % <bound>`",
                        ));
                    }
                };
                let remainder = parse_bound(unwrap_parens(&b.right))?;
                Ok(BoolExpr::NotModulo {
                    ivar,
                    divisor,
                    remainder,
                })
            }
            BinOp::Lt(_) => {
                let ivar = parse_ivar(&b.left)?;
                let bound = parse_bound(unwrap_parens(&b.right))?;
                Ok(BoolExpr::Less { ivar, bound })
            }
            _ => Err(syn::Error::new(
                b.op.span(),
                "only `%`+`==`, `%`+`!=`, and `<` are supported in `if` conditions; \
                 shapes: `ivar % N == M`, `ivar % N != M`, or `ivar < N`",
            )),
        },
        other => Err(syn::Error::new(
            other.span(),
            "`if` condition must be `ivar % <bound> == <bound>`, \
             `ivar % <bound> != <bound>`, or `ivar < <bound>`",
        )),
    }
}

fn parse_ivar(expr: &SynExpr) -> ParseResult<syn::Ident> {
    match unwrap_parens(expr) {
        SynExpr::Path(p) if p.path.get_ident().is_some() => Ok(p.path.get_ident().unwrap().clone()),
        other => Err(syn::Error::new(
            other.span(),
            "expected a plain identifier (the enclosing loop induction variable)",
        )),
    }
}

fn unwrap_parens(expr: &SynExpr) -> &SynExpr {
    let mut cur = expr;
    while let SynExpr::Paren(p) = cur {
        cur = &p.expr;
    }
    cur
}

fn parse_range(expr: &SynExpr) -> ParseResult<(BoundExpr, BoundExpr)> {
    match expr {
        SynExpr::Range(r) => {
            let start = r.start.as_deref().map(parse_bound).transpose()?;
            let end = r
                .end
                .as_deref()
                .map(parse_bound)
                .transpose()?
                .ok_or_else(|| {
                    syn::Error::new(r.span(), "for-loop range must have an upper bound")
                })?;
            Ok((start.unwrap_or(BoundExpr::Lit(0)), end))
        }
        other => Err(syn::Error::new(
            other.span(),
            "for-loop expression must be a range `start..end`",
        )),
    }
}

fn parse_bound(expr: &SynExpr) -> ParseResult<BoundExpr> {
    match expr {
        SynExpr::Lit(ExprLit {
            lit: Lit::Int(i), ..
        }) => Ok(BoundExpr::Lit(i.base10_parse()?)),
        SynExpr::Path(p) if p.path.get_ident().is_some() => {
            Ok(BoundExpr::Ident(p.path.get_ident().unwrap().clone()))
        }
        other => Err(syn::Error::new(
            other.span(),
            "loop bound must be an integer literal or a bare identifier",
        )),
    }
}

pub fn parse_expr(expr: &SynExpr) -> ParseResult<Expr> {
    match expr {
        // A simple identifier — `hidden_states`, `input_ids`.
        SynExpr::Path(p) if p.path.get_ident().is_some() => {
            Ok(Expr::Var(p.path.get_ident().unwrap().clone()))
        }

        // A dotted path — `self_attn.q_proj`.
        SynExpr::Field(_) => {
            let segments = collect_path_segments(expr)?;
            if segments.len() == 1 {
                Ok(Expr::Var(segments.into_iter().next().unwrap()))
            } else {
                Ok(Expr::Path(segments))
            }
        }

        // Indexing: `self_attn.q_proj[layer]`.
        SynExpr::Index(i) => {
            let target = parse_expr(&i.expr)?;
            let index = parse_index(&i.index)?;
            Ok(Expr::Index {
                target: Box::new(target),
                index,
            })
        }

        // Call: `gemm(x, w)`, `rmsnorm(x, w)`, etc.
        //
        // Special case: `sqrt(<bound_name>)` parses to a compile-time
        // scalar expression (resolved per-model at CFG build). Any
        // other shape of `sqrt(...)` is rejected.
        SynExpr::Call(c) => {
            let op = match &*c.func {
                SynExpr::Path(p) if p.path.get_ident().is_some() => {
                    p.path.get_ident().unwrap().clone()
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "call target must be a plain op name",
                    ));
                }
            };
            if op == "sqrt" {
                if c.args.len() != 1 {
                    return Err(syn::Error::new(
                        op.span(),
                        "`sqrt(<bound_name>)` takes exactly one argument",
                    ));
                }
                let ident = match &c.args[0] {
                    SynExpr::Path(p) if p.path.get_ident().is_some() => {
                        p.path.get_ident().unwrap().clone()
                    }
                    other => {
                        return Err(syn::Error::new(
                            other.span(),
                            "`sqrt(...)` argument must be a bound identifier",
                        ));
                    }
                };
                return Ok(Expr::SqrtBound(ident));
            }
            if op == "scalar" || op == "recip_scalar" {
                if c.args.len() != 1 {
                    return Err(syn::Error::new(
                        op.span(),
                        "`scalar(<name>)` / `recip_scalar(<name>)` takes exactly one argument",
                    ));
                }
                let ident = match &c.args[0] {
                    SynExpr::Path(p) if p.path.get_ident().is_some() => {
                        p.path.get_ident().unwrap().clone()
                    }
                    other => {
                        return Err(syn::Error::new(
                            other.span(),
                            "`scalar(...)` / `recip_scalar(...)` argument must be a config key identifier",
                        ));
                    }
                };
                return Ok(Expr::ConfigScalar {
                    name: ident,
                    recip: op == "recip_scalar",
                });
            }
            let args = c
                .args
                .iter()
                .map(parse_expr)
                .collect::<ParseResult<Vec<_>>>()?;
            Ok(Expr::Call { op, args })
        }

        // `gate * up` (SwiGLU tensor×tensor) or `w + 1.0`
        // (scalar offset on a weight — e.g. Gemma's `(1+w)` rmsnorm).
        // Other binary operators are rejected.
        SynExpr::Binary(b) => {
            let lhs = parse_expr(&b.left)?;
            let rhs = parse_expr(&b.right)?;
            match b.op {
                BinOp::Mul(_) => Ok(Expr::Mul {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }),
                BinOp::Add(_) => Ok(Expr::Add {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }),
                _ => Err(syn::Error::new(
                    b.op.span(),
                    "only `*` and `+` are admitted in DSL expressions",
                )),
            }
        }

        // A numeric literal — produces a scalar value usable as an
        // operand to `+` / `*`. Integer literals are promoted to f64.
        SynExpr::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Float(f) => Ok(Expr::ScalarLit(f.base10_parse()?)),
            Lit::Int(i) => Ok(Expr::ScalarLit(i.base10_parse::<u64>()? as f64)),
            _ => Err(syn::Error::new(
                lit.span(),
                "only numeric literals are admitted as DSL scalars",
            )),
        },

        // Parenthesized — unwrap.
        SynExpr::Paren(p) => parse_expr(&p.expr),

        other => Err(syn::Error::new(
            other.span(),
            "expression shape not recognized by the DSL",
        )),
    }
}

/// Turn a `SynExpr` that's a chain of `.field` accesses on a root
/// `Path` ident into a flat `Vec<Ident>` of segments.
fn collect_path_segments(expr: &SynExpr) -> ParseResult<Vec<syn::Ident>> {
    let mut out = Vec::new();
    collect_path_segments_into(expr, &mut out)?;
    Ok(out)
}

fn collect_path_segments_into(expr: &SynExpr, out: &mut Vec<syn::Ident>) -> ParseResult<()> {
    match expr {
        SynExpr::Path(p) if p.path.get_ident().is_some() => {
            out.push(p.path.get_ident().unwrap().clone());
            Ok(())
        }
        SynExpr::Field(f) => {
            collect_path_segments_into(&f.base, out)?;
            let name = match &f.member {
                syn::Member::Named(id) => id.clone(),
                syn::Member::Unnamed(_) => {
                    return Err(syn::Error::new(
                        f.span(),
                        "numeric tuple-field access is not part of the DSL",
                    ));
                }
            };
            out.push(name);
            Ok(())
        }
        other => Err(syn::Error::new(
            other.span(),
            "path root must be a plain identifier",
        )),
    }
}

fn parse_index(expr: &SynExpr) -> ParseResult<syn::Ident> {
    match expr {
        SynExpr::Path(p) if p.path.get_ident().is_some() => Ok(p.path.get_ident().unwrap().clone()),
        other => Err(syn::Error::new(
            other.span(),
            "index must be a single identifier (the enclosing loop variable)",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Ast {
        let file: syn::File =
            syn::parse_str(&format!("fn _carrier() {{ {src} }}")).expect("syntactic parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        parse_block(block).expect("DSL parse")
    }

    #[test]
    fn assign_simple() {
        let ast = parse("x = foo(a, b);");
        assert_eq!(ast.statements.len(), 1);
        match &ast.statements[0] {
            Stmt::Assign { target, value } => {
                assert_eq!(target.to_string(), "x");
                match value {
                    Expr::Call { op, args } => {
                        assert_eq!(op.to_string(), "foo");
                        assert_eq!(args.len(), 2);
                    }
                    _ => panic!("expected call"),
                }
            }
            _ => panic!("expected assign"),
        }
    }

    #[test]
    fn assign_tuple() {
        let ast = parse("(q, k, v) = rope(q, k, v, pos);");
        match &ast.statements[0] {
            Stmt::AssignTuple { targets, value } => {
                let names: Vec<String> = targets.iter().map(|i| i.to_string()).collect();
                assert_eq!(names, vec!["q", "k", "v"]);
                assert!(matches!(value, Expr::Call { .. }));
            }
            _ => panic!("expected tuple assign"),
        }
    }

    #[test]
    fn for_loop_with_symbolic_bound() {
        let ast = parse("for layer in 0..num_hidden_layers { x = y; }");
        match &ast.statements[0] {
            Stmt::For {
                ivar,
                start,
                end,
                body,
            } => {
                assert_eq!(ivar.to_string(), "layer");
                assert!(matches!(start, BoundExpr::Lit(0)));
                match end {
                    BoundExpr::Ident(i) => assert_eq!(i.to_string(), "num_hidden_layers"),
                    _ => panic!("expected symbolic end"),
                }
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected for-loop"),
        }
    }

    #[test]
    fn dotted_path_expression() {
        let ast = parse("x = self_attn.q_proj;");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Path(segs),
                ..
            } => {
                let names: Vec<String> = segs.iter().map(|i| i.to_string()).collect();
                assert_eq!(names, vec!["self_attn", "q_proj"]);
            }
            _ => panic!("expected path expression"),
        }
    }

    #[test]
    fn indexed_weight_reference() {
        let ast = parse("q = gemm(x, self_attn.q_proj[layer]);");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[1] {
                Expr::Index { target, index } => {
                    match &**target {
                        Expr::Path(segs) => assert_eq!(segs.len(), 2),
                        _ => panic!("expected path target"),
                    }
                    assert_eq!(index.to_string(), "layer");
                }
                other => panic!("expected indexed path, got {other:?}"),
            },
            _ => panic!("expected call"),
        }
    }

    #[test]
    fn mul_expression() {
        let ast = parse("down = gemm(gate * up, mlp.down_proj[layer]);");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[0] {
                Expr::Mul { lhs, rhs } => match (&**lhs, &**rhs) {
                    (Expr::Var(l), Expr::Var(r)) => {
                        assert_eq!(l.to_string(), "gate");
                        assert_eq!(r.to_string(), "up");
                    }
                    _ => panic!("expected (gate, up) var operands"),
                },
                other => panic!("expected Mul, got {other:?}"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn bare_var_expression() {
        // `hidden_states = add(oproj, hidden_states)` — the second
        // arg is a bare Var whose ident we can read.
        let ast = parse("x = add(a, b);");
        match &ast.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[1] {
                Expr::Var(i) => assert_eq!(i.to_string(), "b"),
                other => panic!("expected bare Var, got {other:?}"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn let_is_rejected() {
        let src = "let x = foo(a);";
        let result: syn::Result<_> = (|| -> syn::Result<Ast> {
            let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}"))?;
            let block = match &file.items[0] {
                syn::Item::Fn(f) => &*f.block,
                _ => unreachable!(),
            };
            parse_block(block)
        })();
        assert!(result.is_err(), "let should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("`let`"),
            "error should mention let: {err_msg}"
        );
    }

    #[test]
    fn if_modulo_parses() {
        let ast = parse(
            "for layer in 0..4 { \
                if layer % 2 == 0 { x = attention(q, k, v, kv, b); } \
                else { x = sliding_attention(q, k, v, kv, b); } \
            }",
        );
        match &ast.statements[0] {
            Stmt::For { body, .. } => match &body[0] {
                Stmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    match cond {
                        BoolExpr::Modulo {
                            ivar,
                            divisor,
                            remainder,
                        } => {
                            assert_eq!(ivar.to_string(), "layer");
                            assert!(matches!(divisor, BoundExpr::Lit(2)));
                            assert!(matches!(remainder, BoundExpr::Lit(0)));
                        }
                        other => panic!("expected Modulo, got {other:?}"),
                    }
                    assert_eq!(then_body.len(), 1);
                    assert_eq!(else_body.len(), 1);
                }
                other => panic!("expected If, got {other:?}"),
            },
            _ => panic!("expected for-loop"),
        }
    }

    #[test]
    fn if_less_with_symbolic_bound_parses() {
        let ast = parse(
            "for layer in 0..4 { \
                if layer < num_dense_layers { x = gemm(a, b); } \
                else { x = gemm(a, b); } \
            }",
        );
        match &ast.statements[0] {
            Stmt::For { body, .. } => match &body[0] {
                Stmt::If { cond, .. } => match cond {
                    BoolExpr::Less { ivar, bound } => {
                        assert_eq!(ivar.to_string(), "layer");
                        match bound {
                            BoundExpr::Ident(i) => assert_eq!(i.to_string(), "num_dense_layers"),
                            _ => panic!("expected symbolic bound"),
                        }
                    }
                    other => panic!("expected Less, got {other:?}"),
                },
                _ => panic!("expected If"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn if_without_else_is_rejected() {
        let src = "for layer in 0..4 { if layer < 2 { x = gemm(a, b); } }";
        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).unwrap();
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let err = parse_block(block).expect_err("should reject missing else");
        assert!(err.to_string().contains("else"), "error mentions else");
    }

    #[test]
    fn else_if_chain_is_rejected() {
        let src = "for layer in 0..4 { \
            if layer < 2 { x = gemm(a, b); } \
            else if layer < 4 { x = gemm(a, b); } \
            else { x = gemm(a, b); } \
        }";
        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).unwrap();
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let err = parse_block(block).expect_err("should reject else-if");
        assert!(
            err.to_string().contains("else if"),
            "error mentions else if: {}",
            err
        );
    }

    #[test]
    fn unsupported_condition_shape_is_rejected() {
        let src = "for layer in 0..4 { \
            if layer + 1 == 3 { x = gemm(a, b); } else { x = gemm(a, b); } \
        }";
        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).unwrap();
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let err = parse_block(block).expect_err("should reject addition in condition");
        let msg = err.to_string();
        assert!(
            msg.contains("ivar % <bound>") || msg.contains("`%`+`==`"),
            "error names supported shapes: {msg}"
        );
    }

    #[test]
    fn realistic_llama_body_parses() {
        // Structural shape of the Llama body. Not asserting
        // semantics — just that the parser accepts the realistic
        // DSL vocabulary without changes.
        let body = r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                k = gemm(normed, self_attn.k_proj[layer]);
                v = gemm(normed, self_attn.v_proj[layer]);
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                attn = attention(q, k, v, kv_cache[layer], block_table);
                oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);

                normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                up = gemm(normed2, mlp.up_proj[layer]);
                down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }
            normed = rmsnorm(hidden_states, norm);
            logits = gemm(normed, lm_head);
        "#;
        let ast = parse(body);

        // Three top-level statements: embed, for-loop, final two gemms.
        assert_eq!(
            ast.statements.len(),
            4,
            "expected 4 top-level statements (embed, for, norm, lm_head)",
        );
        assert!(matches!(ast.statements[0], Stmt::Assign { .. }));
        match &ast.statements[1] {
            Stmt::For {
                ivar, end, body, ..
            } => {
                assert_eq!(ivar.to_string(), "layer");
                match end {
                    BoundExpr::Ident(i) => assert_eq!(i.to_string(), "num_hidden_layers"),
                    _ => panic!("expected symbolic bound"),
                }
                // 13 statements per iteration:
                //   normed, q, k, v, (q,k,v)=rope, attn, oproj, hidden=add,
                //   normed2, gate, up, down, hidden=add
                assert_eq!(body.len(), 13, "13 body statements per iteration");
            }
            _ => panic!("expected for-loop"),
        }
    }
}
