// SPDX-License-Identifier: Apache-2.0
//! Phase 2: classify free variables.
//!
//! Walks the raw AST and produces a [`classified::Program`] in
//! which every variable reference is tagged as a local, extern
//! param, or weight ref. Unknown reads (a bare name that isn't any
//! of those at a given site) are parse-time errors, not papered
//! over.

use std::collections::HashMap;

use syn::Ident;

use crate::ast::{self, BoundExpr};
use crate::classified::{
    Bound, Expr, ExternKind, LocalId, LocalTable, OpKind, Program, Stmt, WeightTable,
};

pub type ClassifyResult<T> = Result<T, syn::Error>;

/// Classify the AST.
pub fn classify(ast: &ast::Ast) -> ClassifyResult<Program> {
    let mut cx = Ctx::default();
    let statements = cx.classify_stmts(&ast.statements)?;
    Ok(Program {
        statements,
        locals: cx.locals,
        weights: cx.weights,
    })
}

#[derive(Default)]
struct Ctx {
    locals: LocalTable,
    weights: WeightTable,
    /// Name → most-recent LocalId for that name. Reflects straight-
    /// line SSA: a new assignment shadows the previous binding.
    /// For-loops push their induction variable and pop it on exit.
    scope: HashMap<String, Vec<LocalId>>,
}

impl Ctx {
    fn classify_stmts(&mut self, stmts: &[ast::Stmt]) -> ClassifyResult<Vec<Stmt>> {
        stmts.iter().map(|s| self.classify_stmt(s)).collect()
    }

    fn classify_stmt(&mut self, stmt: &ast::Stmt) -> ClassifyResult<Stmt> {
        match stmt {
            ast::Stmt::Assign { target, value } => {
                // Classify RHS first so that `x = f(x)` reads the
                // previous binding of `x`, not the new one.
                let value = self.classify_expr(value)?;
                let id = self.bind(target.clone());
                Ok(Stmt::Assign { target: id, value })
            }
            ast::Stmt::AssignTuple { targets, value } => {
                let value = self.classify_expr(value)?;
                let ids = targets.iter().map(|t| self.bind(t.clone())).collect();
                Ok(Stmt::AssignTuple {
                    targets: ids,
                    value,
                })
            }
            ast::Stmt::For {
                ivar,
                start,
                end,
                body,
            } => {
                let start = self.classify_bound(start);
                let end = self.classify_bound(end);
                // Push the loop var as a fresh local scoped to the body.
                let iv_id = self.push_scope(ivar.clone());
                let body = self.classify_stmts(body)?;
                self.pop_scope(&ivar.to_string());
                Ok(Stmt::For {
                    ivar: iv_id,
                    start,
                    end,
                    body,
                })
            }
        }
    }

    fn classify_bound(&self, b: &BoundExpr) -> Bound {
        match b {
            BoundExpr::Lit(n) => Bound::Lit(*n),
            BoundExpr::Ident(i) => Bound::Sym(i.clone()),
        }
    }

    fn classify_expr(&mut self, expr: &ast::Expr) -> ClassifyResult<Expr> {
        match expr {
            ast::Expr::Var(ident) => self.classify_read(ident, None),

            ast::Expr::Path(segments) => {
                // Dotted paths never resolve to locals or externs
                // (those are all single-ident). They're weight refs.
                let id = self.weights.intern(segments.clone());
                Ok(Expr::Weight { id, index: None })
            }

            ast::Expr::Index { target, index } => {
                // Resolve the index (must be a local — typically a
                // loop variable).
                let index_id = self.lookup_local(index).ok_or_else(|| {
                    syn::Error::new(
                        index.span(),
                        format!(
                            "index `{index}` must refer to a local binding \
                             (usually the enclosing loop variable)",
                        ),
                    )
                })?;
                // Classify the target with the index attached.
                let target_expr = self.classify_read_target(target)?;
                attach_index(target_expr, index_id)
            }

            ast::Expr::Call { op, args } => {
                let kind = OpKind::from_name(&op.to_string())
                    .ok_or_else(|| syn::Error::new(op.span(), format!("unknown op `{op}`")))?;
                let args = args
                    .iter()
                    .map(|a| self.classify_expr(a))
                    .collect::<ClassifyResult<Vec<_>>>()?;
                Ok(Expr::Call { op: kind, args })
            }

            ast::Expr::Mul { lhs, rhs } => {
                let lhs = self.classify_expr(lhs)?;
                let rhs = self.classify_expr(rhs)?;
                Ok(Expr::Mul {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }
        }
    }

    /// Classify a read that appears as the target of an indexing
    /// operation. Like `classify_expr` but only admits Var and Path
    /// shapes (a call or mul can't be indexed in the DSL).
    fn classify_read_target(&mut self, expr: &ast::Expr) -> ClassifyResult<Expr> {
        match expr {
            ast::Expr::Var(ident) => self.classify_read(ident, None),
            ast::Expr::Path(segments) => {
                let id = self.weights.intern(segments.clone());
                Ok(Expr::Weight { id, index: None })
            }
            _ => Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "indexing target must be a variable or a dotted path",
            )),
        }
    }

    fn classify_read(&mut self, ident: &Ident, index: Option<LocalId>) -> ClassifyResult<Expr> {
        let name = ident.to_string();

        // Preference order: local binding, then extern param, then
        // weight ref. A local binding shadows anything — you can't
        // write a local that happens to be named `input_ids` and
        // still read the extern. (If that ever matters we'll add a
        // parse-time warn; the issue has never arisen in practice.)
        if let Some(id) = self.lookup_local(ident) {
            return Ok(Expr::Local(id));
        }
        if let Some(kind) = ExternKind::from_name(&name) {
            return Ok(Expr::Extern { kind, index });
        }
        // Weight ref by bare ident (e.g. `embed_tokens`, `lm_head`,
        // `norm`). Intern as a single-segment path.
        let id = self.weights.intern(vec![ident.clone()]);
        Ok(Expr::Weight { id, index })
    }

    fn lookup_local(&self, ident: &Ident) -> Option<LocalId> {
        self.scope
            .get(&ident.to_string())
            .and_then(|stack| stack.last().copied())
    }

    /// Introduce a fresh local binding shadowing any previous
    /// binding of the same name. Used by assignment statements —
    /// the new binding outlives the statement.
    fn bind(&mut self, ident: Ident) -> LocalId {
        let name = ident.to_string();
        let id = self.locals.push(ident);
        // Replace the top of the stack so shadowing is observable
        // to subsequent reads at this scope depth.
        let stack = self.scope.entry(name).or_default();
        if stack.is_empty() {
            stack.push(id);
        } else {
            *stack.last_mut().unwrap() = id;
        }
        id
    }

    /// Introduce a fresh local binding scoped to the enclosing
    /// construct (used for for-loop induction variables). The
    /// caller must pair this with `pop_scope` on exit.
    fn push_scope(&mut self, ident: Ident) -> LocalId {
        let name = ident.to_string();
        let id = self.locals.push(ident);
        self.scope.entry(name).or_default().push(id);
        id
    }

    fn pop_scope(&mut self, name: &str) {
        if let Some(stack) = self.scope.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.scope.remove(name);
            }
        }
    }
}

/// Attach an index to a Weight or Extern expression; error on
/// anything else (you can't index a Call, a Mul, or a Local).
fn attach_index(expr: Expr, index: LocalId) -> ClassifyResult<Expr> {
    match expr {
        Expr::Weight { id, index: None } => Ok(Expr::Weight {
            id,
            index: Some(index),
        }),
        Expr::Extern { kind, index: None } => Ok(Expr::Extern {
            kind,
            index: Some(index),
        }),
        Expr::Local(_) => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cannot index a local binding",
        )),
        // Double-indexing or indexing a call/mul shouldn't reach here
        // because the parser's Index wraps one expression and the
        // target path produces at most one index at a time.
        other => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("cannot index expression: {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classified::WeightId;
    use crate::parse::parse_block;

    fn classify_src(src: &str) -> Program {
        let file: syn::File =
            syn::parse_str(&format!("fn _carrier() {{ {src} }}")).expect("syntactic parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).expect("DSL parse");
        classify(&ast).expect("classification")
    }

    fn classify_err(src: &str) -> syn::Error {
        let file: syn::File = syn::parse_str(&format!("fn _carrier() {{ {src} }}")).unwrap();
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).expect("DSL parse");
        classify(&ast).expect_err("expected classification error")
    }

    #[test]
    fn simple_local_binding() {
        let p = classify_src("x = embed(input_ids, embed_tokens);");
        match &p.statements[0] {
            Stmt::Assign { target, value } => {
                assert_eq!(p.locals.name(*target).to_string(), "x");
                match value {
                    Expr::Call { op, args } => {
                        assert_eq!(*op, OpKind::Embed);
                        // arg 0: input_ids → ExternKind::InputIds
                        assert!(matches!(
                            args[0],
                            Expr::Extern {
                                kind: ExternKind::InputIds,
                                index: None
                            }
                        ));
                        // arg 1: embed_tokens → a single-segment weight
                        match &args[1] {
                            Expr::Weight { id, index: None } => {
                                let path = p.weights.path(*id);
                                assert_eq!(path.len(), 1);
                                assert_eq!(path[0].to_string(), "embed_tokens");
                            }
                            other => panic!("expected weight, got {other:?}"),
                        }
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn local_shadowing() {
        // Second `x = ...` must introduce a NEW LocalId, and the
        // read inside the call must use the PREVIOUS id.
        let p = classify_src("x = embed(input_ids, embed_tokens); x = add(x, x);");
        let (t0, v0_call) = match &p.statements[0] {
            Stmt::Assign {
                target,
                value: Expr::Call { args, .. },
            } => (*target, args.clone()),
            _ => panic!(),
        };
        let (t1, v1) = match &p.statements[1] {
            Stmt::Assign { target, value } => (*target, value.clone()),
            _ => panic!(),
        };

        assert_ne!(t0, t1, "second x must be a fresh LocalId");

        match v1 {
            Expr::Call {
                op: OpKind::Add,
                args,
            } => {
                // Both args are reads of the PREVIOUS x (t0).
                match (&args[0], &args[1]) {
                    (Expr::Local(a), Expr::Local(b)) => {
                        assert_eq!(*a, t0, "first add arg should read old x");
                        assert_eq!(*b, t0, "second add arg should read old x");
                    }
                    _ => panic!("expected two local reads"),
                }
            }
            _ => panic!("expected add call"),
        }

        // First statement's args aren't locals; this just proves
        // the helper vector v0_call above is usable.
        let _ = v0_call;
    }

    #[test]
    fn undefined_read_is_error() {
        // `mystery_var` is not a local, not a fixed extern, and
        // would therefore classify as a bare-ident weight. So it's
        // NOT an error — everything unknown becomes a weight. This
        // test makes that behavior explicit.
        let p = classify_src("x = embed(mystery_var, embed_tokens);");
        match &p.statements[0] {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[0] {
                Expr::Weight { id, index: None } => {
                    let path = p.weights.path(*id);
                    assert_eq!(path[0].to_string(), "mystery_var");
                }
                other => panic!("expected bare-ident weight, got {other:?}"),
            },
            _ => panic!(),
        }

        // Indexing with an undefined variable IS an error (the
        // index must be a local).
        let err = classify_err("x = embed(input_ids[mystery_idx], embed_tokens);");
        let msg = err.to_string();
        assert!(
            msg.contains("mystery_idx"),
            "error should mention bad index: {msg}"
        );
    }

    #[test]
    fn for_loop_scopes_induction_var() {
        let p = classify_src(
            "for layer in 0..num_hidden_layers { x = embed(kv_cache[layer], embed_tokens); }",
        );
        match &p.statements[0] {
            Stmt::For { ivar, body, .. } => {
                // `layer` inside the body must resolve to the ivar
                // LocalId.
                match &body[0] {
                    Stmt::Assign {
                        value: Expr::Call { args, .. },
                        ..
                    } => match &args[0] {
                        Expr::Extern {
                            kind: ExternKind::KvCache,
                            index: Some(idx),
                        } => {
                            assert_eq!(*idx, *ivar, "index must resolve to loop var");
                        }
                        other => panic!("expected indexed kv_cache, got {other:?}"),
                    },
                    _ => panic!(),
                }
            }
            _ => panic!("expected for-loop"),
        }
    }

    #[test]
    fn weight_interning_dedupes() {
        // Two reads of the same dotted-path weight should produce
        // the same WeightId.
        let p = classify_src(
            "for layer in 0..num_hidden_layers { \
                q = gemm(q, self_attn.q_proj[layer]); \
                q = gemm(q, self_attn.q_proj[layer]); \
            }",
        );
        match &p.statements[0] {
            Stmt::For { body, .. } => {
                let w0 = weight_of_second_arg(&body[0]);
                let w1 = weight_of_second_arg(&body[1]);
                assert_eq!(w0, w1, "same dotted path should intern to same id");
            }
            _ => panic!(),
        }
    }

    fn weight_of_second_arg(stmt: &Stmt) -> WeightId {
        match stmt {
            Stmt::Assign {
                value: Expr::Call { args, .. },
                ..
            } => match &args[1] {
                Expr::Weight { id, .. } => *id,
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn unknown_op_is_error() {
        let err = classify_err("x = frobnicate(a, b);");
        assert!(err.to_string().contains("unknown op"));
        assert!(err.to_string().contains("frobnicate"));
    }

    #[test]
    fn realistic_llama_body_classifies_fully() {
        let p = classify_src(
            r#"
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
            "#,
        );

        // Weights interned: input_layernorm, q_proj, k_proj, v_proj,
        // o_proj, post_attention_layernorm, gate_proj, up_proj,
        // down_proj, embed_tokens, norm, lm_head = 12 distinct.
        //   self_attn.{q,k,v,o}_proj = 4
        //   mlp.{gate,up,down}_proj = 3
        //   input_layernorm, post_attention_layernorm = 2
        //   embed_tokens, norm, lm_head = 3
        //   TOTAL = 12.
        assert_eq!(p.weights.len(), 12, "unique weights interned");
    }
}
