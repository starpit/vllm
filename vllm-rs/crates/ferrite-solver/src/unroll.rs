// SPDX-License-Identifier: Apache-2.0
//! Loop unrolling: CFG → flat sequence of instructions with concrete
//! references.
//!
//! For each natural loop with a compile-time-known trip count, we
//! duplicate the loop body N times. Inside each copy, references
//! to the loop variable are substituted with the concrete iteration
//! index. The result is a loop-free instruction sequence — the
//! Fully Unrolled Forward (FUF).

use std::collections::BTreeSet;

use crate::cfg::{BlockId, Cfg, Instr, Terminator};
use crate::cfg_analysis::{Dominators, Loop, find_loops};
use crate::parse::{Arg, OpCall};

/// Error produced by unrolling.
#[derive(Debug)]
pub enum UnrollError {
    /// A natural loop's bound identifier (e.g. `NL`) has no value in
    /// `cfg.loop_bounds`.
    UnknownLoopBound { var: String, end: String },
    /// A reference uses an identifier that isn't the enclosing loop's
    /// variable (and we don't have a scope analysis yet to check
    /// whether it's defined in an outer scope).
    ///
    /// For now, we treat any unresolved index identifier as an error.
    UnresolvedIndex { buf: String, index: String },
    /// Unsupported CFG shape (e.g. arbitrary branches we can't
    /// unroll with the current algorithm).
    Unsupported(String),
}

/// Which phase of the unrolled program an instruction belongs to.
///
/// Tags attached by [`unroll_tagged`] so downstream passes (namely
/// [`crate::fuf::build_fuf`]) can recover the loop iteration a
/// particular instruction came from. We need this because the
/// instruction stream itself has no trace of "which `for` body
/// expansion produced me" after substitution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopPhase {
    /// Instruction appears before any loop has been entered.
    PreLoop,
    /// Instruction is a copy of a loop body, at the given iteration.
    InLoop { iter: u16 },
    /// Instruction appears after a loop has been exited.
    PostLoop,
}

/// Unroll all natural loops in the CFG.
///
/// Returns a flat sequence of instructions in execution order,
/// with all symbolic indexed references (`Arg::Var(name, Some(id))`)
/// replaced by concrete `Arg::VarAt(name, idx)`.
pub fn unroll(cfg: &Cfg) -> Result<Vec<Instr>, UnrollError> {
    Ok(unroll_tagged(cfg)?.into_iter().map(|(i, _)| i).collect())
}

/// Like [`unroll`] but also tags each instruction with its
/// [`LoopPhase`].
pub fn unroll_tagged(cfg: &Cfg) -> Result<Vec<(Instr, LoopPhase)>, UnrollError> {
    let dom = Dominators::compute(cfg);
    let loops = find_loops(cfg, &dom);

    // Build a map: loop-header BlockId → Loop.
    let loop_by_header: std::collections::BTreeMap<BlockId, &Loop> =
        loops.iter().map(|l| (l.header, l)).collect();

    let mut out = Vec::new();
    let mut visited: BTreeSet<BlockId> = BTreeSet::new();
    let mut have_looped = false;
    walk(
        cfg,
        cfg.entry,
        &loop_by_header,
        &mut visited,
        &mut have_looped,
        &mut out,
    )?;
    Ok(out)
}

/// Walk the CFG in execution order, emitting tagged instructions.
/// When we hit a loop header, we unroll its body and skip the
/// corresponding loop blocks in the linear walk.
fn walk(
    cfg: &Cfg,
    block: BlockId,
    loops: &std::collections::BTreeMap<BlockId, &Loop>,
    visited: &mut BTreeSet<BlockId>,
    have_looped: &mut bool,
    out: &mut Vec<(Instr, LoopPhase)>,
) -> Result<(), UnrollError> {
    let mut cur = block;
    loop {
        if !visited.insert(cur) {
            return Ok(());
        }
        let b = &cfg.blocks[cur.0 as usize];

        if let Some(l) = loops.get(&cur) {
            // This is a loop header. Unroll its body.
            let (var, end) = match &b.term {
                Terminator::LoopHeader { var, end, .. } => (var.clone(), end.clone()),
                _ => {
                    return Err(UnrollError::Unsupported(format!(
                        "expected LoopHeader terminator at block {cur:?}",
                    )));
                }
            };
            let bound = *cfg.loop_bounds.get(&end.to_string()).ok_or_else(|| {
                UnrollError::UnknownLoopBound {
                    var: var.to_string(),
                    end: end.to_string(),
                }
            })?;

            unroll_loop(cfg, l, &var, bound, out)?;
            *have_looped = true;

            // Mark all loop body blocks as visited so the linear
            // walk skips them.
            for bid in &l.body {
                visited.insert(*bid);
            }

            // Continue from the loop's exit block.
            let exit = match &b.term {
                Terminator::LoopHeader { exit, .. } => *exit,
                _ => unreachable!(),
            };
            cur = exit;
            continue;
        }

        // Non-loop block: emit its instructions tagged by phase.
        let phase = if *have_looped {
            LoopPhase::PostLoop
        } else {
            LoopPhase::PreLoop
        };
        for instr in &b.instrs {
            out.push((substitute_instr(instr, None, 0)?, phase));
        }
        match &b.term {
            Terminator::Jump(next) => {
                cur = *next;
            }
            Terminator::Return => {
                return Ok(());
            }
            Terminator::LoopHeader { .. } => {
                // Already handled above.
                unreachable!("LoopHeader should have been matched by loops.get")
            }
        }
    }
}

/// Unroll one natural loop `bound` times, emitting the body's
/// instructions with the loop variable substituted and an
/// [`LoopPhase::InLoop`] tag for the iteration index.
fn unroll_loop(
    cfg: &Cfg,
    l: &Loop,
    var: &syn::Ident,
    bound: usize,
    out: &mut Vec<(Instr, LoopPhase)>,
) -> Result<(), UnrollError> {
    // Determine the body blocks in execution order: start at the
    // header's body successor, walk until we hit the latch (which
    // jumps back to the header).
    let body_entry = match &cfg.blocks[l.header.0 as usize].term {
        Terminator::LoopHeader { body, .. } => *body,
        _ => {
            return Err(UnrollError::Unsupported(format!(
                "loop header block {:?} has no LoopHeader terminator",
                l.header,
            )));
        }
    };

    let body_order = body_linear_order(cfg, body_entry, l)?;

    for i in 0..bound {
        let phase = LoopPhase::InLoop { iter: i as u16 };
        for bid in &body_order {
            let b = &cfg.blocks[bid.0 as usize];
            for instr in &b.instrs {
                out.push((substitute_instr(instr, Some(var), i)?, phase));
            }
        }
    }
    Ok(())
}

/// Find the body blocks in linear execution order, starting from
/// `entry` and walking jumps until we hit the latch (back-edge source).
fn body_linear_order(cfg: &Cfg, entry: BlockId, l: &Loop) -> Result<Vec<BlockId>, UnrollError> {
    let mut order = Vec::new();
    let mut cur = entry;
    let mut seen: BTreeSet<BlockId> = BTreeSet::new();
    loop {
        if !seen.insert(cur) {
            return Err(UnrollError::Unsupported(format!(
                "cycle within loop body at block {cur:?}",
            )));
        }
        order.push(cur);
        let b = &cfg.blocks[cur.0 as usize];
        match &b.term {
            Terminator::Jump(t) => {
                if *t == l.header {
                    // Back edge. Done walking the body.
                    return Ok(order);
                }
                cur = *t;
            }
            Terminator::Return => {
                return Err(UnrollError::Unsupported(format!(
                    "Return terminator inside loop body at block {cur:?}",
                )));
            }
            Terminator::LoopHeader { .. } => {
                return Err(UnrollError::Unsupported(format!(
                    "nested loops not yet supported at block {cur:?}",
                )));
            }
        }
    }
}

fn substitute_instr(
    instr: &Instr,
    loop_var: Option<&syn::Ident>,
    concrete: usize,
) -> Result<Instr, UnrollError> {
    Ok(match instr {
        Instr::Let { name, call } => Instr::Let {
            name: name.clone(),
            call: substitute_call(call, loop_var, concrete)?,
        },
        Instr::LetTuple { names, call } => Instr::LetTuple {
            names: names.clone(),
            call: substitute_call(call, loop_var, concrete)?,
        },
        Instr::Assign { target, call } => Instr::Assign {
            target: target.clone(),
            call: substitute_call(call, loop_var, concrete)?,
        },
    })
}

fn substitute_call(
    call: &OpCall,
    loop_var: Option<&syn::Ident>,
    concrete: usize,
) -> Result<OpCall, UnrollError> {
    Ok(OpCall {
        op: call.op.clone(),
        args: call
            .args
            .iter()
            .map(|a| substitute_arg(a, loop_var, concrete))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn substitute_arg(
    arg: &Arg,
    loop_var: Option<&syn::Ident>,
    concrete: usize,
) -> Result<Arg, UnrollError> {
    Ok(match arg {
        Arg::Var(name, None) => Arg::Var(name.clone(), None),
        Arg::Var(name, Some(idx)) => {
            let idx_str = idx.to_string();
            match loop_var {
                Some(lv) if lv == idx => Arg::VarAt(name.clone(), concrete),
                _ => {
                    // Unresolved index identifier. Either the DSL
                    // references a loop var that doesn't exist in
                    // the enclosing scope, or we're outside a loop
                    // entirely. Return error.
                    return Err(UnrollError::UnresolvedIndex {
                        buf: name.clone(),
                        index: idx_str,
                    });
                }
            }
        }
        Arg::VarAt(name, idx) => Arg::VarAt(name.clone(), *idx),
        Arg::Call(c) => Arg::Call(substitute_call(c, loop_var, concrete)?),
        Arg::Mul(a, b) => Arg::Mul(
            Box::new(substitute_arg(a, loop_var, concrete)?),
            Box::new(substitute_arg(b, loop_var, concrete)?),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::parse::MegakernelDef;

    fn parse_dsl(src: &str) -> MegakernelDef {
        let tokens: proc_macro2::TokenStream = src.parse().unwrap();
        syn::parse2(tokens).unwrap()
    }

    #[test]
    fn unroll_single_loop() {
        let def = parse_dsl(
            r#"
            kernel test<NL=3, HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                for layer in 0..NL {
                    hidden_states = gemm(hidden_states, w[layer]);
                }
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let instrs = unroll(&cfg).expect("unroll succeeded");

        // Expect: 1 embed (Assign) + 3 gemm iterations (Assign) + 1 lm_head (Assign) = 5.
        assert_eq!(
            instrs.len(),
            5,
            "expected 5 instructions, got {}",
            instrs.len()
        );

        // The 3 gemm iterations should reference w[0], w[1], w[2].
        let mut w_indices = Vec::new();
        for instr in &instrs {
            let call = match instr {
                Instr::Let { call, .. }
                | Instr::LetTuple { call, .. }
                | Instr::Assign { call, .. } => call,
            };
            for a in &call.args {
                if let Arg::VarAt(name, idx) = a
                    && name == "w"
                {
                    w_indices.push(*idx);
                }
            }
        }
        assert_eq!(w_indices, vec![0, 1, 2]);
    }

    #[test]
    fn unroll_two_sequential_loops() {
        let def = parse_dsl(
            r#"
            kernel test<NL=2, NH=3, HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                for l in 0..NL {
                    hidden_states = gemm(hidden_states, w1[l]);
                }
                for h in 0..NH {
                    hidden_states = gemm(hidden_states, w2[h]);
                }
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let instrs = unroll(&cfg).expect("unroll succeeded");

        // 1 embed + 2 w1 + 3 w2 + 1 lm_head = 7.
        assert_eq!(instrs.len(), 7);

        let mut w1 = Vec::new();
        let mut w2 = Vec::new();
        for instr in &instrs {
            let call = match instr {
                Instr::Let { call, .. }
                | Instr::LetTuple { call, .. }
                | Instr::Assign { call, .. } => call,
            };
            for a in &call.args {
                if let Arg::VarAt(name, idx) = a {
                    match name.as_str() {
                        "w1" => w1.push(*idx),
                        "w2" => w2.push(*idx),
                        _ => {}
                    }
                }
            }
        }
        assert_eq!(w1, vec![0, 1]);
        assert_eq!(w2, vec![0, 1, 2]);
    }

    #[test]
    fn unroll_undefined_index_errors() {
        let def = parse_dsl(
            r#"
            kernel test<NL=2, HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                for l in 0..NL {
                    hidden_states = gemm(hidden_states, w[layer]);
                }
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let err = unroll(&cfg).unwrap_err();
        match err {
            UnrollError::UnresolvedIndex { buf, index } => {
                assert_eq!(buf, "w");
                assert_eq!(index, "layer");
            }
            other => panic!("expected UnresolvedIndex, got {other:?}"),
        }
    }

    #[test]
    fn unroll_no_loops() {
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let instrs = unroll(&cfg).expect("unroll succeeded");
        assert_eq!(instrs.len(), 2);
    }
}
