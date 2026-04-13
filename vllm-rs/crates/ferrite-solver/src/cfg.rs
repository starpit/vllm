// SPDX-License-Identifier: Apache-2.0
//! Control Flow Graph for the ferrite DSL.
//!
//! Built from the AST ([`crate::parse::MegakernelDef`]). Each basic
//! block holds a sequence of straight-line statements. Terminators
//! encode control flow (jump, branch, return).
//!
//! Today the DSL has exactly one control-flow construct: `for` loops
//! with compile-time bounds. The CFG builder lowers these to a
//! header block (loop test) + body block (with a back edge) + exit
//! block. Loop detection and unrolling operate on this CFG
//! generically — they don't know about DSL-level "for" syntax.

use std::collections::BTreeMap;

use syn::Ident;

use crate::parse::{AssignStmt, LetStmt, LetTupleStmt, OpCall, Stmt};

/// Dense index into [`Cfg::blocks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// One instruction inside a basic block. These are the AST's simple
/// statements (Let, LetTuple, Assign) with their computations.
#[derive(Clone, Debug)]
pub enum Instr {
    /// `let x = op(args...);`
    Let { name: Ident, call: OpCall },
    /// `let (a, b, c) = op(args...);`
    LetTuple { names: Vec<Ident>, call: OpCall },
    /// `x = op(args...);`
    Assign { target: Ident, call: OpCall },
}

/// Control-flow successor of a basic block.
#[derive(Clone, Debug)]
pub enum Terminator {
    /// Unconditional jump to one block.
    Jump(BlockId),
    /// Conditional branch: `if var < end { then_block } else { else_block }`.
    /// The condition form is `var < end`, the only form our DSL
    /// produces (loop header test). `var` and `end` are identifiers
    /// in the DSL's namespace (loop variable, range bound).
    LoopHeader {
        var: Ident,
        end: Ident,
        body: BlockId,
        exit: BlockId,
    },
    /// No successor — end of the kernel.
    Return,
}

/// A basic block: a sequence of instructions followed by a terminator.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instrs: Vec<Instr>,
    pub term: Terminator,
}

/// The control-flow graph.
#[derive(Clone, Debug)]
pub struct Cfg {
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    /// Loop bounds (compile-time): `range_end` name → concrete value.
    /// E.g. `NL → 32` for Llama 1B.
    pub loop_bounds: BTreeMap<String, usize>,
}

impl Cfg {
    /// All successor blocks of `b`.
    pub fn successors(&self, b: BlockId) -> Vec<BlockId> {
        match &self.blocks[b.0 as usize].term {
            Terminator::Jump(t) => vec![*t],
            Terminator::LoopHeader { body, exit, .. } => vec![*body, *exit],
            Terminator::Return => vec![],
        }
    }

    /// Predecessors of `b` (computed by walking every block's terminator).
    pub fn predecessors(&self, b: BlockId) -> Vec<BlockId> {
        let mut preds = Vec::new();
        for block in &self.blocks {
            for succ in self.successors(block.id) {
                if succ == b {
                    preds.push(block.id);
                }
            }
        }
        preds
    }
}

// ── AST → CFG ─────────────────────────────────────────────────────

struct Builder {
    blocks: Vec<BasicBlock>,
}

impl Builder {
    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock {
            id,
            instrs: Vec::new(),
            term: Terminator::Return, // placeholder; lower_stmts overwrites
        });
        id
    }

    fn push_instr(&mut self, block: BlockId, instr: Instr) {
        self.blocks[block.0 as usize].instrs.push(instr);
    }

    fn set_term(&mut self, block: BlockId, term: Terminator) {
        self.blocks[block.0 as usize].term = term;
    }

    /// Lower a sequence of statements into `current` block. Returns
    /// the final block ID (either `current` or a new block created
    /// by a loop). The returned block has its terminator unset (or
    /// set to Return as placeholder) — caller is responsible.
    fn lower_stmts(&mut self, stmts: &[Stmt], mut current: BlockId) -> BlockId {
        for stmt in stmts {
            match stmt {
                Stmt::Let(LetStmt { name, call }) => {
                    self.push_instr(
                        current,
                        Instr::Let {
                            name: name.clone(),
                            call: call.clone(),
                        },
                    );
                }
                Stmt::LetTuple(LetTupleStmt { names, call }) => {
                    self.push_instr(
                        current,
                        Instr::LetTuple {
                            names: names.clone(),
                            call: call.clone(),
                        },
                    );
                }
                Stmt::Assign(AssignStmt { target, call }) => {
                    self.push_instr(
                        current,
                        Instr::Assign {
                            target: target.clone(),
                            call: call.clone(),
                        },
                    );
                }
                Stmt::ForLoop(for_loop) => {
                    // Loop lowering: current block jumps to header.
                    // Header branches to body (back-edge) or exit.
                    // Body ends with jump back to header.
                    let header = self.new_block();
                    let body_entry = self.new_block();
                    let exit = self.new_block();

                    self.set_term(current, Terminator::Jump(header));
                    self.set_term(
                        header,
                        Terminator::LoopHeader {
                            var: for_loop.var.clone(),
                            end: for_loop.range_end.clone(),
                            body: body_entry,
                            exit,
                        },
                    );

                    let body_final = self.lower_stmts(&for_loop.body, body_entry);
                    // body_final jumps back to the header (back edge).
                    self.set_term(body_final, Terminator::Jump(header));

                    // Continue lowering after the loop in `exit`.
                    current = exit;
                }
            }
        }
        current
    }
}

/// Build the CFG from a parsed [`MegakernelDef`].
pub fn build_cfg(def: &crate::parse::MegakernelDef) -> Cfg {
    let mut builder = Builder { blocks: Vec::new() };
    let entry = builder.new_block();
    let last = builder.lower_stmts(&def.body, entry);
    builder.set_term(last, Terminator::Return);

    let loop_bounds: BTreeMap<String, usize> = def
        .params
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

    Cfg {
        blocks: builder.blocks,
        entry,
        loop_bounds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::MegakernelDef;

    fn parse_dsl(src: &str) -> MegakernelDef {
        let tokens: proc_macro2::TokenStream = src.parse().unwrap();
        syn::parse2(tokens).unwrap()
    }

    #[test]
    fn cfg_simple_loop() {
        let def = parse_dsl(
            r#"
            kernel test<NL=3, HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    hidden_states = gemm(normed, weight[layer]);
                }
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);

        // Expect at least: entry (pre-loop) + header + body + exit (post-loop).
        assert!(
            cfg.blocks.len() >= 4,
            "too few blocks: {}",
            cfg.blocks.len()
        );

        // Entry should have the embed assign and jump to a header.
        assert!(matches!(
            cfg.blocks[cfg.entry.0 as usize].term,
            Terminator::Jump(_)
        ));

        // Exactly one LoopHeader terminator.
        let headers: Vec<_> = cfg
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .collect();
        assert_eq!(headers.len(), 1, "expected one loop header");

        // Loop bound is recorded.
        assert_eq!(cfg.loop_bounds.get("NL"), Some(&3));
    }

    #[test]
    fn cfg_two_sequential_loops() {
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

        // Two loop headers.
        let headers: Vec<_> = cfg
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .collect();
        assert_eq!(headers.len(), 2, "expected two loop headers");

        // Both bounds recorded.
        assert_eq!(cfg.loop_bounds.get("NL"), Some(&2));
        assert_eq!(cfg.loop_bounds.get("NH"), Some(&3));
    }

    #[test]
    fn cfg_no_loop() {
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);

        // No loop headers.
        let headers: Vec<_> = cfg
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .collect();
        assert_eq!(headers.len(), 0);

        // Entry block ends in Return.
        assert!(matches!(
            cfg.blocks[cfg.entry.0 as usize].term,
            Terminator::Return
        ));
    }
}
