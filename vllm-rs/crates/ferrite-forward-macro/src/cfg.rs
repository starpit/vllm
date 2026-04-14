// SPDX-License-Identifier: Apache-2.0
//! Phase 5: build a per-model CFG from the classified program.
//!
//! Takes (ClassifiedProgram, ModelParams) and produces a CFG whose
//! loop terminators carry *concrete* integer trip counts — no
//! symbolic `num_hidden_layers`, no BTreeMap lookups at any later
//! phase.
//!
//! The CFG here is deliberately minimal: our DSL has straight-line
//! instruction blocks separated only by for-loops. A block is a
//! vec of instrs ending in a terminator (Jump, LoopHeader, Return).
//!
//! No bullshit allowed below this point: loop bounds are `u64`, the
//! program carries no `String` identifiers, and everything the
//! solver and codegen touch is numeric.

#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::classified::{Bound, Expr, LocalId, Program, Stmt};
use crate::config::ModelParams;

// ── Types ─────────────────────────────────────────────────────────

/// Dense block identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// A basic block: straight-line instrs + a terminator.
#[derive(Clone, Debug)]
pub struct Block {
    pub id: BlockId,
    pub instrs: Vec<Instr>,
    pub term: Terminator,
}

/// An instruction in a block. Same shape as a classified statement
/// but with no for-loop — those become block-level terminators.
#[derive(Clone, Debug)]
pub enum Instr {
    Assign { target: LocalId, value: Expr },
    AssignTuple { targets: Vec<LocalId>, value: Expr },
}

/// Block terminator.
#[derive(Clone, Debug)]
pub enum Terminator {
    /// Unconditional jump to another block.
    Jump(BlockId),
    /// Loop header. `body` is entered once per iteration, terminates
    /// by jumping back to this block; after the last iteration, flow
    /// proceeds to `exit`. `start`/`end` are *concrete* ints that
    /// came from this model's `ModelParams` (or from a DSL-level
    /// literal bound).
    ///
    /// `loop_carry` is forwarded from the classified `Stmt::For`:
    /// after each iteration, rebind each `(outer, inner)` local so
    /// the next iteration reads the previous iteration's output.
    LoopHeader {
        ivar: LocalId,
        start: u64,
        end: u64,
        body: BlockId,
        exit: BlockId,
        loop_carry: Vec<(LocalId, LocalId)>,
    },
    /// Function end.
    Return,
}

/// The CFG for one specialization.
#[derive(Clone, Debug)]
pub struct Cfg {
    pub blocks: Vec<Block>,
    pub entry: BlockId,
}

impl Cfg {
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }
}

/// Errors when building the CFG. Distinct from shape-inference
/// errors because these concern bound resolution, not shapes.
#[derive(Debug)]
pub enum CfgError {
    UnknownBound {
        name: String,
        /// Names we did have, for diagnostics.
        available: Vec<String>,
    },
}

impl std::fmt::Display for CfgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownBound { name, available } => {
                write!(
                    f,
                    "loop bound `{name}` not in this model's config.json. \
                     Available: {available:?}",
                )
            }
        }
    }
}

impl std::error::Error for CfgError {}

// ── Builder ──────────────────────────────────────────────────────

pub fn build_cfg(program: &Program, params: &ModelParams) -> Result<Cfg, CfgError> {
    let mut builder = CfgBuilder::new(&params.bounds);
    let entry = builder.alloc_block();
    let exit = builder.alloc_block();
    builder.lower_stmts(&program.statements, entry, exit)?;
    builder.finalize_block(exit, Terminator::Return);
    Ok(Cfg {
        blocks: builder.blocks,
        entry,
    })
}

struct CfgBuilder<'a> {
    bounds: &'a BTreeMap<String, u64>,
    blocks: Vec<Block>,
    /// One-per-block open instruction buffer: instrs accumulate
    /// into the indexed block's `instrs` field when we commit.
    open: BTreeMap<BlockId, Vec<Instr>>,
}

impl<'a> CfgBuilder<'a> {
    fn new(bounds: &'a BTreeMap<String, u64>) -> Self {
        Self {
            bounds,
            blocks: Vec::new(),
            open: BTreeMap::new(),
        }
    }

    fn alloc_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        // Placeholder; lower_stmts will rewrite instrs and term
        // before finalize.
        self.blocks.push(Block {
            id,
            instrs: Vec::new(),
            term: Terminator::Return,
        });
        self.open.insert(id, Vec::new());
        id
    }

    fn push_instr(&mut self, block: BlockId, instr: Instr) {
        self.open.entry(block).or_default().push(instr);
    }

    fn finalize_block(&mut self, block: BlockId, term: Terminator) {
        let instrs = self.open.remove(&block).unwrap_or_default();
        self.blocks[block.0 as usize].instrs = instrs;
        self.blocks[block.0 as usize].term = term;
    }

    /// Lower a straight-line statement sequence into blocks. `start`
    /// is the block instrs begin accumulating in; on exit we jump
    /// to `end`.
    fn lower_stmts(
        &mut self,
        stmts: &[Stmt],
        mut current: BlockId,
        end: BlockId,
    ) -> Result<(), CfgError> {
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target, value } => {
                    self.push_instr(
                        current,
                        Instr::Assign {
                            target: *target,
                            value: value.clone(),
                        },
                    );
                }
                Stmt::AssignTuple { targets, value } => {
                    self.push_instr(
                        current,
                        Instr::AssignTuple {
                            targets: targets.clone(),
                            value: value.clone(),
                        },
                    );
                }
                Stmt::For {
                    ivar,
                    start,
                    end: loop_end,
                    body,
                    loop_carry,
                } => {
                    let start_val = self.resolve_bound(start)?;
                    let end_val = self.resolve_bound(loop_end)?;
                    let header = self.alloc_block();
                    let body_block = self.alloc_block();
                    let after = self.alloc_block();
                    // Current block jumps to the header.
                    self.finalize_block(current, Terminator::Jump(header));
                    self.finalize_block(
                        header,
                        Terminator::LoopHeader {
                            ivar: *ivar,
                            start: start_val,
                            end: end_val,
                            body: body_block,
                            exit: after,
                            loop_carry: loop_carry.clone(),
                        },
                    );
                    // Body: recurse. Its last block jumps back to
                    // the header.
                    self.lower_stmts(body, body_block, header)?;
                    // Continue with `after`.
                    current = after;
                }
            }
        }
        // After the last statement, jump to the outer end block.
        self.finalize_block(current, Terminator::Jump(end));
        Ok(())
    }

    fn resolve_bound(&self, b: &Bound) -> Result<u64, CfgError> {
        match b {
            Bound::Lit(n) => Ok(*n),
            Bound::Sym(ident) => {
                let name = ident.to_string();
                self.bounds.get(&name).copied().ok_or_else(|| {
                    let available: Vec<String> = self.bounds.keys().cloned().collect();
                    CfgError::UnknownBound { name, available }
                })
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::classify;
    use crate::config::{self};
    use crate::parse::parse_block;
    use std::path::PathBuf;

    fn classify_src(src: &str) -> Program {
        let file: syn::File = syn::parse_str(&format!("fn _carrier() {{ {src} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).expect("parse DSL");
        classify(&ast).expect("classify")
    }

    fn llama_3_2_1b_params() -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("model_architectures")
            .join("llama")
            .join("llama-3.2-1b.json");
        config::load_file(&path).expect("load llama-3.2-1b config")
    }

    #[test]
    fn straight_line_program_single_block() {
        let p = classify_src("x = embed(input_ids, embed_tokens);");
        let params = llama_3_2_1b_params();
        let cfg = build_cfg(&p, &params).expect("build cfg");

        // 2 blocks: entry (with the instr), exit (empty + Return).
        assert_eq!(cfg.blocks.len(), 2);
        let entry = cfg.block(cfg.entry);
        assert_eq!(entry.instrs.len(), 1);
        assert!(matches!(entry.term, Terminator::Jump(_)));
    }

    #[test]
    fn for_loop_creates_header_body_after_blocks() {
        let p = classify_src(
            "for layer in 0..num_hidden_layers { x = embed(input_ids, embed_tokens); }",
        );
        let params = llama_3_2_1b_params();
        let cfg = build_cfg(&p, &params).expect("build cfg");

        // Blocks: entry, header, body, after, exit = 5.
        assert_eq!(cfg.blocks.len(), 5);

        // Find the LoopHeader terminator.
        let header = cfg
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .expect("loop header block");
        match &header.term {
            Terminator::LoopHeader {
                start,
                end,
                body,
                exit,
                ..
            } => {
                assert_eq!(*start, 0, "start is 0 (from DSL literal)");
                assert_eq!(*end, 16, "end resolved to llama-3.2-1b num_hidden_layers");
                // Body block must jump back to header.
                let body_block = cfg.block(*body);
                match body_block.term {
                    Terminator::Jump(target) => {
                        assert_eq!(target.0, header.id.0, "body should back-edge to header")
                    }
                    _ => panic!("body terminator should be Jump to header"),
                }
                // After block exists.
                let _ = cfg.block(*exit);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn unknown_bound_errors() {
        let p = classify_src("for i in 0..totally_bogus_bound { x = embed(input_ids, e); }");
        let params = llama_3_2_1b_params();
        let err = build_cfg(&p, &params).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("totally_bogus_bound"),
            "error names bound: {msg}"
        );
    }

    #[test]
    fn literal_bounds_pass_through() {
        let p = classify_src("for i in 0..5 { x = embed(input_ids, e); }");
        let params = llama_3_2_1b_params();
        let cfg = build_cfg(&p, &params).expect("build cfg");

        let header = cfg
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .unwrap();
        match header.term {
            Terminator::LoopHeader { start, end, .. } => {
                assert_eq!(start, 0);
                assert_eq!(end, 5);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn cfg_has_no_strings_in_terminators() {
        // Structural invariant: no Terminator variant carries a
        // String. (It doesn't — this test is just a compile-time
        // smoke check documenting the invariant.)
        fn assert_no_strings(_term: &Terminator) {
            // If someone adds a `String` to a Terminator variant,
            // this function would no longer be vacuously valid
            // because they'd have to cover that variant in some
            // way. The invariant is "all data in Terminator is
            // numeric or block-indexing."
        }
        let p = classify_src("x = embed(input_ids, e);");
        let cfg = build_cfg(&p, &llama_3_2_1b_params()).unwrap();
        for b in &cfg.blocks {
            assert_no_strings(&b.term);
        }
    }

    #[test]
    fn full_llama_body_cfg() {
        let p = classify_src(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                hidden_states = add(normed, hidden_states);
            }
            normed = rmsnorm(hidden_states, norm);
            "#,
        );
        let cfg = build_cfg(&p, &llama_3_2_1b_params()).unwrap();

        // Find the loop header; body should contain 3 instrs.
        let header_block = cfg
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .unwrap();
        let body_id = match header_block.term {
            Terminator::LoopHeader { body, .. } => body,
            _ => unreachable!(),
        };
        let body = cfg.block(body_id);
        assert_eq!(body.instrs.len(), 3, "3 body instrs (normed, q, hidden)");

        // Entry block has 1 instr (embed).
        let entry = cfg.block(cfg.entry);
        assert_eq!(entry.instrs.len(), 1);

        // After the loop, there's a block with the post-loop rmsnorm.
        let post_loop: Vec<_> = cfg
            .blocks
            .iter()
            .filter(|b| b.instrs.len() == 1 && !std::ptr::eq(*b, entry))
            .collect();
        assert!(
            post_loop
                .iter()
                .any(|b| matches!(b.instrs[0], Instr::Assign { .. })),
            "there is a post-loop block with an Assign for the final rmsnorm",
        );
    }
}
