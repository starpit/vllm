// SPDX-License-Identifier: Apache-2.0
//! CFG analyses: dominators and natural loop detection.
//!
//! Standard compiler-theory textbook stuff. The algorithms don't
//! know anything about the DSL — they operate on abstract CFGs.

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::{BlockId, Cfg, Terminator};

/// Dominator tree: for each block, its immediate dominator.
/// The entry block dominates itself (idom[entry] = entry).
#[derive(Clone, Debug)]
pub struct Dominators {
    /// `idom[b]` = immediate dominator of `b`.
    pub idom: BTreeMap<BlockId, BlockId>,
}

impl Dominators {
    /// Compute dominators for the given CFG using the simple
    /// iterative algorithm (Cooper-Harvey-Kennedy).
    pub fn compute(cfg: &Cfg) -> Self {
        // Reverse-postorder traversal from the entry.
        let rpo = reverse_postorder(cfg);
        let rpo_pos: BTreeMap<BlockId, usize> =
            rpo.iter().enumerate().map(|(i, b)| (*b, i)).collect();

        let mut idom: BTreeMap<BlockId, Option<BlockId>> =
            cfg.blocks.iter().map(|b| (b.id, None)).collect();
        idom.insert(cfg.entry, Some(cfg.entry));

        let mut changed = true;
        while changed {
            changed = false;
            // Skip the entry block (first in RPO).
            for &b in rpo.iter().skip(1) {
                // Pick first predecessor that has a dominator set.
                let preds = cfg.predecessors(b);
                let mut new_idom: Option<BlockId> = None;
                for &p in &preds {
                    if idom[&p].is_some() {
                        if let Some(cur) = new_idom {
                            new_idom = Some(intersect(p, cur, &idom, &rpo_pos));
                        } else {
                            new_idom = Some(p);
                        }
                    }
                }
                if idom[&b] != new_idom && new_idom.is_some() {
                    idom.insert(b, new_idom);
                    changed = true;
                }
            }
        }

        let final_idom: BTreeMap<BlockId, BlockId> = idom
            .into_iter()
            .filter_map(|(k, v)| v.map(|d| (k, d)))
            .collect();

        Dominators { idom: final_idom }
    }

    /// Does `dominator` dominate `b` (including `b == dominator`)?
    pub fn dominates(&self, dominator: BlockId, b: BlockId) -> bool {
        let mut cur = b;
        loop {
            if cur == dominator {
                return true;
            }
            let idom = match self.idom.get(&cur) {
                Some(d) => *d,
                None => return false,
            };
            if idom == cur {
                // Reached entry without finding `dominator`.
                return false;
            }
            cur = idom;
        }
    }
}

fn intersect(
    mut b1: BlockId,
    mut b2: BlockId,
    idom: &BTreeMap<BlockId, Option<BlockId>>,
    rpo_pos: &BTreeMap<BlockId, usize>,
) -> BlockId {
    while b1 != b2 {
        while rpo_pos[&b1] > rpo_pos[&b2] {
            b1 = idom[&b1].unwrap_or(b1);
            if rpo_pos.get(&b1).copied() == rpo_pos.get(&b2).copied() {
                break;
            }
        }
        while rpo_pos[&b2] > rpo_pos[&b1] {
            b2 = idom[&b2].unwrap_or(b2);
            if rpo_pos.get(&b1).copied() == rpo_pos.get(&b2).copied() {
                break;
            }
        }
    }
    b1
}

fn reverse_postorder(cfg: &Cfg) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut visited: BTreeSet<BlockId> = BTreeSet::new();
    postorder_visit(cfg, cfg.entry, &mut visited, &mut order);
    order.reverse();
    order
}

fn postorder_visit(cfg: &Cfg, b: BlockId, visited: &mut BTreeSet<BlockId>, out: &mut Vec<BlockId>) {
    if !visited.insert(b) {
        return;
    }
    for succ in cfg.successors(b) {
        postorder_visit(cfg, succ, visited, out);
    }
    out.push(b);
}

/// A natural loop: a back edge plus all blocks inside.
#[derive(Clone, Debug)]
pub struct Loop {
    /// The loop header (target of the back edge, which dominates its body).
    pub header: BlockId,
    /// The back-edge source — typically the block with "jump back to header".
    pub latch: BlockId,
    /// All blocks in the loop body (including header and latch).
    pub body: BTreeSet<BlockId>,
}

/// Find all natural loops in the CFG.
///
/// A natural loop is identified by a back edge `u → v` where `v`
/// dominates `u`. The loop body is the set of blocks that can
/// reach `u` via predecessor edges without passing through `v`'s
/// predecessors (excluding `v` itself).
pub fn find_loops(cfg: &Cfg, dom: &Dominators) -> Vec<Loop> {
    let mut loops = Vec::new();
    for block in &cfg.blocks {
        let u = block.id;
        for v in cfg.successors(u) {
            if dom.dominates(v, u) {
                // Back edge u → v. Build the natural loop.
                let body = natural_loop_body(cfg, u, v);
                loops.push(Loop {
                    header: v,
                    latch: u,
                    body,
                });
            }
        }
    }
    loops
}

fn natural_loop_body(cfg: &Cfg, latch: BlockId, header: BlockId) -> BTreeSet<BlockId> {
    // Standard algorithm: DFS backward from `latch`, stopping at `header`.
    let mut body = BTreeSet::new();
    body.insert(header);
    body.insert(latch);
    let mut stack = vec![latch];
    while let Some(b) = stack.pop() {
        for pred in cfg.predecessors(b) {
            if body.insert(pred) {
                stack.push(pred);
            }
        }
    }
    body
}

/// Get the loop variable and bound for a natural loop, if the header's
/// terminator is a `LoopHeader` (our DSL's only loop construct).
pub fn loop_header_info(cfg: &Cfg, l: &Loop) -> Option<(syn::Ident, syn::Ident)> {
    match &cfg.blocks[l.header.0 as usize].term {
        Terminator::LoopHeader { var, end, .. } => Some((var.clone(), end.clone())),
        _ => None,
    }
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
    fn dominators_simple() {
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
        let dom = Dominators::compute(&cfg);

        // Entry dominates everything.
        for b in &cfg.blocks {
            assert!(
                dom.dominates(cfg.entry, b.id),
                "entry should dominate {:?}",
                b.id
            );
        }
    }

    #[test]
    fn find_loops_single() {
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
        let dom = Dominators::compute(&cfg);
        let loops = find_loops(&cfg, &dom);

        assert_eq!(loops.len(), 1, "expected one natural loop");
        let l = &loops[0];
        // Header should be the LoopHeader block.
        let (var, end) = loop_header_info(&cfg, l).expect("header should be a LoopHeader");
        assert_eq!(var.to_string(), "layer");
        assert_eq!(end.to_string(), "NL");
    }

    #[test]
    fn find_loops_two_sequential() {
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
        let dom = Dominators::compute(&cfg);
        let loops = find_loops(&cfg, &dom);

        assert_eq!(loops.len(), 2, "expected two natural loops");
        // Both loops should have different headers.
        assert_ne!(loops[0].header, loops[1].header);
    }

    #[test]
    fn find_loops_none() {
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let dom = Dominators::compute(&cfg);
        let loops = find_loops(&cfg, &dom);

        assert_eq!(loops.len(), 0);
    }
}
