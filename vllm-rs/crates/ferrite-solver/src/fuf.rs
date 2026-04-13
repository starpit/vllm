// SPDX-License-Identifier: Apache-2.0
//! Fully Unrolled Forward (FUF) — flat DAG IR.
//!
//! The FUF is the compiler's IR. Produced from the unrolled
//! instruction stream (loops gone, indices concrete), it's a
//! straight-line graph of `Node`s connected by `Buffer`s. Each
//! buffer is either an external input (weight, activation coming
//! in from outside) or the SSA-style output of exactly one node.
//!
//! No `layer` concept. No loops. No booleans flagging special
//! kinds of nodes. Just:
//!
//!   nodes: Vec<Node>   — ops in execution order
//!   buffers: Vec<Buffer>
//!
//! Each node reads a vector of `BufferId`s and writes one or more
//! `BufferId`s. That's it. Downstream passes (solver, codegen)
//! operate on this graph generically.

use std::collections::BTreeMap;

use syn::Ident;

use crate::parse::{Arg, OpCall};
use crate::unroll::UnrollError;

/// Dense index into [`Fuf::nodes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Dense index into [`Fuf::buffers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub u32);

/// Where a buffer comes from.
#[derive(Clone, Debug)]
pub enum BufferSource {
    /// External buffer — either a non-indexed reference like
    /// `input_ids`, `lm_head`, `rotary`, or an indexed reference
    /// like `w[3]` (surviving as `VarAt` post-unroll). Externals
    /// are provided by the caller; the FUF doesn't produce them.
    External { name: String, index: Option<usize> },
    /// Defined by exactly one node (SSA). `slot` picks which of
    /// the node's outputs this buffer corresponds to (0 for
    /// single-output ops, >0 for `LetTuple` destructurings).
    Produced { node: NodeId, slot: u16 },
}

/// A buffer = one SSA value or one external input.
#[derive(Clone, Debug)]
pub struct Buffer {
    pub id: BufferId,
    pub source: BufferSource,
    /// Human-readable label for debugging: `"w[3]"`, `"hidden_states#2"`,
    /// `"rmsnorm_out#5"`, etc. Carries no semantic weight.
    pub label: String,
}

/// A FUF node — one op call with resolved inputs and outputs.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    /// Op name (the DSL identifier, e.g. `gemm`, `rmsnorm`, `silu`).
    pub op: Ident,
    /// Input buffers in argument order. `Arg::Mul(a, b)` is
    /// lowered to a synthetic `mul` node whose output becomes an
    /// input here.
    pub inputs: Vec<BufferId>,
    /// Output buffers. Length 1 for `Let`/`Assign`, N for
    /// `LetTuple`.
    pub outputs: Vec<BufferId>,
}

/// The FUF itself: the list of nodes (execution order) and all
/// buffers referenced.
#[derive(Clone, Debug)]
pub struct Fuf {
    pub nodes: Vec<Node>,
    pub buffers: Vec<Buffer>,
}

impl Fuf {
    pub fn buffer(&self, id: BufferId) -> &Buffer {
        &self.buffers[id.0 as usize]
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }
}

/// Error produced while building the FUF.
#[derive(Debug)]
pub enum FufError {
    /// Propagated from the unroll pass.
    Unroll(UnrollError),
    /// A `LetTuple` had a different arity than the op's output.
    /// For now we infer arity from the number of names, so this
    /// isn't surfaced — left here for future shape/arity checking.
    LetTupleArity {
        op: String,
        expected: usize,
        got: usize,
    },
}

impl From<UnrollError> for FufError {
    fn from(e: UnrollError) -> Self {
        FufError::Unroll(e)
    }
}

/// Build the FUF from an unrolled CFG.
pub fn build_fuf(cfg: &crate::cfg::Cfg) -> Result<Fuf, FufError> {
    let instrs = crate::unroll::unroll(cfg)?;
    build_fuf_from_instrs(&instrs)
}

/// Build the FUF from a pre-unrolled instruction stream. Exposed
/// for tests that want to feed synthetic `Instr` lists.
pub fn build_fuf_from_instrs(instrs: &[crate::cfg::Instr]) -> Result<Fuf, FufError> {
    let mut b = Builder {
        nodes: Vec::new(),
        buffers: Vec::new(),
        env: BTreeMap::new(),
        externals: BTreeMap::new(),
        version: BTreeMap::new(),
    };
    for instr in instrs {
        b.lower_instr(instr)?;
    }
    Ok(Fuf {
        nodes: b.nodes,
        buffers: b.buffers,
    })
}

// ── Builder ───────────────────────────────────────────────────────

struct Builder {
    nodes: Vec<Node>,
    buffers: Vec<Buffer>,
    /// Latest SSA definition of each named variable.
    env: BTreeMap<String, BufferId>,
    /// External buffer cache: (name, optional index) → BufferId, so
    /// the same `w[3]` reference reuses one buffer rather than
    /// creating duplicates.
    externals: BTreeMap<(String, Option<usize>), BufferId>,
    /// SSA version counter per name, used purely for labeling.
    version: BTreeMap<String, u32>,
}

impl Builder {
    fn fresh_buffer(&mut self, source: BufferSource, label: String) -> BufferId {
        let id = BufferId(self.buffers.len() as u32);
        self.buffers.push(Buffer { id, source, label });
        id
    }

    fn next_version(&mut self, name: &str) -> u32 {
        let v = self.version.entry(name.to_string()).or_insert(0);
        let cur = *v;
        *v += 1;
        cur
    }

    /// Emit an SSA def: allocate a fresh buffer, bump the version,
    /// bind it as the latest def for `name`.
    fn bind_ssa(&mut self, name: &str, node: NodeId, slot: u16) -> BufferId {
        let ver = self.next_version(name);
        let label = format!("{name}#{ver}");
        let id = self.fresh_buffer(BufferSource::Produced { node, slot }, label);
        self.env.insert(name.to_string(), id);
        id
    }

    /// Resolve an `Arg` to a `BufferId`, emitting any synthetic
    /// nodes required (for `Call` and `Mul`).
    fn resolve_arg(&mut self, arg: &Arg) -> Result<BufferId, FufError> {
        Ok(match arg {
            Arg::Var(name, None) => {
                // Either a previously-defined name (SSA) or an
                // external.
                if let Some(id) = self.env.get(name) {
                    *id
                } else {
                    self.get_or_make_external(name, None)
                }
            }
            Arg::Var(_, Some(_)) => {
                // Unroll guarantees no symbolic indexed refs
                // survive. If one does, that's a bug upstream.
                unreachable!(
                    "Arg::Var with Some(idx) reached FUF builder — unroll should have rewritten it"
                )
            }
            Arg::VarAt(name, idx) => self.get_or_make_external(name, Some(*idx)),
            Arg::Call(call) => {
                // Nested call: emit a node for it and use its
                // first output as this arg's buffer.
                let node = self.emit_call_node(call, None)?;
                self.nodes[node.0 as usize].outputs[0]
            }
            Arg::Mul(a, b) => {
                let a_id = self.resolve_arg(a)?;
                let b_id = self.resolve_arg(b)?;
                self.emit_mul_node(a_id, b_id)
            }
        })
    }

    fn get_or_make_external(&mut self, name: &str, index: Option<usize>) -> BufferId {
        let key = (name.to_string(), index);
        if let Some(id) = self.externals.get(&key) {
            return *id;
        }
        let label = match index {
            None => name.to_string(),
            Some(i) => format!("{name}[{i}]"),
        };
        let id = self.fresh_buffer(
            BufferSource::External {
                name: name.to_string(),
                index,
            },
            label,
        );
        self.externals.insert(key, id);
        id
    }

    /// Emit a node for an [`OpCall`]. If `bind_name` is `Some`,
    /// its single output is also bound as the latest SSA def of
    /// that name. If `None`, the output is available via the
    /// node's `outputs` vec (used for nested calls).
    fn emit_call_node(
        &mut self,
        call: &OpCall,
        bind_name: Option<&str>,
    ) -> Result<NodeId, FufError> {
        let inputs: Vec<BufferId> = call
            .args
            .iter()
            .map(|a| self.resolve_arg(a))
            .collect::<Result<_, _>>()?;

        let node_id = NodeId(self.nodes.len() as u32);
        // Allocate node first with empty outputs; we'll fill them
        // in after binding so the buffer's `Produced { node }`
        // points at the right id.
        self.nodes.push(Node {
            id: node_id,
            op: call.op.clone(),
            inputs,
            outputs: Vec::new(),
        });

        let out_buf = match bind_name {
            Some(name) => self.bind_ssa(name, node_id, 0),
            None => {
                // Nested-call intermediate: not bound to any DSL
                // name, so label with op + node id.
                let label = format!("{}_out#{}", call.op, node_id.0);
                self.fresh_buffer(
                    BufferSource::Produced {
                        node: node_id,
                        slot: 0,
                    },
                    label,
                )
            }
        };
        self.nodes[node_id.0 as usize].outputs.push(out_buf);
        Ok(node_id)
    }

    fn emit_mul_node(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let node_id = NodeId(self.nodes.len() as u32);
        // Synthesize an Ident for the `mul` op. Using
        // `Span::call_site()` keeps it anchored in the macro's
        // call site.
        let op = Ident::new("mul", proc_macro2::Span::call_site());
        self.nodes.push(Node {
            id: node_id,
            op,
            inputs: vec![a, b],
            outputs: Vec::new(),
        });
        let label = format!("mul_out#{}", node_id.0);
        let out = self.fresh_buffer(
            BufferSource::Produced {
                node: node_id,
                slot: 0,
            },
            label,
        );
        self.nodes[node_id.0 as usize].outputs.push(out);
        out
    }

    fn lower_instr(&mut self, instr: &crate::cfg::Instr) -> Result<(), FufError> {
        match instr {
            crate::cfg::Instr::Let { name, call } => {
                self.emit_call_node(call, Some(&name.to_string()))?;
            }
            crate::cfg::Instr::Assign { target, call } => {
                self.emit_call_node(call, Some(&target.to_string()))?;
            }
            crate::cfg::Instr::LetTuple { names, call } => {
                // Emit inputs first.
                let inputs: Vec<BufferId> = call
                    .args
                    .iter()
                    .map(|a| self.resolve_arg(a))
                    .collect::<Result<_, _>>()?;
                let node_id = NodeId(self.nodes.len() as u32);
                self.nodes.push(Node {
                    id: node_id,
                    op: call.op.clone(),
                    inputs,
                    outputs: Vec::new(),
                });
                // One output per destructured name, in order.
                let mut outs = Vec::with_capacity(names.len());
                for (slot, nm) in names.iter().enumerate() {
                    let name_s = nm.to_string();
                    let id = self.bind_ssa(&name_s, node_id, slot as u16);
                    outs.push(id);
                }
                self.nodes[node_id.0 as usize].outputs = outs;
            }
        }
        Ok(())
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

    /// Helper: fetch the list of SSA def labels produced by the FUF
    /// (in emission order). Externals are excluded.
    fn produced_labels(f: &Fuf) -> Vec<String> {
        f.buffers
            .iter()
            .filter(|b| matches!(b.source, BufferSource::Produced { .. }))
            .map(|b| b.label.clone())
            .collect()
    }

    /// Helper: for each node, return (op_name, input_labels, output_labels).
    fn node_view(f: &Fuf) -> Vec<(String, Vec<String>, Vec<String>)> {
        f.nodes
            .iter()
            .map(|n| {
                (
                    n.op.to_string(),
                    n.inputs
                        .iter()
                        .map(|b| f.buffer(*b).label.clone())
                        .collect(),
                    n.outputs
                        .iter()
                        .map(|b| f.buffer(*b).label.clone())
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn fuf_no_loops() {
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                logits = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let fuf = build_fuf(&cfg).expect("build_fuf");

        let nv = node_view(&fuf);
        assert_eq!(nv.len(), 2);

        // embed takes two externals (input_ids, embed_tokens),
        // writes hidden_states#0.
        assert_eq!(nv[0].0, "embed");
        assert_eq!(nv[0].1, vec!["input_ids", "embed_tokens"]);
        assert_eq!(nv[0].2, vec!["hidden_states#0"]);

        // gemm reads hidden_states#0 + lm_head external, writes
        // logits#0.
        assert_eq!(nv[1].0, "gemm");
        assert_eq!(nv[1].1, vec!["hidden_states#0", "lm_head"]);
        assert_eq!(nv[1].2, vec!["logits#0"]);
    }

    #[test]
    fn fuf_single_loop_threads_hidden_states() {
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
        let fuf = build_fuf(&cfg).expect("build_fuf");

        let nv = node_view(&fuf);
        // 1 embed + 3 gemm (loop) + 1 gemm (lm_head) = 5 nodes.
        assert_eq!(nv.len(), 5);

        // Iteration 0: reads hidden_states#0 (from embed), w[0],
        // writes hidden_states#1.
        assert_eq!(nv[1].0, "gemm");
        assert_eq!(nv[1].1, vec!["hidden_states#0", "w[0]"]);
        assert_eq!(nv[1].2, vec!["hidden_states#1"]);

        // Iteration 1: reads hidden_states#1, w[1], writes hidden_states#2.
        assert_eq!(nv[2].1, vec!["hidden_states#1", "w[1]"]);
        assert_eq!(nv[2].2, vec!["hidden_states#2"]);

        // Iteration 2: reads hidden_states#2, w[2], writes hidden_states#3.
        assert_eq!(nv[3].1, vec!["hidden_states#2", "w[2]"]);
        assert_eq!(nv[3].2, vec!["hidden_states#3"]);

        // Final lm_head gemm reads hidden_states#3.
        assert_eq!(nv[4].0, "gemm");
        assert_eq!(nv[4].1, vec!["hidden_states#3", "lm_head"]);
        assert_eq!(nv[4].2, vec!["logits#0"]);

        // w[0], w[1], w[2] are distinct externals; hidden_states#1..3
        // are distinct produced buffers.
        let labels = produced_labels(&fuf);
        for v in 0..4 {
            assert!(
                labels.contains(&format!("hidden_states#{v}")),
                "missing hidden_states#{v}: {labels:?}",
            );
        }

        // Each w[i] should have exactly one external buffer.
        for i in 0..3 {
            let count = fuf
                .buffers
                .iter()
                .filter(|b| {
                    matches!(&b.source, BufferSource::External { name, index: Some(idx) }
                        if name == "w" && *idx == i)
                })
                .count();
            assert_eq!(count, 1, "expected one buffer for w[{i}], got {count}");
        }
    }

    #[test]
    fn fuf_two_sequential_loops_chain() {
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
        let fuf = build_fuf(&cfg).expect("build_fuf");

        // 1 + 2 + 3 + 1 = 7 nodes.
        assert_eq!(fuf.nodes.len(), 7);

        let nv = node_view(&fuf);
        // Last loop-2 iteration writes hidden_states#6 (0 from
        // embed, 1..2 from w1, 3..5 from w2).
        assert_eq!(nv[5].2, vec!["hidden_states#5"]);
        // lm_head gemm reads it.
        assert_eq!(nv[6].1, vec!["hidden_states#5", "lm_head"]);
    }

    #[test]
    fn fuf_nested_call_flattens() {
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                logits = gemm(silu(hidden_states), lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let fuf = build_fuf(&cfg).expect("build_fuf");

        // embed, silu (nested), gemm — 3 nodes.
        let nv = node_view(&fuf);
        assert_eq!(nv.len(), 3);
        assert_eq!(nv[0].0, "embed");
        assert_eq!(nv[1].0, "silu");
        assert_eq!(nv[1].1, vec!["hidden_states#0"]);
        // silu's output is an unbound intermediate.
        assert_eq!(nv[1].2.len(), 1);
        let silu_out = &nv[1].2[0];
        assert!(
            silu_out.starts_with("silu_out#"),
            "expected silu_out#N label, got {silu_out:?}",
        );
        // gemm reads silu's output and lm_head.
        assert_eq!(nv[2].0, "gemm");
        assert_eq!(nv[2].1, vec![silu_out.clone(), "lm_head".to_string()]);
    }

    #[test]
    fn fuf_let_tuple_destructures() {
        // Use `rope` which is the one LetTuple op in the DSL —
        // but we'll synthesize a minimal DSL that uses it.
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                let (q, k) = rope(hidden_states, rotary);
                logits = gemm(q, k);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let fuf = build_fuf(&cfg).expect("build_fuf");

        let nv = node_view(&fuf);
        // embed, rope (2 outputs), gemm.
        assert_eq!(nv.len(), 3);
        assert_eq!(nv[1].0, "rope");
        assert_eq!(nv[1].2, vec!["q#0", "k#0"]);
        assert_eq!(nv[2].1, vec!["q#0", "k#0"]);
    }

    #[test]
    fn fuf_external_dedup() {
        // Two reads of the same external `lm_head` should share
        // one buffer.
        let def = parse_dsl(
            r#"
            kernel test<HD=128, ID=128, HDM=32, NAH=4, NKH=1, VS=1024> {
                hidden_states = embed(input_ids, embed_tokens);
                a = gemm(hidden_states, lm_head);
                b = gemm(hidden_states, lm_head);
            }
            "#,
        );
        let cfg = build_cfg(&def);
        let fuf = build_fuf(&cfg).expect("build_fuf");

        let lm_head_bufs: Vec<_> = fuf
            .buffers
            .iter()
            .filter(|b| {
                matches!(&b.source,
                BufferSource::External { name, index: None } if name == "lm_head")
            })
            .collect();
        assert_eq!(
            lm_head_bufs.len(),
            1,
            "lm_head should appear as a single external buffer",
        );
    }
}
