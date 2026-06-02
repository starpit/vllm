// SPDX-License-Identifier: Apache-2.0
//! Phase 12: orchestrator-level smem-routing analysis for the
//! megakernel.
//!
//! In a persistent-CTA megakernel, the producer and consumer of an
//! intermediate activation share the same SM and the same shared
//! memory. The current per-op lowering still routes those activations
//! through gmem (the producer's storer drains its smem page to a
//! per-op `op{i}_out` kernel-arg buffer; the consumer's loader brings
//! the same bytes back into a different smem page). For activations
//! consumed only inside this same megakernel, that round-trip is pure
//! waste — the bytes leave the SM and re-enter through HBM.
//!
//! This module computes per-op output routing classifications from
//! the [`crate::lower::LoweringInput`] DAG. Each op's output is
//! either:
//!
//!   * **Internal** — consumed by one or more downstream ops in this
//!     same `LoweringInput`, never by the host or another kernel.
//!     The per-op `op{i}_out` gmem buffer is unnecessary: the
//!     producer's smem page can be carried forward to the consumer
//!     directly, with a Phase 11 cross-IType mbarrier handshake at
//!     the boundary.
//!
//!   * **External** — the canonical's `result` op (must be readable
//!     post-kernel for the dispatch layer's lookup), or any output
//!     the host or another kernel reads via a kernel-arg pointer.
//!     Existing gmem path: storer drains the smem page to gmem.
//!
//! The analysis is one-shot: a backward pass from `result` plus a
//! consumer-count walk over `inputs`. No fixpoint, no iteration.
//!
//! The dispatch / lowering side then consumes this classification to
//! choose between the gmem-roundtrip path and the carry-forward path
//! (Phase 12 follow-up — this file ships only the analysis, no
//! emit-side wiring yet, so default behaviour is byte-identical to
//! Phase 11).

use crate::lower::{InputRef, LoweringInput};

/// Per-op output routing decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputRouting {
    /// Output is consumed only by other ops in this `LoweringInput`,
    /// never read post-kernel. Storer can skip the gmem TMA store;
    /// the producer's smem page is carried forward to the consuming
    /// op's input slot via cross-IType mbarrier handoff.
    ///
    /// `consumer_op_indices` lists the downstream ops that read this
    /// output. Length 1 = single-consumer (simplest carry-forward
    /// case). Length > 1 = multi-consumer (consumers run sequentially
    /// in topo order; the page must remain alive across all of
    /// their rounds — implementation can either carry-forward to the
    /// last consumer or fall back to gmem routing for safety).
    Internal {
        consumer_op_indices: Vec<usize>,
    },

    /// Output is the canonical's `result`, or otherwise observable
    /// from outside this kernel. Existing gmem-roundtrip path stays.
    External,
}

impl OutputRouting {
    /// Single-consumer internal output — the simplest carry-forward
    /// case. Returns `Some(consumer_idx)` for that one consumer.
    pub fn single_consumer(&self) -> Option<usize> {
        match self {
            OutputRouting::Internal { consumer_op_indices } if consumer_op_indices.len() == 1 => {
                Some(consumer_op_indices[0])
            }
            _ => None,
        }
    }

    /// `true` iff this is an internal output (any consumer count).
    pub fn is_internal(&self) -> bool {
        matches!(self, OutputRouting::Internal { .. })
    }
}

/// Classify every op's output in `input` as internal or external.
///
/// Algorithm:
///   1. Build a per-op consumer list by scanning every later op's
///      `inputs` for `InputRef::Op(j)`.
///   2. The op at `input.result` is always External (the canonical's
///      result must be readable post-kernel).
///   3. Every other op with at least one consumer is Internal.
///   4. Ops with zero consumers AND not `result` — dead code, but
///      classify as External (defensive: keep gmem routing so the
///      orchestrator's behaviour stays unchanged from Phase 11).
///
/// Returns a vector of length `input.ops.len()`, parallel to
/// `input.ops`.
pub fn classify_outputs(input: &LoweringInput) -> Vec<OutputRouting> {
    let n = input.ops.len();
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (consumer_idx, desc) in input.ops.iter().enumerate() {
        for r in &desc.inputs {
            if let InputRef::Op(producer_idx) = *r {
                debug_assert!(
                    producer_idx < consumer_idx,
                    "topo violated: op {consumer_idx} consumes op {producer_idx}",
                );
                consumers[producer_idx].push(consumer_idx);
            }
        }
    }

    let mut routing = Vec::with_capacity(n);
    for (idx, c_list) in consumers.into_iter().enumerate() {
        if idx == input.result || c_list.is_empty() {
            routing.push(OutputRouting::External);
        } else {
            routing.push(OutputRouting::Internal {
                consumer_op_indices: c_list,
            });
        }
    }
    routing
}

/// Per-op input routing decision (the producer-side counterpart to
/// `OutputRouting`). For each `InputRef::Op(j)` slot, the consumer
/// can either:
///   * **CarryForward** — the producer's smem page is reused as this
///     op's input page; consumer's loader skips the TMA load and
///     waits on a cross-IType mbarrier instead.
///   * **GmemLoad** — existing path; loader TMA-loads from the
///     producer's `op{j}_out` gmem buffer (or for `InputRef::Ext`,
///     from the external source buffer).
///
/// Carry-forward is only chosen when the producer's output is
/// `Internal` AND this op is the producer's *last* consumer (so the
/// page is released after this op's round). Earlier consumers of a
/// multi-consumer internal output use GmemLoad to avoid the
/// page-lifetime tracking complexity (a simple, conservative rule;
/// follow-up phases can extend to multi-consumer carry-forward by
/// keeping the page alive across N rounds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRouting {
    CarryForward { producer_op_idx: usize },
    GmemLoad,
}

/// Classify every op's input slots. Returns
/// `Vec<Vec<InputRouting>>` parallel to `input.ops` and to
/// `input.ops[i].inputs`.
pub fn classify_inputs(
    input: &LoweringInput,
    outputs: &[OutputRouting],
) -> Vec<Vec<InputRouting>> {
    let n = input.ops.len();
    let mut last_consumer: Vec<Option<usize>> = vec![None; n];
    for routing in outputs.iter() {
        if let OutputRouting::Internal { consumer_op_indices } = routing {
            // The last consumer in topo order is the safe carry-
            // forward target.
            if let Some(last) = consumer_op_indices.iter().max().copied() {
                let producer_for_last = consumer_op_indices.iter().position(|&c| c == last);
                debug_assert!(producer_for_last.is_some());
            }
        }
    }
    // Re-derive: for each producer with internal output, the last
    // consumer (max consumer_idx) is the carry-forward target.
    for (producer_idx, routing) in outputs.iter().enumerate() {
        if let OutputRouting::Internal { consumer_op_indices } = routing {
            if let Some(&last) = consumer_op_indices.iter().max() {
                last_consumer[producer_idx] = Some(last);
            }
        }
    }

    let mut all_inputs = Vec::with_capacity(n);
    for (consumer_idx, desc) in input.ops.iter().enumerate() {
        let mut per_input = Vec::with_capacity(desc.inputs.len());
        for r in &desc.inputs {
            match r {
                InputRef::Op(producer_idx) => {
                    let producer_internal = matches!(
                        outputs[*producer_idx],
                        OutputRouting::Internal { .. }
                    );
                    let is_last_consumer = last_consumer[*producer_idx] == Some(consumer_idx);
                    if producer_internal && is_last_consumer {
                        per_input.push(InputRouting::CarryForward {
                            producer_op_idx: *producer_idx,
                        });
                    } else {
                        per_input.push(InputRouting::GmemLoad);
                    }
                }
                InputRef::Ext(_) => per_input.push(InputRouting::GmemLoad),
            }
        }
        all_inputs.push(per_input);
    }
    all_inputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{LoweredOp, OpDesc};
    use crate::subtile::SourceShape;

    fn input_with_ops(sources: Vec<SourceShape>, ops: Vec<OpDesc>, result: usize) -> LoweringInput {
        LoweringInput { sources, ops, result }
    }

    /// op0 is `result` → External. Single op, zero consumers.
    #[test]
    fn single_op_result_is_external() {
        let input = input_with_ops(
            vec![SourceShape { rows: 1, cols: 8 }],
            vec![OpDesc {
                op: LoweredOp::RmsNorm { eps: 1e-5 },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(0)],
            }],
            0,
        );
        let r = classify_outputs(&input);
        assert_eq!(r.len(), 1);
        assert!(matches!(r[0], OutputRouting::External));
    }

    /// op0 → op1, op1 is `result`. op0 is internal (1 consumer); op1
    /// is external (the result).
    #[test]
    fn linear_chain_internal_then_external() {
        let input = input_with_ops(
            vec![
                SourceShape { rows: 1, cols: 8 }, // 0  x
                SourceShape { rows: 1, cols: 8 }, // 1  w
            ],
            vec![
                OpDesc {
                    op: LoweredOp::RmsNorm { eps: 1e-5 },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(0)],
                },
            ],
            1,
        );
        let r = classify_outputs(&input);
        assert_eq!(r.len(), 2);
        match &r[0] {
            OutputRouting::Internal { consumer_op_indices } => {
                assert_eq!(consumer_op_indices, &vec![1]);
            }
            _ => panic!("op0 should be Internal"),
        }
        assert!(matches!(r[1], OutputRouting::External));
        assert_eq!(r[0].single_consumer(), Some(1));
    }

    /// op0 → op1 + op2, op2 is `result`. op0 is multi-consumer
    /// internal; op1 is internal-but-zero-consumers (dead code, kept
    /// External defensively); op2 is external.
    ///
    /// More importantly: the input routing for op1's input from op0
    /// should be GmemLoad (op1 is NOT the last consumer of op0); for
    /// op2's input from op0 it should be CarryForward (op2 IS the
    /// last consumer).
    #[test]
    fn fanout_routes_last_consumer_as_carry_forward() {
        let input = input_with_ops(
            vec![
                SourceShape { rows: 1, cols: 8 }, // 0
                SourceShape { rows: 1, cols: 8 }, // 1
            ],
            vec![
                OpDesc {
                    op: LoweredOp::RmsNorm { eps: 1e-5 },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(0)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Op(1)],
                },
            ],
            2,
        );
        let outputs = classify_outputs(&input);
        match &outputs[0] {
            OutputRouting::Internal { consumer_op_indices } => {
                assert_eq!(consumer_op_indices, &vec![1, 2]);
            }
            _ => panic!("op0 should be Internal"),
        }
        let inputs = classify_inputs(&input, &outputs);
        // op1 reads op0 — NOT the last consumer (op2 is later) → GmemLoad.
        assert_eq!(inputs[1][0], InputRouting::GmemLoad);
        // op2 reads op0 — the LAST consumer → CarryForward.
        assert_eq!(
            inputs[2][0],
            InputRouting::CarryForward { producer_op_idx: 0 }
        );
        // op2 reads op1 — op1 is Internal (op2 is its only/last consumer)
        // → CarryForward.
        assert_eq!(
            inputs[2][1],
            InputRouting::CarryForward { producer_op_idx: 1 }
        );
    }

    /// External-source inputs (`InputRef::Ext`) always route via
    /// GmemLoad — they're external to the kernel.
    #[test]
    fn ext_inputs_always_gmem_load() {
        let input = input_with_ops(
            vec![
                SourceShape { rows: 1, cols: 8 },
                SourceShape { rows: 1, cols: 8 },
            ],
            vec![OpDesc {
                op: LoweredOp::Add,
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            0,
        );
        let outputs = classify_outputs(&input);
        let inputs = classify_inputs(&input, &outputs);
        assert_eq!(inputs[0][0], InputRouting::GmemLoad);
        assert_eq!(inputs[0][1], InputRouting::GmemLoad);
    }

    /// Llama-1B-shape sequence: rmsnorm → q/k/v gemm → rope → attn →
    /// out_proj → add → ... — verify the multi-consumer rmsnorm is
    /// classified internal, q-gemm output is single-consumer-internal
    /// (carry-forward to rope), out_proj is single-consumer-internal
    /// (carry-forward to add), and the final result op is external.
    #[test]
    fn llama_layer_routing_classification() {
        let input = input_with_ops(
            vec![
                SourceShape { rows: 1, cols: 2048 }, // 0  x
                SourceShape { rows: 1, cols: 2048 }, // 1  rms_w
                SourceShape { rows: 2048, cols: 2048 }, // 2  q_w (one weight, simplified)
            ],
            vec![
                // 0: rmsnorm(x, rms_w) — multi-consumer
                OpDesc {
                    op: LoweredOp::RmsNorm { eps: 1e-5 },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                // 1: q_gemm(rmsnorm, q_w) — first consumer of rmsnorm
                OpDesc {
                    op: LoweredOp::Gemm { n: 2048, k: 2048 },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
                // 2: residual_add(rmsnorm, x) — last consumer of rmsnorm
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(0)],
                },
                // 3: residual_add(q_gemm, rmsnorm-thing) — result
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(1), InputRef::Op(2)],
                },
            ],
            3,
        );
        let outputs = classify_outputs(&input);
        // op0 (rmsnorm): multi-consumer internal (1, 2)
        match &outputs[0] {
            OutputRouting::Internal { consumer_op_indices } => {
                assert_eq!(consumer_op_indices, &vec![1, 2]);
            }
            _ => panic!("rmsnorm should be Internal"),
        }
        // op1 (q_gemm): single-consumer internal (3)
        assert_eq!(outputs[1].single_consumer(), Some(3));
        // op2 (add): single-consumer internal (3)
        assert_eq!(outputs[2].single_consumer(), Some(3));
        // op3 (final add): external (it IS result)
        assert!(matches!(outputs[3], OutputRouting::External));

        let inputs = classify_inputs(&input, &outputs);
        // op1 reads op0 — NOT last consumer (op2 is later) → GmemLoad
        assert_eq!(inputs[1][0], InputRouting::GmemLoad);
        // op2 reads op0 — IS last consumer → CarryForward
        assert_eq!(
            inputs[2][0],
            InputRouting::CarryForward { producer_op_idx: 0 }
        );
        // op3 reads op1 — last (only) consumer of op1 → CarryForward
        assert_eq!(
            inputs[3][0],
            InputRouting::CarryForward { producer_op_idx: 1 }
        );
        // op3 reads op2 — last (only) consumer of op2 → CarryForward
        assert_eq!(
            inputs[3][1],
            InputRouting::CarryForward { producer_op_idx: 2 }
        );
    }
}
