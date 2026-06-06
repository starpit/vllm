// SPDX-License-Identifier: Apache-2.0
//! Tensor-parallel **partition** of the decode SubtileIR for the wavefront
//! megakernel: turn the per-op dataflow into `P` local worker-chains joined
//! only at the genuine reductions, so the megakernel stops paying the
//! cross-worker dependency-WAIT (the measured ~2.3 ms gap vs the per-op path,
//! where 9 workers idle-spin waiting for a whole-op on the 1).
//!
//! The shape is Megatron tensor-parallelism inside one persistent kernel:
//!   - **Attention block** partitions by q-head: rmsnorm (whole) → q/k/v
//!     (N-block) → rope (head) → attn (head) → o_proj (**split-K** over heads
//!     → one partial per head-block + a SumReduce all-reduce).
//!   - **MLP block** partitions by intermediate column: rmsnorm (whole) →
//!     gate/up (N-block) → silu·mul (column) → down (**split-K** → partials +
//!     SumReduce).
//!
//! Worker `w` owns column-slice `w` of every op (slice-index placement,
//! `owner = (col start / head_dim) mod P`), so a producer→consumer chain that
//! preserves columns (q→rope→attn→o-partial; gate/up→silu→down-partial) stays
//! on ONE worker — local, no cross-worker wait. The only cross-worker traffic
//! is the two split-K all-reduces per layer plus the cheap shared K/V (gqa
//! q-heads share one kv-head). rmsnorm/add are whole here and are
//! **replicated** per worker by [`replicate_whole_ops`] so every worker holds
//! the backbone locally (no broadcast wait); replication is host-neutral (same
//! arithmetic) so it layers on after the split-K dataflow is proven.
//!
//! `lower_partitioned` re-lowers from the [`LoweringInput`] (not a post-pass on
//! the flat graph) because the split-K transform needs the op boundaries the
//! flat SubtileIR has erased. It reuses `subtile_ir`'s tiling helpers and per-op
//! arithmetic verbatim, so [`crate::subtile_ir::eval_dag`] validates it the same
//! way: bit-exact for head/column tiling; within-tol where split-K reassociates
//! the reduction (PLAN Tier A′).

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::lower::{InputRef, LoweredOp, LoweringInput};
use crate::subtile_ir::{
    EwKind, KvCacheLayout, KvCacheProducer, NeoX, Range, Region, SoftmaxStateId, SubOp, SubtileId,
    SubtileIR, SubtileNode, TensorId, TensorRegion, TensorShape, head_blocks, n_blocks,
    op_out_cols,
};

/// Lower a decode `LoweringInput` to its tensor-parallel partition: the
/// finely-tiled SubtileIR plus a `owner[node]` worker assignment. Tiling is
/// at `head_dim` granularity so head-structured ops and column-tiled ops share
/// the slice-index. `p` = worker count. Reductions (a GEMM whose activation is
/// a partitioned op output — o_proj, down) become split-K: one partial per
/// activation block (co-located with it) plus a `SumReduce` all-reduce.
/// Everything else is N-block / head-tile / column-tile; rmsnorm/add stay whole
/// (replicated by [`replicate_whole_ops`]). Returns `(graph, owner)` for
/// [`crate::region_schedule::schedule_from_assignment`].
pub fn lower_partitioned(
    input: &LoweringInput,
    head_dim: u32,
    mlp_unit: u32,
    p: u32,
) -> (SubtileIR<NeoX>, Vec<u32>) {
    /// One op's output: partitioned (a single tensor written by per-block
    /// nodes) or replicated (P per-worker whole copies — a consumer on worker
    /// `w` reads copy `w`, so the backbone is local to every worker).
    enum OpOut {
        Part(TensorId),
        Repl(Vec<TensorId>),
    }
    // Two tile units: `chain_unit = head_dim` for the attention chain (q/k/v/
    // rope/attn/o-proj — must be head-aligned), and `mlp_unit` for the MLP
    // chain (gate/up/silu_mul/down — its own column-preserving chain). The
    // partition is COHERENT iff the chain-locality property holds within each
    // chain; using a coarser `mlp_unit` keeps the MLP chain local while letting
    // the BW saturate (cost-sweep finding: unit=256 hits the sweet spot on
    // Llama-1B, measured COMPUTE_ONLY ~7.3ms vs head_dim=64 ~7.9ms). Default
    // `mlp_unit = head_dim` preserves the old behaviour (no-op for tests).
    use std::num::NonZeroU32;
    // chain_unit / mlp_unit are tile widths (block sizes for n_blocks /
    // head_blocks). `block == 0` would loop forever in n_blocks; carrying
    // these as NonZeroU32 makes the termination invariant compile-time
    // (per feedback_compile_time_or_garbage).
    let chain_unit = NonZeroU32::new(head_dim.max(1))
        .expect(".max(1) above guarantees nonzero");
    let mlp_unit = NonZeroU32::new(mlp_unit.max(chain_unit.get()))
        .expect(".max(chain_unit) above guarantees nonzero");
    let p = p.max(1);
    // Pick the tile unit for an op whose output is `out_cols` wide: use
    // `mlp_unit` only when it strictly coarsens AND gives at least P blocks
    // (so workers stay utilised — otherwise the wider tile leaves cores idle).
    let pick_unit = |out_cols: u32| -> NonZeroU32 {
        if mlp_unit > chain_unit
            && out_cols.is_multiple_of(mlp_unit.get())
            && out_cols / mlp_unit.get() >= p
        {
            mlp_unit
        } else {
            chain_unit
        }
    };
    let num_sources = input.sources.len() as u32;
    let mut tensors: Vec<TensorShape> = input
        .sources
        .iter()
        .map(|s| TensorShape {
            rows: s.rows,
            cols: s.cols,
        })
        .collect();
    let mut nodes: Vec<SubtileNode<NeoX>> = Vec::new();
    let mut k_cache_producer_node: HashMap<TensorId, u32> = HashMap::new();
    let mut next_softmax_state: u32 = 0;
    let mut owner: Vec<u32> = Vec::new();
    let mut op_out: Vec<OpOut> = Vec::with_capacity(input.ops.len());
    let mut op_cols: Vec<u32> = Vec::with_capacity(input.ops.len());
    let mut op_blocks: Vec<Vec<Range>> = Vec::with_capacity(input.ops.len());
    let mut op_partitioned: Vec<bool> = Vec::with_capacity(input.ops.len());

    // Worker owning a block: (column start / head_dim) mod P.
    // `u` is a tile width — `NonZeroU32` so division can never be by
    // zero (the substrate-level termination witness from `n_blocks`
    // propagates here).
    let owner_of = |cols_start: u32, u: NonZeroU32| (cols_start / u.get()) % p;

    // Resolve an InputRef as read by a consumer on worker `w`: a replicated
    // producer gives worker `w`'s copy; a partitioned producer / a source gives
    // the single tensor.
    let resolve = |r: InputRef,
                   w: u32,
                   op_out: &[OpOut],
                   op_cols: &[u32],
                   tensors: &[TensorShape]|
     -> (TensorId, u32, u32) {
        match r {
            InputRef::Op(j) => {
                let t = match &op_out[j] {
                    OpOut::Part(t) => *t,
                    OpOut::Repl(reps) => reps[w as usize],
                };
                (t, tensors[t.0 as usize].rows, op_cols[j])
            }
            InputRef::Ext(e) => {
                let s = tensors[e];
                (TensorId(e as u32), s.rows, s.cols)
            }
        }
    };

    for desc in &input.ops {
        let m = desc.m;
        // Shape only (all replicas share shape) — resolve with worker 0.
        let (_in0_t0, _in0_rows, in0_cols) =
            resolve(desc.inputs[0], 0, &op_out, &op_cols, &tensors);
        let out_cols = op_out_cols(desc.op, in0_cols);

        let mut blocks_emitted: Vec<Range> = Vec::new();
        let mut partitioned = true;
        let this_out: OpOut;

        match desc.op {
            LoweredOp::Gemm { n } => {
                let k = in0_cols;
                let (w_t, _, _) = resolve(desc.inputs[1], 0, &op_out, &op_cols, &tensors);
                // A GEMM whose activation is a partitioned op output reduces
                // over the partition axis ⇒ split-K (o_proj over heads, down
                // over intermediate). Else N-block the output (q/k/v/gate/up
                // read the whole replicated rmsnorm output, per worker).
                let split_k = matches!(desc.inputs[0], InputRef::Op(j) if op_partitioned[j]);
                if split_k {
                    let act_op = match desc.inputs[0] {
                        InputRef::Op(j) => j,
                        _ => unreachable!("split-K activation is an op output"),
                    };
                    let (act_t, _, _) = resolve(desc.inputs[0], 0, &op_out, &op_cols, &tensors);
                    let kchunks = op_blocks[act_op].clone();
                    let whole = || TensorShape { rows: m, cols: n }.whole();
                    let new_tensor = |tensors: &mut Vec<TensorShape>| {
                        let t = TensorId(tensors.len() as u32);
                        tensors.push(TensorShape { rows: m, cols: n });
                        t
                    };
                    // One partial per activation K-chunk, grouped by its owner.
                    let mut by_worker: Vec<Vec<TensorId>> = vec![Vec::new(); p as usize];
                    for kb in &kchunks {
                        // partial = act[:, kb] @ W[0..n, kb] → whole [m, n], on
                        // the worker that owns the activation block kb. The
                        // activation was tiled at `kb.len` (uniform tiling), so
                        // that is the unit chain-locality goes by here.
                        // kb came from n_blocks/head_blocks with a NonZeroU32
                        // block width; len is at least 1 (the only zero case
                        // is the degenerate total==0 placeholder, which can't
                        // appear here — split-K runs only when the activation
                        // is actually partitioned, so kchunks is nonempty
                        // with positive widths).
                        let kb_len = NonZeroU32::new(kb.len)
                            .expect("split-K kchunks come from n_blocks with NonZeroU32 width");
                        let w = owner_of(kb.start, kb_len);
                        let pt = new_tensor(&mut tensors);
                        let id = SubtileId(nodes.len() as u32);
                        nodes.push(SubtileNode {
                            id,
                            op: SubOp::MatmulTile,
                            inputs: vec![
                                TensorRegion {
                                    tensor: act_t,
                                    region: Region {
                                        rows: Range::new(0, m),
                                        cols: *kb,
                                    },
                                },
                                TensorRegion {
                                    tensor: w_t,
                                    region: Region {
                                        rows: Range::new(0, n),
                                        cols: *kb,
                                    },
                                },
                            ],
                            output: TensorRegion {
                                tensor: pt,
                                region: whole(),
                            },
                        });
                        owner.push(w);
                        by_worker[w as usize].push(pt);
                    }
                    let sum_node =
                        |nodes: &mut Vec<SubtileNode>, ins: &[TensorId], out: TensorId| {
                            let id = SubtileId(nodes.len() as u32);
                            nodes.push(SubtileNode {
                                id,
                                op: SubOp::SumReduce,
                                inputs: ins
                                    .iter()
                                    .map(|&t| TensorRegion {
                                        tensor: t,
                                        region: whole(),
                                    })
                                    .collect(),
                                output: TensorRegion {
                                    tensor: out,
                                    region: whole(),
                                },
                            });
                        };
                    // Per-worker LOCAL partial: sum the worker's own blocks ON
                    // that worker — a same-worker SumReduce, NO cross-worker
                    // handoff. (One block ⇒ it IS the local partial; zero ⇒ the
                    // worker contributes nothing.) This coalesces the
                    // O(blocks·P) all-reduce traffic down to O(P²): the global
                    // reduce then sums only the P local partials, not every block.
                    let mut local_partials: Vec<TensorId> = Vec::new();
                    for (w, blocks) in by_worker.iter().enumerate() {
                        match blocks.len() {
                            0 => {}
                            1 => local_partials.push(blocks[0]),
                            _ => {
                                let lt = new_tensor(&mut tensors);
                                sum_node(&mut nodes, blocks, lt);
                                owner.push(w as u32);
                                local_partials.push(lt);
                            }
                        }
                    }
                    // Global all-reduce: CENTRALIZED — one SumReduce on worker 0
                    // sums all P local partials; the result is the single
                    // replicated-OpOut tensor that every downstream replicated
                    // consumer (rmsnorm / add) reads. plan_handoffs then emits
                    // (P−1) ACQUIREs on worker 0 reading the cross-worker
                    // partials, plus (P−1) ACQUIREs on workers 1..P−1 reading
                    // the reduce output. Total handoffs per all-reduce: 2·(P−1)
                    // vs P² in the prior P-replicated emit (e.g. 18 vs 100 at
                    // P=10) — measured ~3.6× drop in TOTAL tape handoffs
                    // (3200 → 896 on Llama-1B), so atomic-spin contention on
                    // the publish/acquire bus drops correspondingly.
                    let only_ot = new_tensor(&mut tensors);
                    sum_node(&mut nodes, &local_partials, only_ot);
                    owner.push(0u32);
                    let reps = vec![only_ot; p as usize];
                    blocks_emitted.push(Range::new(0, n));
                    partitioned = false; // reduced → whole, replicated output
                    this_out = OpOut::Repl(reps);
                } else {
                    let out_t = TensorId(tensors.len() as u32);
                    tensors.push(TensorShape {
                        rows: m,
                        cols: out_cols,
                    });
                    let op_unit = pick_unit(n);
                    for blk in n_blocks(n, op_unit) {
                        let w = owner_of(blk.start, op_unit);
                        let (act_t, _, _) = resolve(desc.inputs[0], w, &op_out, &op_cols, &tensors);
                        let id = SubtileId(nodes.len() as u32);
                        nodes.push(SubtileNode {
                            id,
                            op: SubOp::MatmulTile,
                            inputs: vec![
                                TensorRegion {
                                    tensor: act_t,
                                    region: Region {
                                        rows: Range::new(0, m),
                                        cols: Range::new(0, k),
                                    },
                                },
                                TensorRegion {
                                    tensor: w_t,
                                    region: Region {
                                        rows: blk,
                                        cols: Range::new(0, k),
                                    },
                                },
                            ],
                            output: TensorRegion {
                                tensor: out_t,
                                region: Region {
                                    rows: Range::new(0, m),
                                    cols: blk,
                                },
                            },
                        });
                        owner.push(w);
                        blocks_emitted.push(blk);
                    }
                    this_out = OpOut::Part(out_t);
                }
            }
            LoweredOp::AttnDecode {
                num_q_heads,
                num_kv_heads,
                head_dim: hd,
                scale,
            } => {
                let gqa = num_q_heads / num_kv_heads.max(1);
                // Resolve the prefix-K input (desc.inputs[1]) to its
                // TensorId; the witness binds the cache identity, the
                // producer (if any rope_append wrote it earlier in the
                // same forward), and the softmax state id.
                let (prefix_k_t, _, _) = resolve(desc.inputs[1], 0, &op_out, &op_cols, &tensors);
                let layout = KvCacheLayout::for_cache_tensor(prefix_k_t, num_kv_heads, hd);
                let producer = match k_cache_producer_node.get(&prefix_k_t) {
                    Some(&node_idx) => KvCacheProducer::from_rope_append(node_idx),
                    None => KvCacheProducer::pre_populated_ext(),
                };
                let softmax_state = SoftmaxStateId::new(next_softmax_state);
                next_softmax_state += 1;
                let subop: SubOp<NeoX> = SubOp::AttnDecode {
                    num_q_heads,
                    num_kv_heads,
                    head_dim: hd,
                    scale,
                    layout,
                    producer,
                    softmax_state,
                };
                let out_t = TensorId(tensors.len() as u32);
                tensors.push(TensorShape {
                    rows: m,
                    cols: out_cols,
                });
                let hd_nz = NonZeroU32::new(hd).unwrap_or(NonZeroU32::MIN);
                for blk in head_blocks(out_cols, chain_unit, hd_nz) {
                    let w = owner_of(blk.start, chain_unit);
                    let qh_start = blk.start / hd;
                    let qh_end = blk.end().div_ceil(hd);
                    let kvh_start = qh_start / gqa;
                    let kvh_end = (qh_end - 1) / gqa + 1;
                    let kv_cols = Range::new(kvh_start * hd, (kvh_end - kvh_start) * hd);
                    let inputs: Vec<TensorRegion> = desc
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(ii, r)| {
                            let (t, _, _) = resolve(*r, w, &op_out, &op_cols, &tensors);
                            let region = if ii == 0 {
                                Region {
                                    rows: Range::new(0, m),
                                    cols: blk,
                                }
                            } else {
                                Region {
                                    rows: Range::new(0, tensors[t.0 as usize].rows),
                                    cols: kv_cols,
                                }
                            };
                            TensorRegion { tensor: t, region }
                        })
                        .collect();
                    let id = SubtileId(nodes.len() as u32);
                    nodes.push(SubtileNode {
                        id,
                        op: subop,
                        inputs,
                        output: TensorRegion {
                            tensor: out_t,
                            region: Region {
                                rows: Range::new(0, m),
                                cols: blk,
                            },
                        },
                    });
                    owner.push(w);
                    blocks_emitted.push(blk);
                }
                this_out = OpOut::Part(out_t);
            }
            other => {
                let subop: SubOp<NeoX> = match other {
                    LoweredOp::RmsNorm { eps } => SubOp::RmsNorm { eps },
                    LoweredOp::Silu => SubOp::Elementwise(EwKind::Silu),
                    LoweredOp::Mul => SubOp::Elementwise(EwKind::Mul),
                    LoweredOp::SiluMul => SubOp::SiluMul,
                    LoweredOp::Add => SubOp::Elementwise(EwKind::Add),
                    LoweredOp::RopeRotate { head_dim } => SubOp::RopeRotate {
                        head_dim,
                        _form: PhantomData,
                    },
                    LoweredOp::RopeAppend { head_dim, layer } => {
                        // See lower_region — fall back to the K input's
                        // tensor when the LoweringInput omits the
                        // K_cache slot (legacy 4-input shape).
                        let k_cache_t = if desc.inputs.len() > 4 {
                            resolve(desc.inputs[4], 0, &op_out, &op_cols, &tensors).0
                        } else {
                            resolve(desc.inputs[0], 0, &op_out, &op_cols, &tensors).0
                        };
                        let num_kv_heads = (in0_cols / head_dim.max(1)).max(1);
                        let layout =
                            KvCacheLayout::for_cache_tensor(k_cache_t, num_kv_heads, head_dim);
                        k_cache_producer_node.insert(k_cache_t, nodes.len() as u32);
                        SubOp::RopeAppend {
                            head_dim,
                            layer,
                            layout,
                            _form: PhantomData,
                        }
                    }
                    LoweredOp::Gemm { .. } | LoweredOp::AttnDecode { .. } => {
                        unreachable!("handled above")
                    }
                };
                enum Cat {
                    Whole,
                    Elem,
                    Rope,
                }
                let cat = match other {
                    LoweredOp::Silu | LoweredOp::Mul | LoweredOp::SiluMul => Cat::Elem,
                    LoweredOp::RopeRotate { .. } | LoweredOp::RopeAppend { .. } => Cat::Rope,
                    // rmsnorm AND residual-add stay whole — the replicated
                    // backbone every worker recomputes (no broadcast wait).
                    _ => Cat::Whole,
                };
                let hd = match other {
                    LoweredOp::RopeRotate { head_dim } | LoweredOp::RopeAppend { head_dim, .. } => {
                        head_dim
                    }
                    _ => chain_unit.get(), // placeholder; only Cat::Rope reads it
                };
                match cat {
                    Cat::Whole => {
                        // Replicate: P per-worker copies, each reading worker w's
                        // copy of every (replicated) input — whole.
                        partitioned = false;
                        let mut reps = Vec::with_capacity(p as usize);
                        for w in 0..p {
                            let ot = TensorId(tensors.len() as u32);
                            tensors.push(TensorShape {
                                rows: m,
                                cols: out_cols,
                            });
                            let inputs: Vec<TensorRegion> = desc
                                .inputs
                                .iter()
                                .map(|r| {
                                    let (t, _, _) = resolve(*r, w, &op_out, &op_cols, &tensors);
                                    TensorRegion {
                                        tensor: t,
                                        region: tensors[t.0 as usize].whole(),
                                    }
                                })
                                .collect();
                            let id = SubtileId(nodes.len() as u32);
                            nodes.push(SubtileNode {
                                id,
                                op: subop,
                                inputs,
                                output: TensorRegion {
                                    tensor: ot,
                                    region: TensorShape {
                                        rows: m,
                                        cols: out_cols,
                                    }
                                    .whole(),
                                },
                            });
                            owner.push(w);
                            reps.push(ot);
                        }
                        blocks_emitted.push(Range::new(0, out_cols));
                        this_out = OpOut::Repl(reps);
                    }
                    Cat::Elem | Cat::Rope => {
                        let out_t = TensorId(tensors.len() as u32);
                        tensors.push(TensorShape {
                            rows: m,
                            cols: out_cols,
                        });
                        let op_unit = match cat {
                            Cat::Elem => pick_unit(out_cols),
                            Cat::Rope => chain_unit, // rope/attn must be head-aligned
                            Cat::Whole => unreachable!(),
                        };
                        let blks = match cat {
                            Cat::Elem => n_blocks(out_cols, op_unit),
                            Cat::Rope => {
                                let hd_nz = NonZeroU32::new(hd).unwrap_or(NonZeroU32::MIN);
                                head_blocks(out_cols, op_unit, hd_nz)
                            }
                            Cat::Whole => unreachable!(),
                        };
                        for blk in blks {
                            let w = owner_of(blk.start, op_unit);
                            let inputs: Vec<TensorRegion> = desc
                                .inputs
                                .iter()
                                .enumerate()
                                .map(|(ii, r)| {
                                    let (t, _, _) = resolve(*r, w, &op_out, &op_cols, &tensors);
                                    let slice = TensorRegion {
                                        tensor: t,
                                        region: Region {
                                            rows: Range::new(0, m),
                                            cols: blk,
                                        },
                                    };
                                    let whole = TensorRegion {
                                        tensor: t,
                                        region: tensors[t.0 as usize].whole(),
                                    };
                                    match cat {
                                        Cat::Elem => slice,
                                        Cat::Rope if ii == 0 || ii == 3 => slice,
                                        _ => whole,
                                    }
                                })
                                .collect();
                            let id = SubtileId(nodes.len() as u32);
                            nodes.push(SubtileNode {
                                id,
                                op: subop,
                                inputs,
                                output: TensorRegion {
                                    tensor: out_t,
                                    region: Region {
                                        rows: Range::new(0, m),
                                        cols: blk,
                                    },
                                },
                            });
                            owner.push(w);
                            blocks_emitted.push(blk);
                        }
                        this_out = OpOut::Part(out_t);
                    }
                }
            }
        }
        op_out.push(this_out);
        op_cols.push(out_cols);
        op_blocks.push(blocks_emitted);
        op_partitioned.push(partitioned);
    }

    // The forward result is the layer output; if replicated, any copy is the
    // same value — take copy 0.
    let result = match &op_out[input.result] {
        OpOut::Part(t) => *t,
        OpOut::Repl(reps) => reps[0],
    };
    let graph = SubtileIR {
        tensors,
        num_sources,
        nodes,
        result,
    };
    debug_assert_eq!(owner.len(), graph.nodes.len(), "one owner per node");
    (graph, owner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{OpDesc, fuse_silu_mul};
    use crate::mega::{Geometry, SourceDesc, op_kind, serialize};
    use crate::metal_tape::{BufferRef, WeightBundle, WeightLoc, WeightRole};
    use crate::region_schedule::{TapeInstr, play, schedule_from_assignment};
    use crate::subtile_ir::{
        SourceShape, eval_dag, lower_region, predecessors, result_buffer, validate,
    };

    fn rng_fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (s >> 33) as u32;
                (bits as f32 / 2147483648.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// Max abs / rel difference between two buffers.
    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(&x, &y)| {
                let d = (x - y).abs();
                d <= tol || d <= tol * x.abs().max(y.abs())
            })
    }

    /// A whole Llama-style decode layer (same fixture shape as subtile_ir's
    /// `full_decode_layer_nblock_bit_exact`), returning the input + sources +
    /// the bit-exact reference result via `lower_region` (N-block, no split-K).
    fn decode_layer() -> (LoweringInput, Vec<Vec<f32>>, u32) {
        let (h, hd, hq, hkv, i, l) = (16u32, 4u32, 4u32, 2u32, 32u32, 3u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let eps = 1e-5f32;
        let scale = 1.0 / (hd as f32).sqrt();
        let data = vec![
            rng_fill(h as usize, 101),           // 0 res_in
            rng_fill(h as usize, 102),           // 1 in_ln
            rng_fill((qdim * h) as usize, 103),  // 2 wq
            rng_fill((kvdim * h) as usize, 104), // 3 wk
            rng_fill((kvdim * h) as usize, 105), // 4 wv
            rng_fill(hd as usize, 106),          // 5 cos
            rng_fill(hd as usize, 107),          // 6 sin
            rng_fill((l * kvdim) as usize, 108), // 7 prefixK
            rng_fill((l * kvdim) as usize, 109), // 8 prefixV
            rng_fill((h * qdim) as usize, 110),  // 9 wo
            rng_fill(h as usize, 111),           // 10 post_ln
            rng_fill((i * h) as usize, 112),     // 11 wgate
            rng_fill((i * h) as usize, 113),     // 12 wup
            rng_fill((h * i) as usize, 114),     // 13 wdown
        ];
        let ss = |rows: u32, cols: u32| SourceShape { rows, cols };
        let input = LoweringInput {
            sources: vec![
                ss(1, h),
                ss(1, h),
                ss(qdim, h),
                ss(kvdim, h),
                ss(kvdim, h),
                ss(1, hd),
                ss(1, hd),
                ss(l, kvdim),
                ss(l, kvdim),
                ss(h, qdim),
                ss(1, h),
                ss(i, h),
                ss(i, h),
                ss(h, i),
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: qdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(3)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(4)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate { head_dim: hd },
                    m: 1,
                    inputs: vec![InputRef::Op(1), InputRef::Ext(5), InputRef::Ext(6)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate { head_dim: hd },
                    m: 1,
                    inputs: vec![InputRef::Op(2), InputRef::Ext(5), InputRef::Ext(6)],
                },
                OpDesc {
                    op: LoweredOp::AttnDecode {
                        num_q_heads: hq,
                        num_kv_heads: hkv,
                        head_dim: hd,
                        scale,
                    },
                    m: 1,
                    inputs: vec![
                        InputRef::Op(4),
                        InputRef::Ext(7),
                        InputRef::Ext(8),
                        InputRef::Op(5),
                        InputRef::Op(3),
                    ],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: h },
                    m: 1,
                    inputs: vec![InputRef::Op(6), InputRef::Ext(9)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(7), InputRef::Ext(0)],
                },
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Op(8), InputRef::Ext(10)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: i },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(11)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(10)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: i },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(12)],
                },
                OpDesc {
                    op: LoweredOp::Mul,
                    m: 1,
                    inputs: vec![InputRef::Op(11), InputRef::Op(12)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: h },
                    m: 1,
                    inputs: vec![InputRef::Op(13), InputRef::Ext(13)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(14), InputRef::Op(8)],
                },
            ],
            result: 15,
        };
        (input, data, hd)
    }

    /// The partitioned decode layer is structurally valid, replays bit-exact
    /// vs its own `eval_dag` (Tier A: any owner assignment is correct), and is
    /// within f32 tol of the N-block reference (Tier A′: o_proj/down split-K
    /// reassociates the reduction, nothing else).
    #[test]
    fn partitioned_decode_layer_matches_reference() {
        let (input, data, hd) = decode_layer();
        let srcs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        // Bit-exact reference (N-block, single reduction per output).
        let gref = lower_region(&input, std::num::NonZeroU32::new(1000).unwrap());
        let want = result_buffer(&gref, &eval_dag(&gref, &srcs)).to_vec();

        for p in [1u32, 2, 4, 8, 10] {
            let (g, owner) = lower_partitioned(&input, hd, hd, p);
            assert!(validate(&g).is_ok(), "partitioned graph valid (p={p})");
            assert_eq!(owner.len(), g.nodes.len());
            assert!(owner.iter().all(|&w| w < p), "owners in range (p={p})");

            // Tier A: the scheduled tape replays bit-exact vs eval_dag(g).
            let self_ref = result_buffer(&g, &eval_dag(&g, &srcs)).to_vec();
            let preds = predecessors(&g);
            let sched = schedule_from_assignment(&g, &preds, &owner, p as usize);
            assert_eq!(
                sched.total_computes(),
                g.nodes.len(),
                "every node once (p={p})"
            );
            let got = result_buffer(&g, &play(&g, &sched, &srcs)).to_vec();
            assert_eq!(
                got, self_ref,
                "partitioned replay bit-exact vs eval_dag (p={p})"
            );

            // Tier A′: within tol of the N-block reference (split-K only).
            assert!(
                close(&got, &want, 1e-4),
                "partitioned within tol of N-block reference (p={p})"
            );
        }
    }

    /// o_proj and down become split-K: one partial per activation block,
    /// COALESCED per worker (a same-worker local SumReduce), then a
    /// CENTRALIZED global all-reduce — a single SumReduce on worker 0 sums all
    /// P local partials. The two GLOBAL all-reduces are identified here as
    /// SumReduces whose input set is the FULL per-worker partials list
    /// (size P), vs the per-worker local sums whose inputs are within-worker
    /// blocks. Centralizing cuts the global all-reduce handoff count from
    /// O(P²) (P replicated copies × P partials) to O(2P) (one copy reading
    /// P-1 cross-worker partials + P-1 cross-worker broadcasts of its result).
    #[test]
    fn reductions_are_split_k_and_centralized() {
        let p = 4u32;
        let (input, _data, hd) = decode_layer();
        let (g, owner) = lower_partitioned(&input, hd, hd, p);
        let global_reduces: Vec<usize> = g
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, n)| {
                (n.op == SubOp::SumReduce && n.inputs.len() == p as usize).then_some(id)
            })
            .collect();
        assert_eq!(
            global_reduces.len(),
            2,
            "exactly two centralized global all-reduces (o_proj + down)"
        );
        for &id in &global_reduces {
            assert_eq!(
                owner[id], 0,
                "centralized global all-reduce node owned by worker 0"
            );
        }
    }

    /// Replication makes the backbone (rmsnorm/add) local to every worker:
    /// each whole-op becomes P copies (one per worker), so its consumers read a
    /// same-worker copy — no cross-worker broadcast wait.
    #[test]
    fn whole_ops_are_replicated_per_worker() {
        let p = 4u32;
        let (input, _data, hd) = decode_layer();
        let (g, owner) = lower_partitioned(&input, hd, hd, p);
        // 2 rmsnorm + 2 add = 4 whole-ops → 4·P copies, balanced one per worker.
        let rms = g
            .nodes
            .iter()
            .filter(|n| matches!(n.op, SubOp::RmsNorm { .. }))
            .count();
        let add = g
            .nodes
            .iter()
            .filter(|n| n.op == SubOp::Elementwise(EwKind::Add))
            .count();
        assert_eq!(rms as u32, 2 * p, "2 rmsnorm × P copies");
        assert_eq!(add as u32, 2 * p, "2 residual-add × P copies");
        let mut rms_by_worker = vec![0u32; p as usize];
        for (id, n) in g.nodes.iter().enumerate() {
            if matches!(n.op, SubOp::RmsNorm { .. }) {
                rms_by_worker[owner[id] as usize] += 1;
            }
        }
        assert!(
            rms_by_worker.iter().all(|&c| c == 2),
            "each worker holds both rmsnorm copies: {rms_by_worker:?}"
        );
    }

    /// Slice-index placement keeps each q-head chain local: the q-proj head
    /// block, its q-rope block, and its attn block share columns ⇒ same owner;
    /// and the o_proj partial reading that attn block is co-located with it.
    #[test]
    fn q_head_chain_is_local() {
        let (input, _data, hd) = decode_layer();
        let p = 4u32;
        let (g, owner) = lower_partitioned(&input, hd, hd, p);
        // For every attn node, the matching q-rope / o-partial nodes (same
        // q-head columns) share its owner. Find attn nodes and their column.
        let preds = predecessors(&g);
        let sched = schedule_from_assignment(&g, &preds, &owner, p as usize);
        // The only cross-worker Waits should be into SumReduce (the all-reduce)
        // and the whole-op consumers (rmsnorm/add reading the all-reduced
        // output / spreading to the matmul blocks) — NOT inside a q-head chain.
        // Concretely: no attn node Waits on a producer that isn't its own
        // q-rope. Assert each attn block's q-rope predecessor is same-owner.
        for (id, n) in g.nodes.iter().enumerate() {
            if let SubOp::AttnDecode { .. } = n.op {
                // q (input 0) is the q-rope output; its writer must be same owner.
                let q_tensor = n.inputs[0].tensor;
                let q_cols = n.inputs[0].region.cols;
                for p2 in &preds[id] {
                    let prod = &g.nodes[p2.0 as usize];
                    if prod.output.tensor == q_tensor
                        && prod.output.region.cols.start == q_cols.start
                    {
                        assert_eq!(
                            owner[id], owner[p2.0 as usize],
                            "attn block co-located with its q-rope block"
                        );
                    }
                }
            }
        }
        // Sanity: the schedule has some structure (computes == nodes).
        assert_eq!(
            sched
                .workers
                .iter()
                .flat_map(|w| &w.tape)
                .filter(|i| matches!(i, TapeInstr::Compute(_)))
                .count(),
            g.nodes.len()
        );
    }

    /// **The partition's purpose, quantified.** Every cross-worker
    /// producer→consumer edge lands on a *genuine* join only: the consumer is
    /// one of
    ///   - `SumReduce` (the centralized all-reduce on worker 0 reading the
    ///     P−1 cross-worker local partials),
    ///   - `RmsNorm`/`Elementwise(Add)` (the broadcast leg: replicated copies
    ///     on workers 1..P−1 cross-worker-read the centralized all-reduce
    ///     output), or
    ///   - `AttnDecode` (cheap GQA K/V sharing — gqa q-heads share one
    ///     kv-head's rope output).
    /// NO cross-worker edge falls inside a q-head chain (q→rope→attn→
    /// o-partial) or an mlp chain (gate/up→silu→down-partial). The handoff
    /// count is O(2P) per all-reduce (centralized reduce + broadcast),
    /// vs the prior P-replicated emit's O(P²).
    #[test]
    fn cross_worker_edges_only_at_genuine_joins() {
        for p in [2u32, 4, 8, 10] {
            let (input, _data, hd) = decode_layer();
            let (g, owner) = lower_partitioned(&input, hd, hd, p);
            let preds = predecessors(&g);
            let mut cross = 0u32;
            for (cid, ps) in preds.iter().enumerate() {
                for pr in ps {
                    if owner[cid] != owner[pr.0 as usize] {
                        cross += 1;
                        let consumer = &g.nodes[cid].op;
                        assert!(
                            matches!(
                                consumer,
                                SubOp::SumReduce
                                    | SubOp::AttnDecode { .. }
                                    | SubOp::RmsNorm { .. }
                                    | SubOp::Elementwise(EwKind::Add)
                            ),
                            "cross-worker edge into {consumer:?} (node {cid}) — \
                             only all-reduce / broadcast / GQA K-share may cross (p={p})"
                        );
                    }
                }
            }
            assert!(cross > 0, "the all-reduce does cross workers (p={p})");
        }
    }

    fn wloc(i: u32) -> WeightLoc {
        WeightLoc {
            layer: 0,
            bucket: 0,
            op_idx: i,
            slot: 0,
        }
    }
    /// A quantized weight source (gs=4 so the toy fixture's head_dim=4 split-K
    /// chunks are group-aligned; 4-bit, f16 scales).
    fn qw(i: u32) -> SourceDesc {
        let w = |role| BufferRef::Weight {
            bundle: WeightBundle::LinearLayer,
            role,
            loc: wloc(i),
        };
        SourceDesc::QuantWeight {
            weight: w(WeightRole::Weight),
            scales: w(WeightRole::AffineScales),
            biases: w(WeightRole::AffineBiases),
            group_size: 4,
            bits: 4,
            scale_elem: 2,
        }
    }
    fn dense(i: u32, bundle: WeightBundle) -> SourceDesc {
        SourceDesc::Dense {
            buffer: BufferRef::Weight {
                bundle,
                role: WeightRole::Weight,
                loc: wloc(i),
            },
            elem: 2,
        }
    }
    /// SourceDescs parallel to `decode_layer`'s 14 sources (weights → quant,
    /// gains/cos/sin → dense, prefix KV → Prefix*).
    fn decode_layer_descs() -> Vec<SourceDesc> {
        vec![
            dense(0, WeightBundle::Embedding),   // 0 res_in (activation)
            dense(1, WeightBundle::LinearLayer), // 1 in_ln gain
            qw(2),
            qw(3),
            qw(4),                                // 2-4 wq/wk/wv
            dense(5, WeightBundle::CosSin),       // 5 cos (rotary table)
            dense(6, WeightBundle::CosSin),       // 6 sin
            SourceDesc::PrefixK { layer: 0 },     // 7 prefixK
            SourceDesc::PrefixV { layer: 0 },     // 8 prefixV
            qw(9),                                // 9 wo
            dense(10, WeightBundle::LinearLayer), // 10 post_ln gain
            qw(11),
            qw(12),
            qw(13), // 11-13 wgate/wup/wdown
        ]
    }

    /// **The GPU-wiring structural proof**: the partitioned decode layer
    /// serializes end-to-end through the new emit paths — `2·P` `SUM_REDUCE`
    /// all-reduce copies, split-K partials lowered to `QMV_QUAD` (their K is
    /// head_dim, below qmv_fast's 512 minimum), and every attn op carrying a
    /// head_range (one q-head per head-tiled block). (Host serialize only — the
    /// toy fixture's tiny K is GPU-runnable only at real shapes; the arms are
    /// bit-exact-verified separately in `wavefront_layer_gpu`.)
    #[test]
    fn partitioned_decode_layer_serializes() {
        let (raw, _data, hd) = decode_layer();
        // The serializer needs the fused SiluMul (as the production path fuses
        // pre-schedule); lower_partitioned then lowers it to SubOp::SiluMul.
        let input = fuse_silu_mul(&raw);
        let sources = decode_layer_descs();
        let geom = Geometry {
            act_elem: 2,
            block_size: 16,
            max_blocks: 4,
        };
        let is_op = |prog: &crate::mega::MegaProgram, i: &[u32; 4], op: u32| {
            i[0] == 0 && prog.shapes[i[1] as usize][0] == op
        };
        for p in [1u32, 2, 4] {
            let (g, owner) = lower_partitioned(&input, hd, hd, p);
            let preds = predecessors(&g);
            let sched = schedule_from_assignment(&g, &preds, &owner, p as usize);
            let prog = serialize(&g, &sched, &sources, geom)
                .unwrap_or_else(|e| panic!("partition serializes (p={p}): {e}"));

            // ≥ 2 SUM_REDUCE: the centralized global all-reduce is ONE copy
            // × 2 reductions; per-worker coalescing adds local sums where a
            // worker owns >1 block.
            let sumr = prog
                .tape
                .iter()
                .filter(|i| is_op(&prog, i, op_kind::SUM_REDUCE))
                .count();
            assert!(
                sumr >= 2,
                "≥ 2 SUM_REDUCE (centralized global all-reduce + any local coalescing) \
                 (p={p}), got {sumr}"
            );

            let quad = prog
                .tape
                .iter()
                .filter(|i| is_op(&prog, i, op_kind::QMV_QUAD))
                .count();
            assert!(quad > 0, "split-K partials lower to qmv_quad (p={p})");

            for i in &prog.tape {
                if is_op(&prog, i, op_kind::ATTN) {
                    let head_range = prog.shapes[i[1] as usize][7];
                    assert_ne!(head_range, 0, "attn carries a head_range (p={p})");
                    assert_eq!(
                        head_range & 0xFFFF,
                        1,
                        "head-tiled attn computes one q-head per block (p={p})"
                    );
                }
            }

            // Every region node serialized to exactly one compute (handoff
            // PUBLISH/ACQUIRE are extra plumbing, excluded by num_computes).
            assert_eq!(
                prog.num_computes(),
                g.nodes.len(),
                "every partition node serialized once (p={p})"
            );
        }
    }
}
