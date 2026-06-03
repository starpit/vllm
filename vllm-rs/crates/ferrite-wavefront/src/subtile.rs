// SPDX-License-Identifier: Apache-2.0
//! Phase 7 (PD-wavefront): lower the *solved* FUF to a subtile dataflow
//! graph — the granularity at which the persistent decode megakernel is
//! scheduled.
//!
//! `fuf.rs` is frozen. It is whole-op granularity: a linear chain of
//! parallel ops with a join between each, where the irregular ops
//! (attention) leave most workers idle. The wavefront megakernel needs a
//! finer unit so cheap irregular ops can overlap into the matvec
//! bandwidth shadow. That finer unit is the **subtile**: an output
//! (row-block × col-block) computed over a K-chunk, plus the split-K
//! combine that reduces the chunks.
//!
//! This module introduces a distinct IR rather than mutating `FufNode`,
//! because (a) the FUF is a hub type consumed by the solver, cost model,
//! tp/vision lowering, and codegen — adding subtile variants there has
//! large blast radius and breaks the "one node = one DSL op = one Impl"
//! invariant — and (b) producer→consumer *sync* is a scheduling artifact,
//! not forward-pass semantics, so it has no business in an IR that "has
//! no idea what a transformer is." This is the standard lower-to-a-new-
//! dialect move.
//!
//! Two representations live here:
//!   - [`SubtileGraph`] — a pure dataflow **DAG**. Producer→consumer
//!     edges are the [`Operand::Sub`] references. This is what lowering
//!     produces and what the decomposition-equivalence check validates.
//!   - (later) a **Tape** — per-worker instruction lists where the
//!     wavefront scheduler turns surviving cross-worker edges into
//!     explicit `Wait`/`Signal`. That is what the host/GPU players replay.
//!
//! Validation is host-first (correctness is what bit prior attempts):
//!   - **Tier A′ (decomposition equivalence):** [`eval_dag`] of the DAG
//!     equals `cpu_golden` whole-op output. Bit-exact at `k_chunks = 1`
//!     (col-tiling does not change any per-output reduction order); only
//!     temp=0-token-exact once K is split, since float add is not
//!     associative — that is expected and checked separately.
//!   - **Tier A (self-consistency):** tape replay equals this `eval_dag`
//!     baseline (lands with the scheduler/player).
//!
//! `cpu_golden` (in `ferrite-forward`, which this crate depends on) is the
//! host *calculator* — the deterministic f32 substrate the player computes
//! with — not the correctness *oracle*. The oracle is ferrite-metal
//! non-mega at temp=0.

#![allow(dead_code)]

// ── Identifiers & geometry ─────────────────────────────────────────

/// Dense index into [`SubtileGraph::nodes`]. The DAG is topologically
/// ordered: a node's [`Operand::Sub`] inputs always have smaller ids
/// (straight-line SSA, same invariant as `Fuf`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubtileId(pub u32);

/// Dense index into [`SubtileGraph::sources`] — a leaf buffer bound at
/// eval time: a prior-op activation (when a subgraph is lowered in
/// isolation), a weight, or an extern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub u32);

/// Half-open range `[start, start + len)` along one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub start: u32,
    pub len: u32,
}

impl Range {
    pub fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }
    pub fn end(&self) -> u32 {
        self.start + self.len
    }
}

/// A rectangular slice of a logically row-major `[rows, cols]` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub rows: Range,
    pub cols: Range,
}

/// Logical shape of a leaf source buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceShape {
    pub rows: u32,
    pub cols: u32,
}

// ── Nodes ──────────────────────────────────────────────────────────

/// One input edge of a subtile node.
#[derive(Clone, Debug)]
pub enum Operand {
    /// A rectangular slice of a leaf source buffer.
    Source { id: SourceId, region: Region },
    /// The full output buffer of a producer subtile. Producers are sized
    /// to exactly what the consumer needs, so there is no slicing here —
    /// slicing happens where a value is first read off a `Source`.
    Sub(SubtileId),
}

/// The sub-operation a node performs. This enum grows as ops are ported;
/// every variant has a `cpu_golden`-backed host evaluation in [`eval_dag`]
/// and (later) a validated MSL primitive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SubOp {
    /// Matmul output tile over one K-chunk:
    /// `out[i, j] = Σ_l A[i, l] · W[j, l]`.
    /// `inputs[0]` = A slice `[mr, kr]`; `inputs[1]` = W slice `[nr, kr]`
    /// (W is row-major `[N, K]`, read transposed). Output is the dense
    /// partial `[mr.len, nr.len]` contributed by this K-chunk.
    MatmulTile,
    /// Elementwise sum of equal-shaped inputs — the split-K combine. All
    /// inputs and the output are `[out_rows, out_cols]`.
    SumReduce,
    /// Shape-preserving elementwise op over a tile. Unary (`Silu`) reads
    /// `inputs[0]`; binary (`Mul`, `Add`) read `inputs[0]` and
    /// `inputs[1]`, both matching the output shape. Col-tiling never
    /// reorders a computation, so always bit-exact vs the whole op.
    Elementwise(EwKind),
    /// Fused SwiGLU activation: `out[j] = silu(gate[j]) * up[j]`.
    /// `inputs[0]` = gate, `inputs[1]` = up, both `[out_rows, out_cols]`.
    /// Matches `cpu_golden::fused_gate_up_silu_mul`. The GPU has only a
    /// *fused* `silu_mul` arm (no standalone silu), so the MLP's separate
    /// `Silu` + `Mul` are fused into this one node *before scheduling* (so
    /// the pair lands on one worker); see `crate::lower::fuse_silu_mul`.
    SiluMul,
    /// RMS-norm over each row: `out[i] = x[i] / rms(x[i,:]) * weight`,
    /// `rms = sqrt(mean(x²) + eps)`. `inputs[0]` = x `[rows, cols]`,
    /// `inputs[1]` = weight `[1, cols]`. The per-row reduction is kept
    /// whole (single node) so it is bit-exact vs `cpu_golden::rmsnorm`;
    /// rms-norm is cheap and hides in the matvec shadow, so there is no
    /// reason to split its reduction.
    RmsNorm { eps: f32 },
    /// NeoX-pairing rotary embedding over `[rows, heads * head_dim]`.
    /// `inputs[0]` = x, `inputs[1]` = cos row `[1, >=head_dim]`,
    /// `inputs[2]` = sin row — the new token's position, pre-sliced.
    /// Pairs `(d, d + half)`; matches `cpu_golden::rope`/`rope_append`'s
    /// rotation. Shape-preserving, so bit-exact vs the reference.
    RopeRotate { head_dim: u32 },
    /// The K-side `rope_append` for the GPU megakernel: rotate K (NeoX) **and**
    /// write the rotated K + un-rotated V into the paged KV cache, so the
    /// downstream attention reads the new token from the cache like the oracle
    /// non-mega path (Tier-B exact). `inputs[0]` = K, `inputs[1]` = cos,
    /// `inputs[2]` = sin, `inputs[3]` = V. The host eval is **rotation only**
    /// (identical to [`SubOp::RopeRotate`]): the abstract dataflow model keeps
    /// the new K as an edge into attention (decision #4), so `inputs[3]` (V)
    /// and the cache write are GPU-only — V is carried for the schedule edge
    /// (the cache write consumes it) and so the serializer can bind it. `layer`
    /// names the KV-cache layer the serializer routes the cache operands to.
    RopeAppend { head_dim: u32, layer: u32 },
    /// Decode attention. `inputs[0]` = Q `[Mq, num_q_heads * head_dim]`;
    /// the remaining inputs are alternating `(K_seg, V_seg)`, each
    /// `[seg_len, num_kv_heads * head_dim]`, concatenated along the KV
    /// axis in input order. The **fused** decode passes the prefix cache
    /// as a read-only `Source` segment and the just-rotated new token as
    /// a `Sub` segment — so the new K/V is an internal dataflow edge, not
    /// a cache round-trip. Whole-KV (single node) matches
    /// `cpu_golden::attention_decode` bit-exact; KV-block subtiling +
    /// online-softmax combine comes later (within-tol, the gemm pattern).
    AttnDecode {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    },
}

/// Elementwise op kind. Numerics mirror `cpu_golden` exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwKind {
    /// `x / (1 + e^-x)`.
    Silu,
    /// `a * b`.
    Mul,
    /// `a + b`.
    Add,
}

/// A subtile node: one unit of work the wavefront schedule places on a
/// worker. Produces a dense row-major `[out_rows, out_cols]` f32 buffer.
#[derive(Clone, Debug)]
pub struct SubtileNode {
    pub id: SubtileId,
    pub op: SubOp,
    pub inputs: Vec<Operand>,
    pub out_rows: u32,
    pub out_cols: u32,
}

/// Where one graph result lands in the assembled output buffer.
#[derive(Clone, Copy, Debug)]
pub struct OutputSlot {
    pub node: SubtileId,
    pub dest: Region,
}

/// A subtile dataflow DAG (see module docs).
#[derive(Clone, Debug)]
pub struct SubtileGraph {
    pub nodes: Vec<SubtileNode>,
    pub sources: Vec<SourceShape>,
    pub result_rows: u32,
    pub result_cols: u32,
    pub outputs: Vec<OutputSlot>,
}

// ── Tiling policy ──────────────────────────────────────────────────

/// How finely to subtile a matmul.
#[derive(Clone, Copy, Debug)]
pub struct TilingPolicy {
    /// Output-column block width (N is split into `ceil(N / nb)` blocks).
    pub nb: u32,
    /// Number of split-K reductions. `1` = no K-split, which is bit-exact
    /// with a single sequential reduction. The FUF lowering will set this
    /// from the chosen split-K Impl; for host validation it is a free knob.
    pub k_chunks: u32,
}

impl Default for TilingPolicy {
    fn default() -> Self {
        Self {
            nb: 64,
            k_chunks: 1,
        }
    }
}

/// Split `[0, total)` into `parts` contiguous ranges, as even as
/// possible (the first `total % parts` ranges are one longer). Never
/// produces more ranges than elements.
fn split_range(total: u32, parts: u32) -> Vec<Range> {
    let parts = parts.max(1).min(total.max(1));
    let base = total / parts;
    let rem = total % parts;
    let mut out = Vec::with_capacity(parts as usize);
    let mut start = 0;
    for i in 0..parts {
        let len = base + if i < rem { 1 } else { 0 };
        out.push(Range::new(start, len));
        start += len;
    }
    out
}

/// Tile `[0, total)` into contiguous blocks of width `block` (last block
/// may be shorter).
fn blocks(total: u32, block: u32) -> Vec<Range> {
    assert!(block >= 1, "block width must be >= 1");
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let len = block.min(total - start);
        out.push(Range::new(start, len));
        start += len;
    }
    out
}

// ── Standalone GEMM lowering ───────────────────────────────────────

/// Lower a single dense GEMM `out[M, N] = A[M, K] @ W[N, K]^T` into a
/// subtile DAG with `SourceId(0) = A [M, K]` and `SourceId(1) = W [N, K]`.
///
/// Used for isolated host validation; the FUF lowering reuses the same
/// (N-block × K-chunk) decomposition with FUF-anchored operands. With
/// `k_chunks = 1` each N-block is a single `MatmulTile` (no reduce) and
/// the result is bit-exact with `cpu_golden::gemm`.
pub fn lower_gemm_standalone(m: u32, n: u32, k: u32, policy: TilingPolicy) -> SubtileGraph {
    const A: SourceId = SourceId(0);
    const W: SourceId = SourceId(1);

    let mut nodes: Vec<SubtileNode> = Vec::new();
    let mut outputs: Vec<OutputSlot> = Vec::new();

    let n_blocks = blocks(n, policy.nb);
    let k_chunks = split_range(k, policy.k_chunks);

    for nb in &n_blocks {
        let mut partials: Vec<SubtileId> = Vec::with_capacity(k_chunks.len());
        for kc in &k_chunks {
            let id = SubtileId(nodes.len() as u32);
            nodes.push(SubtileNode {
                id,
                op: SubOp::MatmulTile,
                inputs: vec![
                    Operand::Source {
                        id: A,
                        region: Region {
                            rows: Range::new(0, m),
                            cols: *kc,
                        },
                    },
                    Operand::Source {
                        id: W,
                        region: Region {
                            rows: *nb,
                            cols: *kc,
                        },
                    },
                ],
                out_rows: m,
                out_cols: nb.len,
            });
            partials.push(id);
        }

        // One K-chunk → the matmul tile is already the N-block output.
        // Many → a SumReduce combines them (the split-K reduction).
        let out_node = if partials.len() == 1 {
            partials[0]
        } else {
            let id = SubtileId(nodes.len() as u32);
            nodes.push(SubtileNode {
                id,
                op: SubOp::SumReduce,
                inputs: partials.iter().map(|p| Operand::Sub(*p)).collect(),
                out_rows: m,
                out_cols: nb.len,
            });
            id
        };
        outputs.push(OutputSlot {
            node: out_node,
            dest: Region {
                rows: Range::new(0, m),
                cols: *nb,
            },
        });
    }

    SubtileGraph {
        nodes,
        sources: vec![
            SourceShape { rows: m, cols: k },
            SourceShape { rows: n, cols: k },
        ],
        result_rows: m,
        result_cols: n,
        outputs,
    }
}

/// Lower the SwiGLU MLP activation `out[M, D] = silu(gate) * up`
/// (`SourceId(0) = gate`, `SourceId(1) = up`), col-tiled by `nb`. Each
/// block is a `Silu` node feeding a `Mul` node — the fused-group pattern
/// at subtile granularity (a chained `Sub` edge that crosses workers
/// under partitioning). Bit-exact vs `cpu_golden::fused_gate_up_silu_mul`.
pub fn lower_gate_up_silu_mul_standalone(m: u32, d: u32, nb: u32) -> SubtileGraph {
    const GATE: SourceId = SourceId(0);
    const UP: SourceId = SourceId(1);
    let mut nodes: Vec<SubtileNode> = Vec::new();
    let mut outputs: Vec<OutputSlot> = Vec::new();
    for cb in &blocks(d, nb) {
        let region = Region {
            rows: Range::new(0, m),
            cols: *cb,
        };
        let silu = SubtileId(nodes.len() as u32);
        nodes.push(SubtileNode {
            id: silu,
            op: SubOp::Elementwise(EwKind::Silu),
            inputs: vec![Operand::Source { id: GATE, region }],
            out_rows: m,
            out_cols: cb.len,
        });
        let mul = SubtileId(nodes.len() as u32);
        nodes.push(SubtileNode {
            id: mul,
            op: SubOp::Elementwise(EwKind::Mul),
            inputs: vec![Operand::Sub(silu), Operand::Source { id: UP, region }],
            out_rows: m,
            out_cols: cb.len,
        });
        outputs.push(OutputSlot {
            node: mul,
            dest: region,
        });
    }
    SubtileGraph {
        nodes,
        sources: vec![
            SourceShape { rows: m, cols: d },
            SourceShape { rows: m, cols: d },
        ],
        result_rows: m,
        result_cols: d,
        outputs,
    }
}

/// Lower a residual add `out[M, D] = x + residual` (`SourceId(0) = x`,
/// `SourceId(1) = residual`), col-tiled by `nb`. Bit-exact vs
/// `cpu_golden::add`.
pub fn lower_residual_add_standalone(m: u32, d: u32, nb: u32) -> SubtileGraph {
    const X: SourceId = SourceId(0);
    const RES: SourceId = SourceId(1);
    let mut nodes: Vec<SubtileNode> = Vec::new();
    let mut outputs: Vec<OutputSlot> = Vec::new();
    for cb in &blocks(d, nb) {
        let region = Region {
            rows: Range::new(0, m),
            cols: *cb,
        };
        let add = SubtileId(nodes.len() as u32);
        nodes.push(SubtileNode {
            id: add,
            op: SubOp::Elementwise(EwKind::Add),
            inputs: vec![
                Operand::Source { id: X, region },
                Operand::Source { id: RES, region },
            ],
            out_rows: m,
            out_cols: cb.len,
        });
        outputs.push(OutputSlot {
            node: add,
            dest: region,
        });
    }
    SubtileGraph {
        nodes,
        sources: vec![
            SourceShape { rows: m, cols: d },
            SourceShape { rows: m, cols: d },
        ],
        result_rows: m,
        result_cols: d,
        outputs,
    }
}

/// Lower an RMS-norm `out[M, D] = rmsnorm(x, weight, eps)`
/// (`SourceId(0) = x [M, D]`, `SourceId(1) = weight [1, D]`) as a single
/// whole-row-reduction node. Bit-exact vs `cpu_golden::rmsnorm`.
pub fn lower_rmsnorm_standalone(m: u32, d: u32, eps: f32) -> SubtileGraph {
    const X: SourceId = SourceId(0);
    const WEIGHT: SourceId = SourceId(1);
    let full = Region {
        rows: Range::new(0, m),
        cols: Range::new(0, d),
    };
    let node = SubtileNode {
        id: SubtileId(0),
        op: SubOp::RmsNorm { eps },
        inputs: vec![
            Operand::Source {
                id: X,
                region: full,
            },
            Operand::Source {
                id: WEIGHT,
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(0, d),
                },
            },
        ],
        out_rows: m,
        out_cols: d,
    };
    SubtileGraph {
        nodes: vec![node],
        sources: vec![
            SourceShape { rows: m, cols: d },
            SourceShape { rows: 1, cols: d },
        ],
        result_rows: m,
        result_cols: d,
        outputs: vec![OutputSlot {
            node: SubtileId(0),
            dest: full,
        }],
    }
}

/// Lower a **fused** decode attention step (Mq=1 typical). Sources:
/// `0=q_in [M, Hq*hd]`, `1=k_in [M, Hkv*hd]`, `2=v_in [M, Hkv*hd]`,
/// `3=cos [1, hd]`, `4=sin [1, hd]`, `5=prefix_k [L, Hkv*hd]`,
/// `6=prefix_v [L, Hkv*hd]`.
///
/// Builds `RopeRotate(Q)` and `RopeRotate(K)` nodes, then an `AttnDecode`
/// reading Q_rot, the prefix cache as a read-only `Source` segment, and
/// the rotated new token (`K_new` as a `Sub` edge, `V_new` from the
/// `Source`) as the second segment. The new token is therefore an
/// internal dataflow edge — no cache round-trip. Bit-exact vs
/// `cpu_golden::rope` + `cpu_golden::attention_decode`.
pub fn lower_decode_attention_standalone(
    m: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    prefix_len: u32,
    scale: f32,
) -> SubtileGraph {
    const Q: SourceId = SourceId(0);
    const K: SourceId = SourceId(1);
    const V: SourceId = SourceId(2);
    const COS: SourceId = SourceId(3);
    const SIN: SourceId = SourceId(4);
    const PK: SourceId = SourceId(5);
    const PV: SourceId = SourceId(6);
    let qdim = num_q_heads * head_dim;
    let kvdim = num_kv_heads * head_dim;
    let full = |rows: u32, cols: u32| Region {
        rows: Range::new(0, rows),
        cols: Range::new(0, cols),
    };
    let trig = |id: SourceId| Operand::Source {
        id,
        region: full(1, head_dim),
    };

    let rope_q = SubtileId(0);
    let rope_k = SubtileId(1);
    let attn = SubtileId(2);
    let nodes = vec![
        SubtileNode {
            id: rope_q,
            op: SubOp::RopeRotate { head_dim },
            inputs: vec![
                Operand::Source {
                    id: Q,
                    region: full(m, qdim),
                },
                trig(COS),
                trig(SIN),
            ],
            out_rows: m,
            out_cols: qdim,
        },
        SubtileNode {
            id: rope_k,
            op: SubOp::RopeRotate { head_dim },
            inputs: vec![
                Operand::Source {
                    id: K,
                    region: full(m, kvdim),
                },
                trig(COS),
                trig(SIN),
            ],
            out_rows: m,
            out_cols: kvdim,
        },
        SubtileNode {
            id: attn,
            op: SubOp::AttnDecode {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
            },
            inputs: vec![
                Operand::Sub(rope_q),
                Operand::Source {
                    id: PK,
                    region: full(prefix_len, kvdim),
                },
                Operand::Source {
                    id: PV,
                    region: full(prefix_len, kvdim),
                },
                Operand::Sub(rope_k),
                Operand::Source {
                    id: V,
                    region: full(m, kvdim),
                },
            ],
            out_rows: m,
            out_cols: qdim,
        },
    ];
    SubtileGraph {
        nodes,
        sources: vec![
            SourceShape {
                rows: m,
                cols: qdim,
            },
            SourceShape {
                rows: m,
                cols: kvdim,
            },
            SourceShape {
                rows: m,
                cols: kvdim,
            },
            SourceShape {
                rows: 1,
                cols: head_dim,
            },
            SourceShape {
                rows: 1,
                cols: head_dim,
            },
            SourceShape {
                rows: prefix_len,
                cols: kvdim,
            },
            SourceShape {
                rows: prefix_len,
                cols: kvdim,
            },
        ],
        result_rows: m,
        result_cols: qdim,
        outputs: vec![OutputSlot {
            node: attn,
            dest: full(m, qdim),
        }],
    }
}

// ── Host evaluation (the direct topological reference) ─────────────

/// Materialize one operand into a dense row-major `(buf, rows, cols)`.
fn read_operand(
    op: &Operand,
    graph: &SubtileGraph,
    sources: &[&[f32]],
    outs: &[Vec<f32>],
) -> (Vec<f32>, u32, u32) {
    match op {
        Operand::Sub(id) => {
            let node = &graph.nodes[id.0 as usize];
            (outs[id.0 as usize].clone(), node.out_rows, node.out_cols)
        }
        Operand::Source { id, region } => {
            let shape = graph.sources[id.0 as usize];
            let src = sources[id.0 as usize];
            let (r, c) = (region.rows.len, region.cols.len);
            let mut buf = Vec::with_capacity((r * c) as usize);
            for i in 0..r {
                let row = region.rows.start + i;
                let base = ((row * shape.cols) + region.cols.start) as usize;
                buf.extend_from_slice(&src[base..base + c as usize]);
            }
            (buf, r, c)
        }
    }
}

/// Evaluate a subtile DAG on the host by direct topological replay.
/// `sources[s]` is the row-major buffer for `SourceId(s)`, matching
/// `graph.sources[s]`. Returns one dense buffer per node (indexed by
/// `SubtileId`). This is both the Tier A′ reference and the baseline the
/// tape player must reproduce bit-for-bit (Tier A).
pub fn eval_dag(graph: &SubtileGraph, sources: &[&[f32]]) -> Vec<Vec<f32>> {
    assert_eq!(sources.len(), graph.sources.len(), "source count mismatch");
    for (s, shape) in sources.iter().zip(&graph.sources) {
        assert_eq!(
            s.len(),
            (shape.rows * shape.cols) as usize,
            "source buffer size mismatch"
        );
    }

    let mut outs: Vec<Vec<f32>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let buf = eval_node(node, graph, sources, &outs);
        outs.push(buf);
    }
    outs
}

/// Compute one node's output buffer. `outs[p]` must already hold the
/// output of every producer `p` read via [`Operand::Sub`]: [`eval_dag`]
/// guarantees this by topological order, the tape player by `Wait`
/// edges. Pure — depends only on its inputs — so direct topological eval
/// and scheduled tape replay agree bit-for-bit (Tier A).
pub fn eval_node(
    node: &SubtileNode,
    graph: &SubtileGraph,
    sources: &[&[f32]],
    outs: &[Vec<f32>],
) -> Vec<f32> {
    match node.op {
        SubOp::MatmulTile => {
            let (a, ar, ac) = read_operand(&node.inputs[0], graph, sources, outs);
            let (w, wr, wc) = read_operand(&node.inputs[1], graph, sources, outs);
            assert_eq!(ac, wc, "matmul K mismatch");
            assert_eq!(ar, node.out_rows, "matmul A rows vs out_rows");
            assert_eq!(wr, node.out_cols, "matmul W rows vs out_cols");
            let (m, n, k) = (ar as usize, wr as usize, ac as usize);
            let mut out = vec![0f32; m * n];
            // Identical loop nest and accumulation order to
            // cpu_golden::gemm so k_chunks=1 is bit-exact.
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0f32;
                    for l in 0..k {
                        sum += a[i * k + l] * w[j * k + l];
                    }
                    out[i * n + j] = sum;
                }
            }
            out
        }
        SubOp::SumReduce => {
            let len = (node.out_rows * node.out_cols) as usize;
            let mut out = vec![0f32; len];
            for inp in &node.inputs {
                let (b, _, _) = read_operand(inp, graph, sources, outs);
                assert_eq!(b.len(), len, "reduce operand size mismatch");
                for (o, v) in out.iter_mut().zip(&b) {
                    *o += *v;
                }
            }
            out
        }
        SubOp::Elementwise(kind) => {
            let (a, ar, ac) = read_operand(&node.inputs[0], graph, sources, outs);
            assert_eq!(ar, node.out_rows, "elementwise rows");
            assert_eq!(ac, node.out_cols, "elementwise cols");
            match kind {
                // silu: x / (1 + e^-x) — exactly cpu_golden::silu.
                EwKind::Silu => a.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
                EwKind::Mul | EwKind::Add => {
                    let (b, br, bc) = read_operand(&node.inputs[1], graph, sources, outs);
                    assert_eq!((br, bc), (ar, ac), "elementwise binary shape");
                    a.iter()
                        .zip(&b)
                        .map(|(&x, &y)| match kind {
                            EwKind::Mul => x * y,
                            _ => x + y,
                        })
                        .collect()
                }
            }
        }
        SubOp::SiluMul => {
            let (a, ar, ac) = read_operand(&node.inputs[0], graph, sources, outs);
            let (b, br, bc) = read_operand(&node.inputs[1], graph, sources, outs);
            assert_eq!(
                (ar, ac),
                (node.out_rows, node.out_cols),
                "silu_mul gate shape"
            );
            assert_eq!((br, bc), (ar, ac), "silu_mul up shape");
            a.iter()
                .zip(&b)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect()
        }
        SubOp::RmsNorm { eps } => {
            let (x, xr, xc) = read_operand(&node.inputs[0], graph, sources, outs);
            let (wt, _wr, wc) = read_operand(&node.inputs[1], graph, sources, outs);
            assert_eq!(xr, node.out_rows, "rmsnorm rows");
            assert_eq!(xc, node.out_cols, "rmsnorm cols");
            assert_eq!(wc, node.out_cols, "rmsnorm weight width");
            let (m, d) = (xr as usize, xc as usize);
            let mut out = vec![0f32; m * d];
            for i in 0..m {
                let row = &x[i * d..(i + 1) * d];
                // Exactly cpu_golden::rmsnorm: left-to-right sum of squares.
                let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
                let inv_rms = 1.0 / (sum_sq / d as f32 + eps).sqrt();
                for j in 0..d {
                    out[i * d + j] = row[j] * inv_rms * wt[j];
                }
            }
            out
        }
        // RopeAppend's host eval is rotation only (identical to RopeRotate);
        // its V input (3) and the paged-cache write are GPU-only.
        SubOp::RopeRotate { head_dim } | SubOp::RopeAppend { head_dim, .. } => {
            let (x, xr, xc) = read_operand(&node.inputs[0], graph, sources, outs);
            let (cos, _, cc) = read_operand(&node.inputs[1], graph, sources, outs);
            let (sin, _, sc) = read_operand(&node.inputs[2], graph, sources, outs);
            assert_eq!(xr, node.out_rows, "rope rows");
            assert_eq!(xc, node.out_cols, "rope cols");
            let hd = head_dim as usize;
            let half = hd / 2;
            let (rows, cols) = (xr as usize, xc as usize);
            assert_eq!(cols % hd, 0, "rope cols not a multiple of head_dim");
            assert!(cc as usize >= hd && sc as usize >= hd, "rope cos/sin width");
            let heads = cols / hd;
            let mut out = x.clone();
            for r in 0..rows {
                for h in 0..heads {
                    let base = r * cols + h * hd;
                    for d in 0..half {
                        let x0 = x[base + d];
                        let x1 = x[base + half + d];
                        let (c, s) = (cos[d], sin[d]);
                        // Matches cpu_golden::rope: (x0,x1) → (x0 c − x1 s, x0 s + x1 c).
                        out[base + d] = x0 * c - x1 * s;
                        out[base + half + d] = x0 * s + x1 * c;
                    }
                }
            }
            out
        }
        SubOp::AttnDecode {
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
        } => {
            let (hq, hkv, hd) = (
                num_q_heads as usize,
                num_kv_heads as usize,
                head_dim as usize,
            );
            let (q, qr, qc) = read_operand(&node.inputs[0], graph, sources, outs);
            assert_eq!(qc as usize, hq * hd, "attn Q width");
            let mq = qr as usize;
            // Concatenate the (K_seg, V_seg) pairs along the KV axis — the
            // prefix Source segment then the new-token Sub segment.
            assert!(node.inputs.len() >= 3, "attn needs Q + >=1 (K,V) segment");
            assert_eq!(node.inputs.len() % 2, 1, "attn inputs = Q + (K,V) pairs");
            let mut k_all: Vec<f32> = Vec::new();
            let mut v_all: Vec<f32> = Vec::new();
            let mut i = 1;
            while i < node.inputs.len() {
                let (k, kr, kc) = read_operand(&node.inputs[i], graph, sources, outs);
                let (v, vr, vc) = read_operand(&node.inputs[i + 1], graph, sources, outs);
                assert_eq!(kc as usize, hkv * hd, "attn K seg width");
                assert_eq!((vr, vc), (kr, kc), "attn V seg shape");
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(&v);
                i += 2;
            }
            let seq_len = k_all.len() / (hkv * hd);
            let gqa = hq / hkv;
            let mut out = vec![0f32; mq * hq * hd];
            // Per-(row, head) softmax over the full KV — identical to
            // cpu_golden::attention_decode (max-shift, exp, normalize).
            for qi in 0..mq {
                for h in 0..hq {
                    let kv_h = h / gqa;
                    let q_off = qi * hq * hd + h * hd;
                    let mut scores = vec![0f32; seq_len];
                    for (s, score) in scores.iter_mut().enumerate() {
                        let k_off = s * hkv * hd + kv_h * hd;
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += q[q_off + d] * k_all[k_off + d];
                        }
                        *score = dot * scale;
                    }
                    let maxs = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - maxs).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
                    }
                    for d in 0..hd {
                        let mut val = 0f32;
                        for (s, &w) in scores.iter().enumerate() {
                            val += w * v_all[s * hkv * hd + kv_h * hd + d];
                        }
                        out[q_off + d] = val;
                    }
                }
            }
            out
        }
    }
}

/// Scatter the node outputs named by `graph.outputs` into the assembled
/// `[result_rows, result_cols]` row-major buffer.
pub fn assemble_result(graph: &SubtileGraph, outs: &[Vec<f32>]) -> Vec<f32> {
    let (rr, rc) = (graph.result_rows as usize, graph.result_cols as usize);
    let mut res = vec![0f32; rr * rc];
    for slot in &graph.outputs {
        let buf = &outs[slot.node.0 as usize];
        let (br, bc) = (slot.dest.rows.len as usize, slot.dest.cols.len as usize);
        assert_eq!(buf.len(), br * bc, "output buffer vs dest region");
        for i in 0..br {
            let dst_row = slot.dest.rows.start as usize + i;
            for j in 0..bc {
                let dst_col = slot.dest.cols.start as usize + j;
                res[dst_row * rc + dst_col] = buf[i * bc + j];
            }
        }
    }
    res
}

// ── Structural validation ─────────────────────────────────────────

/// Check a [`SubtileGraph`]'s structural invariants without evaluating
/// it: every `Sub` edge points strictly backward (so the DAG is acyclic
/// and already topologically ordered), every `Source` operand is in
/// range with an in-bounds region, each node's input arity matches its
/// op, and the outputs reference real nodes. Returns the node count on
/// success, or a human-readable description of the first violation.
///
/// Cheap and allocation-free; the macro→wavefront bridge runs it on a
/// real model's lowered graph at expansion time to turn "`lower()`
/// didn't panic" into an explicit, logged well-formedness guarantee.
pub fn validate(graph: &SubtileGraph) -> Result<usize, String> {
    let n_src = graph.sources.len() as u32;
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.id.0 as usize != i {
            return Err(format!("node {i} has non-dense id {}", node.id.0));
        }
        let arity = node.inputs.len();
        let arity_ok = match node.op {
            SubOp::MatmulTile => arity == 2,
            SubOp::SumReduce => arity >= 1,
            SubOp::Elementwise(EwKind::Silu) => arity == 1,
            SubOp::Elementwise(EwKind::Mul | EwKind::Add) => arity == 2,
            SubOp::SiluMul => arity == 2,
            SubOp::RmsNorm { .. } => arity == 2,
            SubOp::RopeRotate { .. } => arity == 3,
            // E.12: K, cos, sin, V, K_cache, V_cache.
            SubOp::RopeAppend { .. } => arity == 6,
            // Q followed by one or more (K_seg, V_seg) pairs.
            SubOp::AttnDecode { .. } => arity >= 3 && arity % 2 == 1,
        };
        if !arity_ok {
            return Err(format!("node {i} op {:?} bad arity {arity}", node.op));
        }
        for (a, op) in node.inputs.iter().enumerate() {
            match op {
                Operand::Sub(dep) => {
                    if dep.0 as usize >= i {
                        return Err(format!(
                            "node {i} input {a} reads Sub({}) which is not strictly upstream",
                            dep.0
                        ));
                    }
                }
                Operand::Source { id, region } => {
                    if id.0 >= n_src {
                        return Err(format!(
                            "node {i} input {a} reads Source({}) out of {n_src}",
                            id.0
                        ));
                    }
                    let shape = graph.sources[id.0 as usize];
                    if region.rows.end() > shape.rows || region.cols.end() > shape.cols {
                        return Err(format!(
                            "node {i} input {a} region {region:?} exceeds source {:?}",
                            shape
                        ));
                    }
                }
            }
        }
    }
    if graph.outputs.is_empty() {
        return Err("graph has no outputs".into());
    }
    for slot in &graph.outputs {
        if slot.node.0 as usize >= graph.nodes.len() {
            return Err(format!("output references missing node {}", slot.node.0));
        }
    }
    Ok(graph.nodes.len())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_forward::cpu_golden;

    /// Deterministic f32 fill in `[-1, 1)` (LCG; reproducible per seed).
    fn rng_fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (s >> 33) as u32; // top 31 bits → [0, 2^31)
                (bits as f32 / 2147483648.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// `validate` accepts a well-formed graph and rejects a backward-edge
    /// violation. The fused decode-attention lowering is a known-good
    /// graph (Sub edges strictly upstream, in-range sources, odd attn
    /// arity); flipping one `Sub` edge forward must be caught.
    #[test]
    fn validate_accepts_good_rejects_cycle() {
        let g = lower_decode_attention_standalone(1, 4, 2, 4, 3, 0.5);
        assert_eq!(super::validate(&g), Ok(g.nodes.len()));

        // Point the attention node's first input at itself → forward
        // (cyclic) Sub edge; validate must reject it.
        let mut bad = g.clone();
        let last = bad.nodes.len() - 1;
        bad.nodes[last].inputs[0] = Operand::Sub(SubtileId(last as u32));
        assert!(super::validate(&bad).is_err(), "self-edge must be rejected");
    }

    #[test]
    fn split_range_partitions_exactly() {
        let r = split_range(200, 4);
        assert_eq!(r.len(), 4);
        assert_eq!(r.iter().map(|x| x.len).sum::<u32>(), 200);
        assert_eq!(r[0].start, 0);
        assert_eq!(r[3].end(), 200);
        // Uneven split distributes the remainder to the front.
        let u = split_range(10, 3);
        assert_eq!(u.iter().map(|x| x.len).collect::<Vec<_>>(), vec![4, 3, 3]);
        // Contiguous, no gaps/overlaps.
        for w in u.windows(2) {
            assert_eq!(w[0].end(), w[1].start);
        }
    }

    #[test]
    fn blocks_tile_with_short_tail() {
        let b = blocks(100, 32);
        assert_eq!(
            b.iter().map(|x| x.len).collect::<Vec<_>>(),
            vec![32, 32, 32, 4]
        );
        assert_eq!(b.iter().map(|x| x.len).sum::<u32>(), 100);
    }

    #[test]
    fn gemm_structure_block_and_chunk_counts() {
        // N=100, nb=32 → 4 N-blocks; K=200, k_chunks=4 → 4 chunks.
        // 4×4 matmul tiles + 4 reduces = 20 nodes, 4 output slots.
        let g = lower_gemm_standalone(
            3,
            100,
            200,
            TilingPolicy {
                nb: 32,
                k_chunks: 4,
            },
        );
        let mm = g.nodes.iter().filter(|n| n.op == SubOp::MatmulTile).count();
        let rd = g.nodes.iter().filter(|n| n.op == SubOp::SumReduce).count();
        assert_eq!(mm, 16, "4 N-blocks × 4 K-chunks");
        assert_eq!(rd, 4, "one split-K reduce per N-block");
        assert_eq!(g.outputs.len(), 4);
        // Topological invariant: every Sub input points at a smaller id.
        for node in &g.nodes {
            for inp in &node.inputs {
                if let Operand::Sub(p) = inp {
                    assert!(p.0 < node.id.0, "Sub input must precede consumer");
                }
            }
        }
    }

    #[test]
    fn gemm_kchunks1_bit_exact_vs_cpu_golden() {
        let (m, n, k) = (1u32, 96, 256);
        let a = rng_fill((m * k) as usize, 1);
        let w = rng_fill((n * k) as usize, 2);

        let mut want = vec![0f32; (m * n) as usize];
        cpu_golden::gemm(&a, &w, &mut want, m as usize, k as usize, n as usize);

        let g = lower_gemm_standalone(
            m,
            n,
            k,
            TilingPolicy {
                nb: 32,
                k_chunks: 1,
            },
        );
        let got = assemble_result(&g, &eval_dag(&g, &[&a, &w]));

        assert_eq!(got, want, "k_chunks=1 column-tiled GEMM must be bit-exact");
    }

    #[test]
    fn gemm_kchunks1_bit_exact_multirow_uneven_blocks() {
        // M=4 rows, N not a multiple of nb, K not a multiple of nothing
        // (k_chunks=1). Still bit-exact: col-tiling never reorders a
        // per-output reduction.
        let (m, n, k) = (4u32, 130, 257);
        let a = rng_fill((m * k) as usize, 7);
        let w = rng_fill((n * k) as usize, 11);

        let mut want = vec![0f32; (m * n) as usize];
        cpu_golden::gemm(&a, &w, &mut want, m as usize, k as usize, n as usize);

        let g = lower_gemm_standalone(
            m,
            n,
            k,
            TilingPolicy {
                nb: 48,
                k_chunks: 1,
            },
        );
        let got = assemble_result(&g, &eval_dag(&g, &[&a, &w]));

        assert_eq!(got, want, "uneven multi-row k_chunks=1 GEMM bit-exact");
    }

    #[test]
    fn gemm_split_k_matches_within_tol() {
        // Split-K reorders the reduction, so it is NOT bit-exact, but it
        // must agree numerically to f32 tolerance.
        let (m, n, k) = (2u32, 64, 320);
        let a = rng_fill((m * k) as usize, 3);
        let w = rng_fill((n * k) as usize, 5);

        let mut want = vec![0f32; (m * n) as usize];
        cpu_golden::gemm(&a, &w, &mut want, m as usize, k as usize, n as usize);

        let g = lower_gemm_standalone(
            m,
            n,
            k,
            TilingPolicy {
                nb: 16,
                k_chunks: 5,
            },
        );
        let got = assemble_result(&g, &eval_dag(&g, &[&a, &w]));

        for (i, (&x, &y)) in got.iter().zip(&want).enumerate() {
            assert!(
                (x - y).abs() <= 1e-3 + 1e-3 * y.abs(),
                "split-K idx {i}: got {x} vs seq {y}"
            );
        }
    }

    #[test]
    fn gate_up_silu_mul_bit_exact_vs_cpu_golden() {
        let (m, d) = (2u32, 96);
        let gate = rng_fill((m * d) as usize, 31);
        let up = rng_fill((m * d) as usize, 32);
        let mut want = vec![0f32; (m * d) as usize];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut want);
        // Bit-exact at every col-block width, including a tail (d=96, nb=100)
        // and exact divisors.
        for nb in [8u32, 32, 96, 100] {
            let g = lower_gate_up_silu_mul_standalone(m, d, nb);
            let got = assemble_result(&g, &eval_dag(&g, &[&gate, &up]));
            assert_eq!(got, want, "silu(gate)*up must be bit-exact (nb={nb})");
        }
    }

    #[test]
    fn residual_add_bit_exact_vs_cpu_golden() {
        let (m, d) = (3u32, 70);
        let x = rng_fill((m * d) as usize, 41);
        let r = rng_fill((m * d) as usize, 42);
        let mut want = vec![0f32; (m * d) as usize];
        cpu_golden::add(&x, &r, &mut want);
        let g = lower_residual_add_standalone(m, d, 16);
        let got = assemble_result(&g, &eval_dag(&g, &[&x, &r]));
        assert_eq!(got, want, "residual add bit-exact");
    }

    #[test]
    fn rmsnorm_bit_exact_vs_cpu_golden() {
        let (m, d) = (2u32, 128);
        let eps = 1e-5f32;
        let x = rng_fill((m * d) as usize, 61);
        let wt = rng_fill(d as usize, 62);
        // cpu_golden::rmsnorm is per-row; build the [M, D] reference row by row.
        let mut want = vec![0f32; (m * d) as usize];
        let (ms, ds) = (m as usize, d as usize);
        for i in 0..ms {
            cpu_golden::rmsnorm(
                &x[i * ds..(i + 1) * ds],
                &wt,
                &mut want[i * ds..(i + 1) * ds],
                eps,
            );
        }
        let g = lower_rmsnorm_standalone(m, d, eps);
        let got = assemble_result(&g, &eval_dag(&g, &[&x, &wt]));
        assert_eq!(got, want, "rmsnorm bit-exact");
    }

    #[test]
    fn fused_decode_attention_bit_exact_vs_cpu_golden() {
        // GQA: 4 q-heads, 2 kv-heads, head_dim 8, prefix len 5, 1 new token.
        let (hq, hkv, hd, l) = (4u32, 2u32, 8u32, 5u32);
        let (qdim, kvdim) = ((hq * hd) as usize, (hkv * hd) as usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let q_in = rng_fill(qdim, 71);
        let k_in = rng_fill(kvdim, 72);
        let v_in = rng_fill(kvdim, 73);
        let cos = rng_fill(hd as usize, 74);
        let sin = rng_fill(hd as usize, 75);
        let prefix_k = rng_fill(l as usize * kvdim, 76);
        let prefix_v = rng_fill(l as usize * kvdim, 77);

        // Reference: rotate Q and the new K (cpu_golden::rope, position 0),
        // append the rotated new token to the prefix, then
        // cpu_golden::attention_decode over the full L+1 sequence.
        let mut q_rot = vec![0f32; qdim];
        cpu_golden::rope(
            &q_in,
            &cos,
            &sin,
            &[0i32],
            1,
            hq as usize,
            hd as usize,
            &mut q_rot,
        );
        let mut k_rot = vec![0f32; kvdim];
        cpu_golden::rope(
            &k_in,
            &cos,
            &sin,
            &[0i32],
            1,
            hkv as usize,
            hd as usize,
            &mut k_rot,
        );
        let mut k_all = prefix_k.clone();
        k_all.extend_from_slice(&k_rot);
        let mut v_all = prefix_v.clone();
        v_all.extend_from_slice(&v_in);
        let mut want = vec![0f32; qdim];
        cpu_golden::attention_decode(
            &q_rot,
            &k_all,
            &v_all,
            &mut want,
            (l + 1) as usize,
            hq as usize,
            hkv as usize,
            hd as usize,
            scale,
        );

        let g = lower_decode_attention_standalone(1, hq, hkv, hd, l, scale);
        let got = assemble_result(
            &g,
            &eval_dag(&g, &[&q_in, &k_in, &v_in, &cos, &sin, &prefix_k, &prefix_v]),
        );
        assert_eq!(got, want, "fused decode attention must be bit-exact");
    }
}
