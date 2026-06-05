// SPDX-License-Identifier: Apache-2.0
//! The canonical wavefront **SubtileIR** — region-granular SSA over
//! tensors, the fold of `region.rs` (v2) + `subtile.rs` (v1) into one
//! module.
//!
//! ## What this is
//!
//! Every value lives in a **tensor** (a logical row-major buffer): leaf
//! `sources` (weights / activations / prefix-KV / embed / cos-sin) and one
//! **op-output tensor** per op. A [`SubtileNode`] *writes a region* of
//! one output tensor and *reads regions* of input tensors. Dependencies
//! are derived from **region overlap**: a node that reads region `R` of
//! an op-output tensor `T` depends on every earlier node whose
//! write-region on `T` overlaps `R`. Reads of a leaf source have no
//! dependency (sources are bound at eval time).
//!
//! That is the SSA model that supports **true subtile granularity** on
//! the GPU: `q_proj` split into N-blocks where each block writes a slice
//! of Q, then rope → attention reads the *whole* Q assembled from those
//! slices. The v1 whole-output `Operand::Sub` model couldn't express
//! that — and is gone (it lives only in [`legacy`] for the few v1-only
//! carcasses that have not been deleted yet: [`crate::tape`],
//! [`crate::scheduler`], [`crate::lower::lower`] which are slated for
//! deletion in later staged commits).
//!
//! ## Two-tier validation
//!
//! - **Tier A′ (decomposition equivalence):** [`eval_dag`] equals
//!   `cpu_golden` whole-op output. Bit-exact at `nb >= n` and at every
//!   N-block width that doesn't reorder a per-output reduction; only
//!   token-exact once a reduction is reassociated (split-K).
//! - **Tier A (self-consistency):** scheduled tape replay equals
//!   [`eval_dag`] (lands with the wavefront scheduler).
//!
//! `cpu_golden` (in `ferrite-forward`) is the deterministic f32 substrate
//! the player computes with — not the correctness *oracle* (that is
//! ferrite-metal non-mega at temp=0).

#![allow(dead_code)]

// ── Identifiers & geometry ─────────────────────────────────────────

/// Dense index into a [`SubtileIR::nodes`] vector. The DAG is
/// topologically ordered: every overlap-predecessor of a node has a
/// smaller id (so a single pass over `nodes` is a valid evaluation
/// order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubtileId(pub u32);

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

/// Logical shape of a leaf source buffer (used by the v1
/// `LoweringInput`-side bridge in [`crate::lower`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceShape {
    pub rows: u32,
    pub cols: u32,
}

// ── Sub-operations (shared by both granularities) ──────────────────

/// The sub-operation a node performs. Every variant has a
/// `cpu_golden`-backed host evaluation in [`eval_node`].
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
    /// The K-side `rope_append` for the GPU megakernel: rotate K (NeoX)
    /// **and** write the rotated K + un-rotated V into the paged KV
    /// cache, so the downstream attention reads the new token from the
    /// cache like the oracle non-mega path (Tier-B exact). The host eval
    /// is **rotation only** (identical to [`SubOp::RopeRotate`]); V and
    /// the cache write are GPU-only — V is carried for the schedule edge
    /// and so the serializer can bind it. `layer` names the KV-cache
    /// layer the serializer routes the cache operands to.
    RopeAppend { head_dim: u32, layer: u32 },
    /// Decode attention. `inputs[0]` = Q `[Mq, num_q_heads * head_dim]`;
    /// the remaining inputs are alternating `(K_seg, V_seg)`, each
    /// `[seg_len, num_kv_heads * head_dim]`, concatenated along the KV
    /// axis in input order. The **fused** decode passes the prefix cache
    /// as a read-only segment and the just-rotated new token as the
    /// dataflow segment — so the new K/V is an internal edge, not a
    /// cache round-trip.
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

// ── Tensors & regions ──────────────────────────────────────────────

/// Dense index into [`SubtileIR::tensors`]. Tensors `[0, num_sources)`
/// are leaf sources bound at eval time; the rest are op outputs written
/// by subtile nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

/// Logical shape of a tensor (row-major `[rows, cols]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    pub rows: u32,
    pub cols: u32,
}

impl TensorShape {
    /// The region covering the whole tensor.
    pub fn whole(&self) -> Region {
        Region {
            rows: Range::new(0, self.rows),
            cols: Range::new(0, self.cols),
        }
    }
}

/// A rectangular slice of a tensor — used for both a node's input reads
/// and its single output write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorRegion {
    pub tensor: TensorId,
    pub region: Region,
}

// ── Nodes & graph ──────────────────────────────────────────────────

/// One unit of work: reads `inputs` (regions of tensors), computes its
/// `op`, and writes the result to `output` (a region of one op-output
/// tensor). Produces a dense
/// `[output.region.rows.len, output.region.cols.len]` buffer that is
/// scattered into the output tensor.
#[derive(Clone, Debug)]
pub struct SubtileNode {
    pub id: SubtileId,
    pub op: SubOp,
    pub inputs: Vec<TensorRegion>,
    pub output: TensorRegion,
}

/// The canonical wavefront SubtileIR — a tensor-region SSA dataflow
/// graph. Nodes are topologically ordered: every node that reads an
/// op-output region is preceded by the nodes that write the overlapping
/// region (so a single pass over `nodes` is a valid evaluation order).
#[derive(Clone, Debug)]
pub struct SubtileIR {
    pub tensors: Vec<TensorShape>,
    /// `tensors[0..num_sources]` are leaf sources.
    pub num_sources: u32,
    pub nodes: Vec<SubtileNode>,
    /// The tensor whose buffer is the forward result (logits).
    pub result: TensorId,
}

impl SubtileIR {
    pub fn shape(&self, t: TensorId) -> TensorShape {
        self.tensors[t.0 as usize]
    }
    fn is_source(&self, t: TensorId) -> bool {
        t.0 < self.num_sources
    }
}

// ── Host evaluation ────────────────────────────────────────────────

/// Gather a tensor region into a dense row-major `(buf, rows, cols)`.
fn gather(tr: &TensorRegion, graph: &SubtileIR, bufs: &[Vec<f32>]) -> (Vec<f32>, u32, u32) {
    let shape = graph.shape(tr.tensor);
    let src = &bufs[tr.tensor.0 as usize];
    let (r, c) = (tr.region.rows.len, tr.region.cols.len);
    let mut out = Vec::with_capacity((r * c) as usize);
    for i in 0..r {
        let row = tr.region.rows.start + i;
        let base = ((row * shape.cols) + tr.region.cols.start) as usize;
        out.extend_from_slice(&src[base..base + c as usize]);
    }
    (out, r, c)
}

/// Scatter a dense `[rows, cols]` buffer into `bufs[tensor]` at `region`.
pub fn scatter(
    bufs: &mut [Vec<f32>],
    tensor: TensorId,
    region: Region,
    data: &[f32],
    shape: TensorShape,
) {
    let (r, c) = (region.rows.len as usize, region.cols.len as usize);
    debug_assert_eq!(data.len(), r * c, "scatter data size vs region");
    let dst = &mut bufs[tensor.0 as usize];
    for i in 0..r {
        let row = region.rows.start as usize + i;
        let base = row * shape.cols as usize + region.cols.start as usize;
        dst[base..base + c].copy_from_slice(&data[i * c..(i + 1) * c]);
    }
}

/// Evaluate the SubtileIR on the host. `sources[s]` is the row-major
/// buffer for source tensor `s` (`s < num_sources`), matching
/// `graph.tensors[s]`. Returns the backing buffer of every tensor
/// (indexed by [`TensorId`]); the logits are `bufs[graph.result]`.
pub fn eval_dag(graph: &SubtileIR, sources: &[&[f32]]) -> Vec<Vec<f32>> {
    assert_eq!(
        sources.len(),
        graph.num_sources as usize,
        "source count mismatch"
    );
    let mut bufs: Vec<Vec<f32>> = graph
        .tensors
        .iter()
        .map(|t| vec![0f32; (t.rows * t.cols) as usize])
        .collect();
    for (s, src) in sources.iter().enumerate() {
        assert_eq!(src.len(), bufs[s].len(), "source {s} buffer size mismatch");
        bufs[s].copy_from_slice(src);
    }
    for node in &graph.nodes {
        let out = eval_node(node, graph, &bufs);
        let shape = graph.shape(node.output.tensor);
        scatter(
            &mut bufs,
            node.output.tensor,
            node.output.region,
            &out,
            shape,
        );
    }
    bufs
}

/// Compute one node's dense `[out_rows, out_cols]` output. Per-op
/// arithmetic mirrors `cpu_golden`.
pub fn eval_node(node: &SubtileNode, graph: &SubtileIR, bufs: &[Vec<f32>]) -> Vec<f32> {
    let out_rows = node.output.region.rows.len;
    let out_cols = node.output.region.cols.len;
    match node.op {
        SubOp::MatmulTile => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (w, wr, wc) = gather(&node.inputs[1], graph, bufs);
            assert_eq!(ac, wc, "matmul K mismatch");
            assert_eq!(ar, out_rows, "matmul A rows vs out_rows");
            assert_eq!(wr, out_cols, "matmul W rows vs out_cols");
            let (m, n, k) = (ar as usize, wr as usize, ac as usize);
            let mut out = vec![0f32; m * n];
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
            let len = (out_rows * out_cols) as usize;
            let mut out = vec![0f32; len];
            for inp in &node.inputs {
                let (b, _, _) = gather(inp, graph, bufs);
                assert_eq!(b.len(), len, "reduce operand size mismatch");
                for (o, v) in out.iter_mut().zip(&b) {
                    *o += *v;
                }
            }
            out
        }
        SubOp::Elementwise(kind) => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            assert_eq!((ar, ac), (out_rows, out_cols), "elementwise shape");
            match kind {
                EwKind::Silu => a.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
                EwKind::Mul | EwKind::Add => {
                    let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
                    assert_eq!((br, bc), (ar, ac), "elementwise binary shape");
                    a.iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            if matches!(kind, EwKind::Mul) {
                                x * y
                            } else {
                                x + y
                            }
                        })
                        .collect()
                }
            }
        }
        SubOp::SiluMul => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
            assert_eq!((ar, ac), (out_rows, out_cols), "silu_mul gate shape");
            assert_eq!((br, bc), (ar, ac), "silu_mul up shape");
            a.iter()
                .zip(&b)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect()
        }
        SubOp::RmsNorm { eps } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (wt, _wr, wc) = gather(&node.inputs[1], graph, bufs);
            assert_eq!((xr, xc), (out_rows, out_cols), "rmsnorm shape");
            assert_eq!(wc, out_cols, "rmsnorm weight width");
            let (m, d) = (xr as usize, xc as usize);
            let mut out = vec![0f32; m * d];
            for i in 0..m {
                let row = &x[i * d..(i + 1) * d];
                let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
                let inv_rms = 1.0 / (sum_sq / d as f32 + eps).sqrt();
                for j in 0..d {
                    out[i * d + j] = row[j] * inv_rms * wt[j];
                }
            }
            out
        }
        // RopeAppend's host eval is rotation only (identical to RopeRotate);
        // its V input + the paged-cache write are GPU-only.
        SubOp::RopeRotate { head_dim } | SubOp::RopeAppend { head_dim, .. } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (cos, _, cc) = gather(&node.inputs[1], graph, bufs);
            let (sin, _, sc) = gather(&node.inputs[2], graph, bufs);
            assert_eq!((xr, xc), (out_rows, out_cols), "rope shape");
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
            // Head-block aware: this node computes a contiguous q-head range,
            // derived from the OUTPUT region (its column slice), and reads the
            // matching kv-head range — derived from the K input region. The
            // SubOp keeps the GLOBAL head counts so the GQA ratio is exact; the
            // block's own counts come from the slice widths. Each q-head's
            // attention is independent ⇒ bit-exact vs the whole op.
            let hd = head_dim as usize;
            let gqa = (num_q_heads / num_kv_heads.max(1)) as usize;
            let (q, qr, qc) = gather(&node.inputs[0], graph, bufs);
            let mq = qr as usize;
            let qh_count = qc as usize / hd; // q-heads in this block
            let qh_start = node.output.region.cols.start as usize / hd; // global first q-head
            assert_eq!(
                qc as usize,
                qh_count * hd,
                "attn Q width is a head multiple"
            );
            assert!(node.inputs.len() >= 3, "attn needs Q + >=1 (K,V) segment");
            assert_eq!(node.inputs.len() % 2, 1, "attn inputs = Q + (K,V) pairs");
            // kv-head offset of this block (from the first K segment's column slice).
            let kvh_start = node.inputs[1].region.cols.start as usize / hd;
            // Concatenate K/V segments along sequence; each segment spans this
            // block's kv-heads (kv_count * hd wide).
            let mut k_all: Vec<f32> = Vec::new();
            let mut v_all: Vec<f32> = Vec::new();
            let mut kv_count = 0usize;
            let mut i = 1;
            while i < node.inputs.len() {
                let (k, kr, kc) = gather(&node.inputs[i], graph, bufs);
                let (v, vr, vc) = gather(&node.inputs[i + 1], graph, bufs);
                assert_eq!((vr, vc), (kr, kc), "attn V seg shape");
                kv_count = kc as usize / hd;
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(&v);
                i += 2;
            }
            assert!(kv_count >= 1, "attn K seg has at least one kv-head");
            let seg_w = kv_count * hd;
            let seq_len = k_all.len() / seg_w;
            let mut out = vec![0f32; mq * qh_count * hd];
            for qi in 0..mq {
                for hl in 0..qh_count {
                    // local q-head hl → global q-head → global kv-head → local kv.
                    let global_kv = (qh_start + hl) / gqa;
                    let local_kv = global_kv - kvh_start;
                    let q_off = qi * qh_count * hd + hl * hd;
                    let mut scores = vec![0f32; seq_len];
                    for (s, score) in scores.iter_mut().enumerate() {
                        let k_off = s * seg_w + local_kv * hd;
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
                        for (s, &wgt) in scores.iter().enumerate() {
                            val += wgt * v_all[s * seg_w + local_kv * hd + d];
                        }
                        out[q_off + d] = val;
                    }
                }
            }
            out
        }
    }
}

/// The forward result buffer (logits) — `bufs[graph.result]`.
pub fn result_buffer<'a>(graph: &SubtileIR, bufs: &'a [Vec<f32>]) -> &'a [f32] {
    &bufs[graph.result.0 as usize]
}

// ── Dependencies (region overlap) ──────────────────────────────────

fn ranges_overlap(a: Range, b: Range) -> bool {
    a.start < b.end() && b.start < a.end()
}

fn regions_overlap(a: Region, b: Region) -> bool {
    ranges_overlap(a.rows, b.rows) && ranges_overlap(a.cols, b.cols)
}

/// Predecessor node ids for each node: the earlier nodes whose write
/// overlaps one of this node's input reads on the same op-output tensor.
/// Reads of leaf sources contribute no dependency. This is the edge set
/// the wavefront scheduler turns into cross-worker `Wait`/`Signal`.
pub fn predecessors(graph: &SubtileIR) -> Vec<Vec<SubtileId>> {
    // writers[t] = (node_id, out_region) for each op-output tensor, in id order.
    let mut writers: Vec<Vec<(u32, Region)>> = vec![Vec::new(); graph.tensors.len()];
    let mut preds: Vec<Vec<SubtileId>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let mut p: Vec<SubtileId> = Vec::new();
        for inp in &node.inputs {
            if graph.is_source(inp.tensor) {
                continue;
            }
            for (wid, wreg) in &writers[inp.tensor.0 as usize] {
                if regions_overlap(*wreg, inp.region) && !p.contains(&SubtileId(*wid)) {
                    p.push(SubtileId(*wid));
                }
            }
        }
        p.sort();
        preds.push(p);
        writers[node.output.tensor.0 as usize].push((node.id.0, node.output.region));
    }
    preds
}

// ── Structural validation ──────────────────────────────────────────

/// Check the graph's invariants without evaluating: dense ids, in-range
/// tensors, in-bounds regions, op-output (not source) write targets,
/// op arity, and that every op-output read is covered by writers with a
/// strictly smaller id (acyclic + assembled-before-read). Returns the
/// node count on success.
pub fn validate(graph: &SubtileIR) -> Result<usize, String> {
    let n_tensors = graph.tensors.len() as u32;
    if graph.num_sources > n_tensors {
        return Err(format!(
            "num_sources {} exceeds tensor count {n_tensors}",
            graph.num_sources
        ));
    }
    // Track written sub-regions per op-output tensor to check coverage.
    let mut writes: Vec<Vec<(u32, Region)>> = vec![Vec::new(); graph.tensors.len()];
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
            // E.12: RopeAppend takes [K, cos, sin, V, K_cache, V_cache]
            // — the K_cache / V_cache are per-layer PrefixK / PrefixV
            // sources used as TMA-store destinations for the new
            // decode token's K/V.
            SubOp::RopeAppend { .. } => arity == 6,
            SubOp::AttnDecode { .. } => arity >= 3 && arity % 2 == 1,
        };
        if !arity_ok {
            return Err(format!("node {i} op {:?} bad arity {arity}", node.op));
        }
        // Output must target an op-output tensor, in bounds.
        let ot = node.output.tensor;
        if ot.0 >= n_tensors {
            return Err(format!("node {i} writes tensor {} out of range", ot.0));
        }
        if graph.is_source(ot) {
            return Err(format!("node {i} writes leaf source tensor {}", ot.0));
        }
        let oshape = graph.shape(ot);
        if node.output.region.rows.end() > oshape.rows
            || node.output.region.cols.end() > oshape.cols
        {
            return Err(format!("node {i} output region exceeds tensor {}", ot.0));
        }
        // Inputs in bounds; op-output reads must be covered by prior writers.
        for (a, inp) in node.inputs.iter().enumerate() {
            if inp.tensor.0 >= n_tensors {
                return Err(format!(
                    "node {i} input {a} tensor {} out of range",
                    inp.tensor.0
                ));
            }
            let ishape = graph.shape(inp.tensor);
            if inp.region.rows.end() > ishape.rows || inp.region.cols.end() > ishape.cols {
                return Err(format!(
                    "node {i} input {a} region exceeds tensor {}",
                    inp.tensor.0
                ));
            }
            if !graph.is_source(inp.tensor) {
                // Some earlier write must overlap (cheap acyclicity / use-before-def check).
                let has = writes[inp.tensor.0 as usize]
                    .iter()
                    .any(|(wid, wreg)| *wid < node.id.0 && regions_overlap(*wreg, inp.region));
                if !has {
                    return Err(format!(
                        "node {i} input {a} reads tensor {} region with no prior writer",
                        inp.tensor.0
                    ));
                }
            }
        }
        writes[ot.0 as usize].push((node.id.0, node.output.region));
    }
    if graph.result.0 >= n_tensors {
        return Err(format!("result tensor {} out of range", graph.result.0));
    }
    if graph.is_source(graph.result) {
        return Err("result is a leaf source, not an op output".into());
    }
    Ok(graph.nodes.len())
}

// ── LoweringInput → SubtileIR ──────────────────────────────────────

/// Tile `[0, total)` into contiguous blocks of width `block` (last block
/// may be shorter). `block >= total` yields a single whole block.
pub fn n_blocks(total: u32, block: u32) -> Vec<Range> {
    assert!(block >= 1, "block width must be >= 1");
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let len = block.min(total - start);
        out.push(Range::new(start, len));
        start += len;
    }
    if out.is_empty() {
        out.push(Range::new(0, 0)); // total==0 (degenerate); keep one empty block
    }
    out
}

/// Tile `[0, total)` into **head-aligned** blocks of width ~`nb`, snapped
/// down to a whole number of `head_dim`-wide heads (at least one head). Used
/// for rope/attention so every block is a clean set of heads — the q→rope→
/// attn chain partitions on head boundaries (and o_proj split-Ks on them).
pub fn head_blocks(total: u32, nb: u32, head_dim: u32) -> Vec<Range> {
    let hd = head_dim.max(1);
    let heads_per_block = (nb / hd).max(1);
    n_blocks(total, heads_per_block * hd)
}

/// Out-columns of an op (mirrors `crate::lower`): GEMM → n, attention →
/// `num_q_heads * head_dim`, everything else preserves input-0 width.
pub(crate) fn op_out_cols(op: crate::lower::LoweredOp, in0_cols: u32) -> u32 {
    use crate::lower::LoweredOp;
    match op {
        LoweredOp::Gemm { n, .. } => n,
        LoweredOp::AttnDecode {
            num_q_heads,
            head_dim,
            ..
        } => num_q_heads * head_dim,
        LoweredOp::RmsNorm { .. }
        | LoweredOp::Silu
        | LoweredOp::Mul
        | LoweredOp::SiluMul
        | LoweredOp::Add
        | LoweredOp::RopeRotate { .. }
        | LoweredOp::RopeAppend { .. } => in0_cols,
    }
}

/// Lower a flat [`crate::lower::LoweringInput`] to a SubtileIR,
/// **N-block tiling every GEMM** by `nb` (output columns split into
/// `ceil(n/nb)` MatmulTile subtiles, each writing a disjoint column
/// slice of the op's output tensor — no reduce, bit-exact). All other
/// ops stay whole (one subtile writing the whole output tensor).
/// `nb >= n` ⇒ a single block (coarse, equivalent to v1). Source tensors
/// mirror `input.sources`; op-output tensor `i` is
/// `TensorId(num_sources + i)`.
pub fn lower_region(input: &crate::lower::LoweringInput, nb: u32) -> SubtileIR {
    use crate::lower::{InputRef, LoweredOp};
    let num_sources = input.sources.len() as u32;
    let mut tensors: Vec<TensorShape> = input
        .sources
        .iter()
        .map(|s| TensorShape {
            rows: s.rows,
            cols: s.cols,
        })
        .collect();
    let mut op_tensor: Vec<TensorId> = Vec::with_capacity(input.ops.len());
    let mut op_cols: Vec<u32> = Vec::with_capacity(input.ops.len());
    let mut nodes: Vec<SubtileNode> = Vec::new();

    // Resolve an InputRef to (tensor id, shape).
    let resolve = |r: InputRef,
                   op_tensor: &[TensorId],
                   op_cols: &[u32],
                   tensors: &[TensorShape]|
     -> (TensorId, u32, u32) {
        match r {
            InputRef::Op(j) => {
                let t = op_tensor[j];
                (t, tensors[t.0 as usize].rows, op_cols[j])
            }
            InputRef::Ext(e) => {
                let s = tensors[e];
                (TensorId(e as u32), s.rows, s.cols)
            }
        }
    };
    let whole = |t: TensorId, tensors: &[TensorShape]| -> TensorRegion {
        TensorRegion {
            tensor: t,
            region: tensors[t.0 as usize].whole(),
        }
    };

    for desc in &input.ops {
        let m = desc.m;
        let (in0_t, _in0_rows, in0_cols) = resolve(desc.inputs[0], &op_tensor, &op_cols, &tensors);
        let out_cols = op_out_cols(desc.op, in0_cols);
        let out_t = TensorId(tensors.len() as u32);
        tensors.push(TensorShape {
            rows: m,
            cols: out_cols,
        });

        match desc.op {
            LoweredOp::Gemm { n, k } => {
                assert_eq!(in0_cols, k, "gemm activation cols must equal k");
                let (w_t, _wr, _wc) = resolve(desc.inputs[1], &op_tensor, &op_cols, &tensors);
                let act = TensorRegion {
                    tensor: in0_t,
                    region: Region {
                        rows: Range::new(0, m),
                        cols: Range::new(0, k),
                    },
                };
                for blk in n_blocks(n, nb) {
                    let id = SubtileId(nodes.len() as u32);
                    nodes.push(SubtileNode {
                        id,
                        op: SubOp::MatmulTile,
                        inputs: vec![
                            act,
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
                }
            }
            other => {
                let subop = match other {
                    LoweredOp::RmsNorm { eps } => SubOp::RmsNorm { eps },
                    LoweredOp::Silu => SubOp::Elementwise(EwKind::Silu),
                    LoweredOp::Mul => SubOp::Elementwise(EwKind::Mul),
                    LoweredOp::SiluMul => SubOp::SiluMul,
                    LoweredOp::Add => SubOp::Elementwise(EwKind::Add),
                    LoweredOp::RopeRotate { head_dim } => SubOp::RopeRotate { head_dim },
                    LoweredOp::RopeAppend { head_dim, layer } => {
                        SubOp::RopeAppend { head_dim, layer }
                    }
                    LoweredOp::AttnDecode {
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                    } => SubOp::AttnDecode {
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                    },
                    LoweredOp::Gemm { .. } => unreachable!("gemm handled above"),
                };
                // A pure elementwise op (silu/mul/add/silu·mul) is tiled by the
                // output column slice like the GEMM N-blocks, so the scheduler
                // can spread it across workers. rope / attn (head structure) and
                // rmsnorm (RMS reduction) stay WHOLE here — this is the bit-exact,
                // GPU-correct reference lowering + the live per-op-schedule path.
                // The head-tiling, split-K all-reduce and replication of the
                // tensor-parallel partition live in
                // [`crate::partition::lower_partitioned`], kept separate because
                // split-K reassociates (breaks this fn's bit-exact contract) and
                // because the partition needs the new GPU emit/player arms.
                let elementwise = matches!(
                    other,
                    LoweredOp::Silu | LoweredOp::Mul | LoweredOp::Add | LoweredOp::SiluMul
                );
                let blocks = if elementwise {
                    n_blocks(out_cols, nb)
                } else {
                    vec![Range::new(0, out_cols)]
                };
                for blk in blocks {
                    let inputs: Vec<TensorRegion> = desc
                        .inputs
                        .iter()
                        .map(|r| {
                            let (t, _, _) = resolve(*r, &op_tensor, &op_cols, &tensors);
                            if elementwise {
                                TensorRegion {
                                    tensor: t,
                                    region: Region {
                                        rows: Range::new(0, m),
                                        cols: blk,
                                    },
                                }
                            } else {
                                whole(t, &tensors)
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
                }
            }
        }
        op_tensor.push(out_t);
        op_cols.push(out_cols);
    }

    SubtileIR {
        tensors,
        num_sources,
        nodes,
        result: op_tensor[input.result],
    }
}

// ── Tests (canonical region IR) ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{InputRef, LoweredOp, OpDesc};
    use ferrite_forward::cpu_golden;

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

    /// N-block-tiled GEMM is bit-exact vs `cpu_golden::gemm` (disjoint
    /// output columns, same per-output reduction order). Built by hand at
    /// the IR level: source 0 = act [m,k], source 1 = W [n,k], one
    /// op-output tensor [m,n] written by `ceil(n/nb)` MatmulTile blocks.
    #[test]
    fn gemm_nblock_bit_exact_vs_cpu_golden() {
        let (m, n, k) = (1u32, 130, 257);
        let act = rng_fill((m * k) as usize, 1);
        let w = rng_fill((n * k) as usize, 2);
        let mut want = vec![0f32; (m * n) as usize];
        cpu_golden::gemm(&act, &w, &mut want, m as usize, k as usize, n as usize);

        for nb in [16u32, 48, 64, 130, 256] {
            let tensors = vec![
                TensorShape { rows: m, cols: k }, // 0: act
                TensorShape { rows: n, cols: k }, // 1: W
                TensorShape { rows: m, cols: n }, // 2: out
            ];
            let out_t = TensorId(2);
            let mut nodes = Vec::new();
            let mut start = 0u32;
            while start < n {
                let len = nb.min(n - start);
                let id = SubtileId(nodes.len() as u32);
                nodes.push(SubtileNode {
                    id,
                    op: SubOp::MatmulTile,
                    inputs: vec![
                        TensorRegion {
                            tensor: TensorId(0),
                            region: Region {
                                rows: Range::new(0, m),
                                cols: Range::new(0, k),
                            },
                        },
                        TensorRegion {
                            tensor: TensorId(1),
                            region: Region {
                                rows: Range::new(start, len),
                                cols: Range::new(0, k),
                            },
                        },
                    ],
                    output: TensorRegion {
                        tensor: out_t,
                        region: Region {
                            rows: Range::new(0, m),
                            cols: Range::new(start, len),
                        },
                    },
                });
                start += len;
            }
            let g = SubtileIR {
                tensors,
                num_sources: 2,
                nodes,
                result: out_t,
            };
            assert!(validate(&g).is_ok(), "valid nb={nb}");
            let bufs = eval_dag(&g, &[&act, &w]);
            assert_eq!(result_buffer(&g, &bufs), &want[..], "nb={nb} bit-exact");
        }
    }

    /// Build a whole Llama-style decode layer as a `LoweringInput` and
    /// lower it with [`lower_region`] at several N-block widths; each must
    /// be bit-exact vs the `cpu_golden` composition.
    #[test]
    fn full_decode_layer_nblock_bit_exact() {
        let (h, hd, hq, hkv, i, l) = (16u32, 4u32, 4u32, 2u32, 32u32, 3u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let eps = 1e-5f32;
        let scale = 1.0 / (hd as f32).sqrt();

        let res_in = rng_fill(h as usize, 101);
        let in_ln = rng_fill(h as usize, 102);
        let wq = rng_fill((qdim * h) as usize, 103);
        let wk = rng_fill((kvdim * h) as usize, 104);
        let wv = rng_fill((kvdim * h) as usize, 105);
        let cos = rng_fill(hd as usize, 106);
        let sin = rng_fill(hd as usize, 107);
        let prefix_k = rng_fill((l * kvdim) as usize, 108);
        let prefix_v = rng_fill((l * kvdim) as usize, 109);
        let wo = rng_fill((h * qdim) as usize, 110);
        let post_ln = rng_fill(h as usize, 111);
        let wgate = rng_fill((i * h) as usize, 112);
        let wup = rng_fill((i * h) as usize, 113);
        let wdown = rng_fill((h * i) as usize, 114);

        // Reference via cpu_golden.
        let (hs, hds, hqs, hkvs, is, qd, kvd) = (
            h as usize,
            hd as usize,
            hq as usize,
            hkv as usize,
            i as usize,
            qdim as usize,
            kvdim as usize,
        );
        let mut xn = vec![0f32; hs];
        cpu_golden::rmsnorm(&res_in, &in_ln, &mut xn, eps);
        let mut q = vec![0f32; qd];
        cpu_golden::gemm(&xn, &wq, &mut q, 1, hs, qd);
        let mut k = vec![0f32; kvd];
        cpu_golden::gemm(&xn, &wk, &mut k, 1, hs, kvd);
        let mut v = vec![0f32; kvd];
        cpu_golden::gemm(&xn, &wv, &mut v, 1, hs, kvd);
        let mut q_rot = vec![0f32; qd];
        cpu_golden::rope(&q, &cos, &sin, &[0i32], 1, hqs, hds, &mut q_rot);
        let mut k_rot = vec![0f32; kvd];
        cpu_golden::rope(&k, &cos, &sin, &[0i32], 1, hkvs, hds, &mut k_rot);
        let mut k_all = prefix_k.clone();
        k_all.extend_from_slice(&k_rot);
        let mut v_all = prefix_v.clone();
        v_all.extend_from_slice(&v);
        let mut attn = vec![0f32; qd];
        cpu_golden::attention_decode(
            &q_rot,
            &k_all,
            &v_all,
            &mut attn,
            (l + 1) as usize,
            hqs,
            hkvs,
            hds,
            scale,
        );
        let mut o = vec![0f32; hs];
        cpu_golden::gemm(&attn, &wo, &mut o, 1, qd, hs);
        let mut res_mid = vec![0f32; hs];
        cpu_golden::add(&o, &res_in, &mut res_mid);
        let mut xn2 = vec![0f32; hs];
        cpu_golden::rmsnorm(&res_mid, &post_ln, &mut xn2, eps);
        let mut gate = vec![0f32; is];
        cpu_golden::gemm(&xn2, &wgate, &mut gate, 1, hs, is);
        let mut up = vec![0f32; is];
        cpu_golden::gemm(&xn2, &wup, &mut up, 1, hs, is);
        let mut act = vec![0f32; is];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut act);
        let mut down = vec![0f32; hs];
        cpu_golden::gemm(&act, &wdown, &mut down, 1, is, hs);
        let mut want = vec![0f32; hs];
        cpu_golden::add(&down, &res_mid, &mut want);

        // Same layer as a LoweringInput (sources 0..=13).
        let ss = |rows: u32, cols: u32| SourceShape { rows, cols };
        let input = crate::lower::LoweringInput {
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
                    op: LoweredOp::Gemm { n: qdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(3)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
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
                    op: LoweredOp::Gemm { n: h, k: qdim },
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
                    op: LoweredOp::Gemm { n: i, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(11)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(10)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: i, k: h },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(12)],
                },
                OpDesc {
                    op: LoweredOp::Mul,
                    m: 1,
                    inputs: vec![InputRef::Op(11), InputRef::Op(12)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: h, k: i },
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
        let srcs: Vec<&[f32]> = vec![
            &res_in, &in_ln, &wq, &wk, &wv, &cos, &sin, &prefix_k, &prefix_v, &wo, &post_ln,
            &wgate, &wup, &wdown,
        ];

        for nb in [4u32, 8, 1000] {
            let g = lower_region(&input, nb);
            assert!(validate(&g).is_ok(), "valid layer nb={nb}");
            let bufs = eval_dag(&g, &srcs);
            assert_eq!(
                result_buffer(&g, &bufs),
                &want[..],
                "decode layer nb={nb} bit-exact"
            );
        }
    }

    /// A consumer reading a whole N-block-tiled output depends on EVERY
    /// block; a reader of a leaf source has no dependency.
    #[test]
    fn tiled_elementwise_depends_on_matching_block() {
        let (m, n, k) = (1u32, 6u32, 8u32);
        let input = crate::lower::LoweringInput {
            sources: vec![
                SourceShape { rows: m, cols: k },
                SourceShape { rows: n, cols: k },
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n, k },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            result: 1,
        };
        let g = lower_region(&input, 2);
        let preds = predecessors(&g);
        // 3 matmul blocks (0,1,2) + 3 silu tiles (3,4,5).
        assert_eq!(g.nodes.len(), 6);
        assert!(
            preds[0].is_empty() && preds[1].is_empty() && preds[2].is_empty(),
            "matmul blocks read only sources"
        );
        assert_eq!(preds[3], vec![SubtileId(0)], "silu tile 0 ← matmul block 0");
        assert_eq!(preds[4], vec![SubtileId(1)], "silu tile 1 ← matmul block 1");
        assert_eq!(preds[5], vec![SubtileId(2)], "silu tile 2 ← matmul block 2");
    }

    #[test]
    fn validate_rejects_uncovered_read() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // 0 source
            TensorShape { rows: 1, cols: 4 }, // 1 op output (never written)
        ];
        let bad = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![SubtileNode {
                id: SubtileId(0),
                op: SubOp::Elementwise(EwKind::Silu),
                inputs: vec![TensorRegion {
                    tensor: TensorId(1), // reads op-output 1, which no prior node wrote
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                }],
                output: TensorRegion {
                    tensor: TensorId(1),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
            }],
            result: TensorId(1),
        };
        assert!(validate(&bad).is_err(), "use-before-def must be rejected");
    }
}
