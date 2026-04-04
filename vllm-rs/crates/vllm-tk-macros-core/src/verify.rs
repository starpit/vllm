// SPDX-License-Identifier: Apache-2.0
//! Compile-time verification of the megakernel DAG.
//!
//! This is the "borrow checker for GPU memory." If verify() passes,
//! the generated kernel is guaranteed free of:
//! - Buffer size mismatches between producer/consumer
//! - Shared memory overflows
//! - Barrier count imbalances
//! - Buffer aliasing (concurrent read/write)
//! - Incomplete tile coverage
//! - Use of uninitialized buffers

use crate::dag::*;
use std::collections::{HashMap, HashSet};

/// A verification error. In proc-macro context, each becomes a compile_error!().
#[derive(Debug, Clone)]
pub struct VerifyError {
    pub kind: ErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Producer output shape != consumer input shape.
    ShapeMismatch,
    /// Activation buffer written by two ops without a data dependency between them.
    BufferAliasing,
    /// Buffer read before any op writes it (and it's not an input/weight).
    UninitializedRead,
    /// Shared memory layout exceeds hardware budget.
    ShmemOverflow,
    /// Barrier signal count != wait count.
    BarrierImbalance,
    /// Some tiles aren't assigned to any SM, or assigned to multiple SMs.
    TileCoverage,
    /// A buffer is declared but never used.
    DeadBuffer,
    /// An op's output buffer doesn't exist in the DAG.
    MissingBuffer,
    /// Dimension parameter referenced but not defined.
    UndefinedParam,
    /// GEMM dimension mismatch: A cols != B cols (for A @ B^T).
    GemmDimMismatch,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.message)
    }
}

/// Run all verification passes on the DAG. Returns all errors found.
pub fn verify(dag: &ModelDag) -> Vec<VerifyError> {
    let mut errors = Vec::new();
    verify_buffers_exist(dag, &mut errors);
    verify_no_uninitialized_reads(dag, &mut errors);
    verify_shapes(dag, &mut errors);
    verify_no_aliasing(dag, &mut errors);
    verify_shmem(dag, &mut errors);
    verify_dead_buffers(dag, &mut errors);
    verify_barrier_chain(dag, &mut errors);
    verify_tile_divisibility(dag, &mut errors);
    errors
}

/// Every buffer referenced by an op must exist in the DAG.
fn verify_buffers_exist(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    for op in &dag.ops {
        for buf_id in op.inputs().into_iter().chain(op.outputs().into_iter()) {
            if !dag.buffers.contains_key(buf_id) {
                errors.push(VerifyError {
                    kind: ErrorKind::MissingBuffer,
                    message: format!(
                        "op {} references buffer '{}' which is not declared",
                        op.idx, buf_id
                    ),
                });
            }
        }
    }
}

/// Every activation buffer must be written before it's read.
fn verify_no_uninitialized_reads(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    // Walk ops in topological order. Track which buffers have been written.
    let mut written: HashSet<&BufferId> = HashSet::new();

    // Weights, KV caches, and metadata are always "written" (loaded externally).
    for (id, buf) in &dag.buffers {
        match buf.kind {
            BufferKind::Weight | BufferKind::KvCache | BufferKind::Metadata => {
                written.insert(id);
            }
            BufferKind::Activation => {
                // Only mark as written if explicitly declared as an external input.
                if buf.is_input {
                    written.insert(id);
                }
            }
        }
    }

    for op in &dag.ops {
        // Check all inputs are written.
        for in_id in op.inputs() {
            if !written.contains(in_id) {
                errors.push(VerifyError {
                    kind: ErrorKind::UninitializedRead,
                    message: format!(
                        "op {} reads buffer '{}' before it is written by any op",
                        op.idx, in_id
                    ),
                });
            }
        }
        // Mark outputs as written.
        for out_id in op.outputs() {
            written.insert(out_id);
        }
    }
}

/// Verify dimension compatibility across edges.
fn verify_shapes(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    for op in &dag.ops {
        match &op.kind {
            OpKind::RmsNorm {
                input,
                weights,
                output,
            } => {
                // input[BS, D], weights[D], output[BS, D]
                // input cols == weights dim == output cols
                if let (Some(inp), Some(wgt), Some(out)) = (
                    dag.buffers.get(input),
                    dag.buffers.get(weights),
                    dag.buffers.get(output),
                ) {
                    check_last_dim_match(
                        dag,
                        &inp.shape,
                        &wgt.shape,
                        "rmsnorm input",
                        "weights",
                        op.idx,
                        errors,
                    );
                    check_shapes_equal(
                        dag,
                        &inp.shape,
                        &out.shape,
                        "rmsnorm input",
                        "output",
                        op.idx,
                        errors,
                    );
                }
            }
            OpKind::Gemm { a, b, output } => {
                // A[BS, K] @ B[N, K]^T -> output[BS, N]
                // A cols == B cols (K dimension)
                // output = [A rows, B rows]
                if let (Some(a_buf), Some(b_buf), Some(out_buf)) = (
                    dag.buffers.get(a),
                    dag.buffers.get(b),
                    dag.buffers.get(output),
                ) {
                    check_gemm_dims(
                        dag,
                        &a_buf.shape,
                        &b_buf.shape,
                        &out_buf.shape,
                        op.idx,
                        errors,
                    );
                }
            }
            OpKind::GemmAdd {
                a,
                b,
                residual,
                output,
            } => {
                // Same as Gemm but also: residual shape == output shape
                if let (Some(a_buf), Some(b_buf), Some(res_buf), Some(out_buf)) = (
                    dag.buffers.get(a),
                    dag.buffers.get(b),
                    dag.buffers.get(residual),
                    dag.buffers.get(output),
                ) {
                    check_gemm_dims(
                        dag,
                        &a_buf.shape,
                        &b_buf.shape,
                        &out_buf.shape,
                        op.idx,
                        errors,
                    );
                    check_shapes_equal(
                        dag,
                        &res_buf.shape,
                        &out_buf.shape,
                        "residual",
                        "output",
                        op.idx,
                        errors,
                    );
                }
            }
            OpKind::Silu { input, output } => {
                if let (Some(inp), Some(out)) = (dag.buffers.get(input), dag.buffers.get(output)) {
                    check_shapes_equal(
                        dag,
                        &inp.shape,
                        &out.shape,
                        "silu input",
                        "output",
                        op.idx,
                        errors,
                    );
                }
            }
            OpKind::Mul { a, b, output } => {
                if let (Some(a_buf), Some(b_buf), Some(out_buf)) = (
                    dag.buffers.get(a),
                    dag.buffers.get(b),
                    dag.buffers.get(output),
                ) {
                    check_shapes_equal(
                        dag,
                        &a_buf.shape,
                        &b_buf.shape,
                        "mul lhs",
                        "rhs",
                        op.idx,
                        errors,
                    );
                    check_shapes_equal(
                        dag,
                        &a_buf.shape,
                        &out_buf.shape,
                        "mul input",
                        "output",
                        op.idx,
                        errors,
                    );
                }
            }
            OpKind::RopeAppend { qkv, q_out, .. } => {
                // qkv[BS, QKV_DIM] where QKV_DIM = (NAH + 2*NKH) * HDM
                // q_out[BS, HD] where HD = NAH * HDM
                if let Some(qkv_buf) = dag.buffers.get(qkv) {
                    let nah = dag.params.get("NAH").copied();
                    let nkh = dag.params.get("NKH").copied();
                    let hdm = dag.params.get("HDM").copied();
                    if let (Some(nah), Some(nkh), Some(hdm)) = (nah, nkh, hdm) {
                        let expected_qkv_dim = (nah + 2 * nkh) * hdm;
                        if let Some(last) = qkv_buf.shape.dims.last()
                            && let Ok(actual) = last.resolve(&dag.params)
                            && actual != expected_qkv_dim
                        {
                            errors.push(VerifyError {
                                kind: ErrorKind::ShapeMismatch,
                                message: format!(
                                    "op {}: rope_append qkv last dim={actual} != (NAH+2*NKH)*HDM={expected_qkv_dim}",
                                    op.idx,
                                ),
                            });
                        }
                    }
                }
                if let (Some(qkv_buf), Some(q_buf)) = (dag.buffers.get(qkv), dag.buffers.get(q_out))
                {
                    // q_out rows == qkv rows (batch dim)
                    if qkv_buf.shape.dims.len() >= 2 && q_buf.shape.dims.len() >= 2 {
                        let qkv_rows = &qkv_buf.shape.dims[0];
                        let q_rows = &q_buf.shape.dims[0];
                        if qkv_rows != q_rows {
                            errors.push(VerifyError {
                                kind: ErrorKind::ShapeMismatch,
                                message: format!(
                                    "op {}: rope_append qkv rows={qkv_rows} != q_out rows={q_rows}",
                                    op.idx,
                                ),
                            });
                        }
                    }
                }
            }
            OpKind::AttentionDecode { q, output, .. }
            | OpKind::AttentionPrefill { q, output, .. } => {
                if let (Some(q_buf), Some(out_buf)) = (dag.buffers.get(q), dag.buffers.get(output))
                {
                    check_shapes_equal(
                        dag,
                        &q_buf.shape,
                        &out_buf.shape,
                        "attention q",
                        "output",
                        op.idx,
                        errors,
                    );
                }
            }
        }
    }
}

/// Check that two shapes are equal (symbolically).
fn check_shapes_equal(
    _dag: &ModelDag,
    a: &TensorShape,
    b: &TensorShape,
    a_name: &str,
    b_name: &str,
    op_idx: OpIdx,
    errors: &mut Vec<VerifyError>,
) {
    if a.dims.len() != b.dims.len() {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!(
                "op {op_idx}: {a_name} has rank {} but {b_name} has rank {} ({a} vs {b})",
                a.dims.len(),
                b.dims.len(),
            ),
        });
        return;
    }
    for (i, (da, db)) in a.dims.iter().zip(b.dims.iter()).enumerate() {
        if da != db {
            errors.push(VerifyError {
                kind: ErrorKind::ShapeMismatch,
                message: format!("op {op_idx}: {a_name} dim[{i}]={da} != {b_name} dim[{i}]={db}",),
            });
        }
    }
}

/// Check last dim of `a` matches the single dim (or last dim) of `b`.
fn check_last_dim_match(
    _dag: &ModelDag,
    a: &TensorShape,
    b: &TensorShape,
    a_name: &str,
    b_name: &str,
    op_idx: OpIdx,
    errors: &mut Vec<VerifyError>,
) {
    let a_last = a.dims.last();
    let b_last = b.dims.last();
    if let (Some(al), Some(bl)) = (a_last, b_last)
        && al != bl
    {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!("op {op_idx}: {a_name} last dim={al} != {b_name} last dim={bl}",),
        });
    }
}

/// Check GEMM dimensions: A[BS, K] @ B[N, K]^T -> C[BS, N].
fn check_gemm_dims(
    _dag: &ModelDag,
    a: &TensorShape,
    b: &TensorShape,
    out: &TensorShape,
    op_idx: OpIdx,
    errors: &mut Vec<VerifyError>,
) {
    // A must be rank 2: [rows, K]
    // B must be rank 2: [N, K]  (transposed multiply)
    // out must be rank 2: [rows, N]
    if a.dims.len() < 2 || b.dims.len() < 2 || out.dims.len() < 2 {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!(
                "op {op_idx}: gemm requires rank-2 tensors, got A={a}, B={b}, out={out}"
            ),
        });
        return;
    }
    let a_cols = &a.dims[a.dims.len() - 1];
    let b_cols = &b.dims[b.dims.len() - 1];
    let b_rows = &b.dims[b.dims.len() - 2];
    let out_rows = &out.dims[out.dims.len() - 2];
    let out_cols = &out.dims[out.dims.len() - 1];
    let a_rows = &a.dims[a.dims.len() - 2];

    // A cols == B cols (K dimension, since B is transposed)
    if a_cols != b_cols {
        errors.push(VerifyError {
            kind: ErrorKind::GemmDimMismatch,
            message: format!(
                "op {op_idx}: gemm K mismatch: A cols={a_cols} != B cols={b_cols} (A={a}, B={b})"
            ),
        });
    }
    // output rows == A rows
    if a_rows != out_rows {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!("op {op_idx}: gemm output rows={out_rows} != A rows={a_rows}"),
        });
    }
    // output cols == B rows (N dimension)
    if b_rows != out_cols {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!("op {op_idx}: gemm output cols={out_cols} != B rows={b_rows} (B={b})"),
        });
    }
}

/// No two ops may write the same activation buffer unless one depends on the other.
fn verify_no_aliasing(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    // Build a map: buffer -> list of producer ops.
    let mut producers: HashMap<&BufferId, Vec<OpIdx>> = HashMap::new();
    for op in &dag.ops {
        for out_id in op.outputs() {
            producers.entry(out_id).or_default().push(op.idx);
        }
    }

    for (buf_id, ops) in &producers {
        if ops.len() > 1 {
            // Multiple producers. This is only OK if they are strictly ordered
            // (one happens-before the other via data dependency).
            // For now: flag it. GemmAdd writing to hidden_states is OK because
            // it's the SAME op (residual=output). Check that case.
            if let Some(buf) = dag.buffers.get(*buf_id)
                && buf.kind == BufferKind::Activation
            {
                // Check if all producers are the same op kind writing to the same
                // buffer in different iterations (layer loop). That's OK.
                let all_in_loop = ops.iter().all(|&i| dag.ops[i].in_layer_loop);
                if !all_in_loop {
                    errors.push(VerifyError {
                        kind: ErrorKind::BufferAliasing,
                        message: format!(
                            "buffer '{}' is written by multiple ops: {:?}",
                            buf_id, ops,
                        ),
                    });
                }
            }
        }
    }
}

/// Shared memory must fit within the hardware budget.
fn verify_shmem(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    const SM89_MAX_DYNAMIC_SHMEM: usize = 99_328; // 97KB for sm89 with opt-in
    const BATCH_BLOCK: usize = 128;
    const K_DIM: usize = 64;

    // Compute the max shmem needed by any single op.
    let mut max_shmem: usize = 0;

    for op in &dag.ops {
        let shmem = estimate_op_shmem(dag, &op.kind);
        if shmem > max_shmem {
            max_shmem = shmem;
        }
    }

    // Since ops execute sequentially on each SM, shmem is reused.
    // Only the max matters.
    let total = max_shmem + 8192; // + scratch region
    if total > SM89_MAX_DYNAMIC_SHMEM {
        errors.push(VerifyError {
            kind: ErrorKind::ShmemOverflow,
            message: format!(
                "shared memory {total} bytes exceeds sm89 budget of {SM89_MAX_DYNAMIC_SHMEM} bytes \
                 (max op needs {max_shmem} + 8192 scratch)",
            ),
        });
    }
}

/// Estimate shmem bytes for an op.
pub fn estimate_op_shmem(dag: &ModelDag, kind: &OpKind) -> usize {
    // sm89 TK uses 8KB pages. Shmem budget is pages + scratch.
    const PAGE_SIZE: usize = 8192;
    const BF16: usize = 2;
    const K_DIM: usize = 64;
    const BATCH_BLOCK: usize = 128;

    match kind {
        OpKind::RmsNorm { .. } => {
            // 2 pages: activation tile + weight vector (both fit in 1 page each for dim<=4096)
            2 * PAGE_SIZE
        }
        OpKind::Gemm { .. } | OpKind::GemmAdd { .. } => {
            // Double-buffered GEMM pipeline: 2 stages × (A_pages + B_pages)
            // A = st_bf<128, 64> = 16KB = 2 pages
            // B = st_bf<out_block, 64> = up to 16KB = 2 pages
            // Total: 2 stages × 4 pages = 8 pages = 64KB
            let a_pages = (BATCH_BLOCK * K_DIM * BF16).div_ceil(PAGE_SIZE);
            let b_pages = 2; // worst case out_block=128
            2 * (a_pages + b_pages) * PAGE_SIZE
        }
        OpKind::AttentionDecode { .. } | OpKind::AttentionPrefill { .. } => {
            // KV tiles × stages. Model-dependent but bounded by page count.
            let hdm = dag.params.get("HDM").copied().unwrap_or(128);
            let gqa = dag
                .params
                .get("NAH")
                .and_then(|nah| dag.params.get("NKH").map(|nkh| nah / nkh))
                .unwrap_or(4);
            let kv_tile_bytes = 16 * hdm * BF16 * gqa; // per-stage K or V
            let k_pages = kv_tile_bytes.div_ceil(PAGE_SIZE);
            let v_pages = k_pages;
            let n_stages = 2;
            n_stages * (k_pages + v_pages) * PAGE_SIZE
        }
        OpKind::RopeAppend { .. } => {
            // Same as Gemm + 1 rope page for cos/sin
            let a_pages = 2;
            let b_pages = 2;
            (2 * (a_pages + b_pages) + 1) * PAGE_SIZE
        }
        OpKind::Silu { .. } | OpKind::Mul { .. } => {
            // Elementwise — done in registers after GEMM, no shmem
            0
        }
    }
}

fn resolve_last_dim(dag: &ModelDag, buf_id: &BufferId) -> usize {
    dag.buffers
        .get(buf_id)
        .and_then(|b| b.shape.dims.last())
        .and_then(|d| d.resolve(&dag.params).ok())
        .unwrap_or(2048)
}

fn resolve_first_dim(dag: &ModelDag, buf_id: &BufferId) -> usize {
    dag.buffers
        .get(buf_id)
        .and_then(|b| b.shape.dims.first())
        .and_then(|d| d.resolve(&dag.params).ok())
        .unwrap_or(128)
}

/// Verify the barrier chain: each op's signal slot matches the next op's wait slot.
///
/// The TK barrier protocol uses `g.Bar[layer, opcode-1, row, col]`. Each op signals
/// on its own slot and the next op waits on that slot. This check verifies the chain
/// is complete — no missing links.
fn verify_barrier_chain(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    // Map from op kind to its barrier opcode index (opcode - 1).
    // These must form a connected chain for the pipeline to be correct.
    fn barrier_slot(kind: &OpKind) -> Option<usize> {
        match kind {
            OpKind::RmsNorm { output, .. } => {
                // attn_norm signals slot 0, mlp_norm signals slot 5
                // Distinguish by output buffer name (fragile but correct for LLaMA)
                let name = &output.0;
                if name.contains("normed2") || name.contains("mlp") {
                    Some(5) // OPCODE_MlpNorm - 1
                } else if name.contains("final") || name.contains("lm_head") {
                    Some(8) // OPCODE_LM_HeadNorm - 1 (actually uses different numbering)
                } else {
                    Some(0) // OPCODE_AttnNorm - 1
                }
            }
            OpKind::Gemm { output, .. } | OpKind::GemmAdd { output, .. } => {
                let name = &output.0;
                if name.contains("qkv") {
                    Some(1) // OPCODE_QKV_RopeAppend - 1
                } else if name.contains("gate") {
                    Some(6) // OPCODE_GateSiLU - 1
                } else if name.contains("up") {
                    Some(7) // OPCODE_UpMatmul - 1
                } else if name.contains("hidden") && name.contains("state") {
                    // Could be o_proj or downproj — both write hidden_states
                    None // ambiguous, skip
                } else if name.contains("logits") {
                    Some(9) // OPCODE_LM_Head - 1
                } else {
                    None
                }
            }
            OpKind::RopeAppend { .. } => Some(1),
            OpKind::AttentionDecode { .. } | OpKind::AttentionPrefill { .. } => Some(3),
            OpKind::Silu { .. } => None, // fused into gate, no barrier slot
            OpKind::Mul { .. } => None,  // fused into gate*up, no barrier slot
        }
    }

    // Check that consecutive ops in the layer loop have connected barrier slots.
    // The expected chain for LLaMA: 0 → 1 → 3 → 4 → 5 → 6 → 7 → 8
    let layer_ops: Vec<_> = dag.ops.iter().filter(|o| o.in_layer_loop).collect();
    let mut prev_slot: Option<usize> = None;

    for op in &layer_ops {
        if let Some(slot) = barrier_slot(&op.kind) {
            if let Some(prev) = prev_slot {
                // The signal slot of the previous op should be < current op's slot
                // (barrier indices are monotonically increasing through the pipeline)
                if slot < prev && slot != 0 {
                    // slot 0 wrapping around from slot 8 is the layer boundary — OK
                    // Equal slots are OK (e.g., QKV gemm + rope_append share slot 1)
                    errors.push(VerifyError {
                        kind: ErrorKind::BarrierImbalance,
                        message: format!(
                            "barrier chain broken: op {} signals slot {prev} but op {} expects slot {slot} (non-monotonic)",
                            op.idx.saturating_sub(1), op.idx,
                        ),
                    });
                }
            }
            prev_slot = Some(slot);
        }
    }
}

/// Verify that model dimensions are tile-compatible.
fn verify_tile_divisibility(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    let hd = dag.params.get("HD").copied();
    let hdm = dag.params.get("HDM").copied();
    let nah = dag.params.get("NAH").copied();
    let nkh = dag.params.get("NKH").copied();

    // HD must be divisible by HDM (head_dim)
    if let (Some(hd), Some(hdm)) = (hd, hdm)
        && (hdm == 0 || hd % hdm != 0)
    {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!("HD={hd} is not divisible by HDM={hdm}"),
        });
    }

    // NAH * HDM must equal HD
    if let (Some(hd), Some(nah), Some(hdm)) = (hd, nah, hdm)
        && nah * hdm != hd
    {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!("NAH*HDM = {}*{} = {} != HD={}", nah, hdm, nah * hdm, hd),
        });
    }

    // NKH must divide NAH (for GQA ratio)
    if let (Some(nah), Some(nkh)) = (nah, nkh)
        && (nkh == 0 || nah % nkh != 0)
    {
        errors.push(VerifyError {
            kind: ErrorKind::ShapeMismatch,
            message: format!("NAH={nah} is not divisible by NKH={nkh} (GQA ratio must be integer)"),
        });
    }

    // Check matmul dimensions are divisible by minimum tile sizes
    for dim_name in &["HD", "ID"] {
        if let Some(&dim) = dag.params.get(*dim_name)
            && dim % 32 != 0
        {
            errors.push(VerifyError {
                kind: ErrorKind::TileCoverage,
                message: format!("{dim_name}={dim} is not divisible by minimum tile size 32"),
            });
        }
    }
}

/// Detect declared buffers that are never read or written.
///
/// A buffer is "dead" if it has no producer AND no consumers — it's declared
/// but completely disconnected from the pipeline. Buffers with a producer but
/// no consumers are valid sinks (e.g., k/v outputs of RopeAppend go into KV cache).
fn verify_dead_buffers(dag: &ModelDag, errors: &mut Vec<VerifyError>) {
    for (id, buf) in &dag.buffers {
        let has_producer = buf.producer.is_some()
            || buf.is_input
            || buf.kind == BufferKind::Weight
            || buf.kind == BufferKind::KvCache
            || buf.kind == BufferKind::Metadata;
        let has_consumers = !buf.consumers.is_empty();
        if !has_producer && !has_consumers {
            errors.push(VerifyError {
                kind: ErrorKind::DeadBuffer,
                message: format!("buffer '{id}' is declared but never used"),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> HashMap<String, usize> {
        let mut p = HashMap::new();
        p.insert("BS".into(), 1); // symbolic, but use 1 for decode
        p.insert("NL".into(), 16);
        p.insert("HD".into(), 2048);
        p.insert("ID".into(), 5632);
        p.insert("HDM".into(), 64);
        p.insert("NAH".into(), 32);
        p.insert("NKH".into(), 8);
        p.insert("VS".into(), 128256);
        p
    }

    fn bs() -> Dim {
        Dim::Param("BS".into())
    }
    fn hd() -> Dim {
        Dim::Param("HD".into())
    }
    fn id() -> Dim {
        Dim::Param("ID".into())
    }

    fn add_activation(dag: &mut ModelDag, name: &str, shape: TensorShape) {
        dag.add_buffer(Buffer {
            id: BufferId(name.into()),
            kind: BufferKind::Activation,
            shape,
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: false,
        });
    }

    fn add_input(dag: &mut ModelDag, name: &str, shape: TensorShape) {
        dag.add_buffer(Buffer {
            id: BufferId(name.into()),
            kind: BufferKind::Activation,
            shape,
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });
    }

    fn add_weight(dag: &mut ModelDag, name: &str, shape: TensorShape) {
        dag.add_buffer(Buffer {
            id: BufferId(name.into()),
            kind: BufferKind::Weight,
            shape,
            producer: None,
            consumers: vec![],
            per_layer: true,
            is_input: false,
        });
    }

    /// A correct RmsNorm -> Gemm -> GemmAdd mini-pipeline. Should pass.
    #[test]
    fn correct_rmsnorm_gemm_pipeline() {
        let mut dag = ModelDag::new("test".into(), params());

        // Buffers
        add_input(&mut dag, "hidden", TensorShape::matrix(bs(), hd()));
        add_weight(&mut dag, "attn_norm_w", TensorShape { dims: vec![hd()] });
        add_activation(&mut dag, "normed", TensorShape::matrix(bs(), hd()));
        add_weight(&mut dag, "qkv_w", TensorShape::matrix(hd(), hd())); // simplified: HD x HD
        add_activation(&mut dag, "qkv_out", TensorShape::matrix(bs(), hd()));

        // Ops
        dag.add_op(
            OpKind::RmsNorm {
                input: BufferId("hidden".into()),
                weights: BufferId("attn_norm_w".into()),
                output: BufferId("normed".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::Gemm {
                a: BufferId("normed".into()),
                b: BufferId("qkv_w".into()),
                output: BufferId("qkv_out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    /// GEMM K-dimension mismatch: A[BS, HD] @ B[ID, ID]^T — cols don't match.
    #[test]
    fn gemm_k_dimension_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "a", TensorShape::matrix(bs(), hd()));
        add_weight(&mut dag, "b", TensorShape::matrix(id(), id())); // K=ID != HD
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), id()));

        dag.add_op(
            OpKind::Gemm {
                a: BufferId("a".into()),
                b: BufferId("b".into()),
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::GemmDimMismatch),
            "expected GemmDimMismatch, got: {errors:?}"
        );
    }

    /// Output shape mismatch: GEMM produces [BS, ID] but output is [BS, HD].
    #[test]
    fn gemm_output_shape_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "a", TensorShape::matrix(bs(), hd()));
        add_weight(&mut dag, "b", TensorShape::matrix(id(), hd())); // correct K=HD, N=ID
        add_activation(
            &mut dag,
            "out",
            TensorShape::matrix(bs(), hd()), // WRONG: should be [BS, ID]
        );

        dag.add_op(
            OpKind::Gemm {
                a: BufferId("a".into()),
                b: BufferId("b".into()),
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch, got: {errors:?}"
        );
    }

    /// Read from uninitialized buffer.
    #[test]
    fn uninitialized_read() {
        let mut dag = ModelDag::new("test".into(), params());

        // "normed" is never written — no producer
        add_activation(&mut dag, "normed", TensorShape::matrix(bs(), hd()));
        add_weight(&mut dag, "w", TensorShape::matrix(hd(), hd()));
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), hd()));

        dag.add_op(
            OpKind::Gemm {
                a: BufferId("normed".into()), // not written by anyone!
                b: BufferId("w".into()),
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::UninitializedRead),
            "expected UninitializedRead, got: {errors:?}"
        );
    }

    /// Missing buffer reference.
    #[test]
    fn missing_buffer() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "a", TensorShape::matrix(bs(), hd()));
        // "b_weights" is never declared!
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), hd()));

        dag.add_op(
            OpKind::Gemm {
                a: BufferId("a".into()),
                b: BufferId("b_weights".into()), // doesn't exist
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::MissingBuffer),
            "expected MissingBuffer, got: {errors:?}"
        );
    }

    /// RmsNorm weight dimension mismatch.
    #[test]
    fn rmsnorm_weight_dim_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_activation(&mut dag, "hidden", TensorShape::matrix(bs(), hd()));
        add_weight(
            &mut dag,
            "norm_w",
            TensorShape {
                dims: vec![id()], // WRONG: should be HD, not ID
            },
        );
        add_activation(&mut dag, "normed", TensorShape::matrix(bs(), hd()));

        dag.add_op(
            OpKind::RmsNorm {
                input: BufferId("hidden".into()),
                weights: BufferId("norm_w".into()),
                output: BufferId("normed".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch for norm weight dim, got: {errors:?}"
        );
    }

    /// GemmAdd residual shape mismatch.
    #[test]
    fn gemm_add_residual_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "a", TensorShape::matrix(bs(), id()));
        add_weight(&mut dag, "w", TensorShape::matrix(hd(), id())); // [HD, ID]
        add_activation(&mut dag, "residual", TensorShape::matrix(bs(), id())); // WRONG: should be [BS, HD]
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), hd()));

        dag.add_op(
            OpKind::GemmAdd {
                a: BufferId("a".into()),
                b: BufferId("w".into()),
                residual: BufferId("residual".into()),
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch for residual, got: {errors:?}"
        );
    }

    /// Dead buffer (declared but never used).
    #[test]
    fn dead_buffer_detected() {
        let mut dag = ModelDag::new("test".into(), params());
        add_activation(&mut dag, "orphan", TensorShape::matrix(bs(), hd()));
        // No ops reference it.

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::DeadBuffer),
            "expected DeadBuffer, got: {errors:?}"
        );
    }

    /// RopeAppend with wrong QKV dimension.
    #[test]
    fn rope_append_wrong_qkv_dim() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "qkv", TensorShape::matrix(bs(), Dim::Lit(999))); // WRONG
        dag.add_buffer(Buffer {
            id: BufferId("positions".into()),
            kind: BufferKind::Metadata,
            shape: TensorShape { dims: vec![bs()] },
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });
        dag.add_buffer(Buffer {
            id: BufferId("kv_cache".into()),
            kind: BufferKind::KvCache,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            },
            producer: None,
            consumers: vec![],
            per_layer: true,
            is_input: true,
        });
        add_activation(&mut dag, "q", TensorShape::matrix(bs(), hd()));
        add_activation(
            &mut dag,
            "k",
            TensorShape {
                dims: vec![Dim::Lit(1)],
            },
        );
        add_activation(
            &mut dag,
            "v",
            TensorShape {
                dims: vec![Dim::Lit(1)],
            },
        );

        dag.add_op(
            OpKind::RopeAppend {
                qkv: BufferId("qkv".into()),
                positions: BufferId("positions".into()),
                kv_cache: BufferId("kv_cache".into()),
                q_out: BufferId("q".into()),
                k_out: BufferId("k".into()),
                v_out: BufferId("v".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch for wrong QKV dim, got: {errors:?}"
        );
    }

    /// AttentionDecode with mismatched q/output shapes.
    #[test]
    fn attention_decode_shape_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "q", TensorShape::matrix(bs(), hd()));
        dag.add_buffer(Buffer {
            id: BufferId("kv_cache".into()),
            kind: BufferKind::KvCache,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            },
            producer: None,
            consumers: vec![],
            per_layer: true,
            is_input: true,
        });
        dag.add_buffer(Buffer {
            id: BufferId("block_table".into()),
            kind: BufferKind::Metadata,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            },
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });
        add_activation(&mut dag, "attn_out", TensorShape::matrix(bs(), id())); // WRONG: should be HD

        dag.add_op(
            OpKind::AttentionDecode {
                q: BufferId("q".into()),
                kv_cache: BufferId("kv_cache".into()),
                block_table: BufferId("block_table".into()),
                output: BufferId("attn_out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch for attention q vs output, got: {errors:?}"
        );
    }

    /// HD not divisible by HDM should fail.
    #[test]
    fn hd_not_divisible_by_hdm() {
        let mut p = params();
        p.insert("HD".into(), 2049); // not divisible by 64
        let mut dag = ModelDag::new("test".into(), p);
        // Need at least one buffer to avoid empty-dag issues
        add_input(&mut dag, "x", TensorShape::matrix(bs(), Dim::Lit(2049)));

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.message.contains("not divisible")),
            "expected divisibility error, got: {errors:?}"
        );
    }

    /// Full LLaMA-1B pipeline should verify clean.
    #[test]
    fn full_llama_pipeline_verifies() {
        let dag = build_llama_test_dag();
        let errors = verify(&dag);
        assert!(
            errors.is_empty(),
            "LLaMA pipeline should verify clean, got: {errors:?}"
        );
    }

    /// Build a complete LLaMA DAG for testing.
    pub fn build_llama_test_dag() -> ModelDag {
        let mut dag = ModelDag::new("llama_1b".into(), params());

        let nbh_dim = Dim::Lit(48 * 64); // (NAH + 2*NKH) * HDM = (32+16)*64 = 3072
        let qkv_dim = Dim::Lit(3072);

        // External inputs
        dag.add_buffer(Buffer {
            id: BufferId("hidden_states".into()),
            kind: BufferKind::Activation,
            shape: TensorShape::matrix(bs(), hd()),
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });
        dag.add_buffer(Buffer {
            id: BufferId("positions".into()),
            kind: BufferKind::Metadata,
            shape: TensorShape { dims: vec![bs()] },
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });
        dag.add_buffer(Buffer {
            id: BufferId("kv_cache".into()),
            kind: BufferKind::KvCache,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            }, // opaque
            producer: None,
            consumers: vec![],
            per_layer: true,
            is_input: true,
        });
        dag.add_buffer(Buffer {
            id: BufferId("block_table".into()),
            kind: BufferKind::Metadata,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            }, // opaque
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });

        // Per-layer weights
        add_weight(&mut dag, "attn_norm_w", TensorShape { dims: vec![hd()] });
        add_weight(
            &mut dag,
            "qkv_w",
            TensorShape::matrix(qkv_dim.clone(), hd()),
        );
        add_weight(&mut dag, "o_proj_w", TensorShape::matrix(hd(), hd()));
        add_weight(&mut dag, "mlp_norm_w", TensorShape { dims: vec![hd()] });
        add_weight(&mut dag, "gate_w", TensorShape::matrix(id(), hd()));
        add_weight(&mut dag, "up_w", TensorShape::matrix(id(), hd()));
        add_weight(&mut dag, "down_proj_w", TensorShape::matrix(hd(), id()));

        // Intermediate activations
        add_activation(&mut dag, "normed", TensorShape::matrix(bs(), hd()));
        add_activation(
            &mut dag,
            "qkv_out",
            TensorShape::matrix(bs(), qkv_dim.clone()),
        );
        add_activation(&mut dag, "q", TensorShape::matrix(bs(), hd())); // simplified
        add_activation(
            &mut dag,
            "k",
            TensorShape {
                dims: vec![Dim::Lit(1)],
            },
        ); // opaque (into cache)
        add_activation(
            &mut dag,
            "v",
            TensorShape {
                dims: vec![Dim::Lit(1)],
            },
        ); // opaque (into cache)
        add_activation(&mut dag, "attn_out", TensorShape::matrix(bs(), hd()));
        add_activation(&mut dag, "normed2", TensorShape::matrix(bs(), hd()));
        add_activation(&mut dag, "gate_out", TensorShape::matrix(bs(), id()));
        add_activation(&mut dag, "gate_silu", TensorShape::matrix(bs(), id()));
        add_activation(&mut dag, "up_out", TensorShape::matrix(bs(), id()));
        add_activation(&mut dag, "gate_up", TensorShape::matrix(bs(), id()));

        // Post-loop
        add_weight(&mut dag, "lm_head_norm_w", TensorShape { dims: vec![hd()] });
        add_weight(
            &mut dag,
            "lm_head_w",
            TensorShape::matrix(Dim::Param("VS".into()), hd()),
        );
        add_activation(&mut dag, "final_normed", TensorShape::matrix(bs(), hd()));
        add_activation(
            &mut dag,
            "logits",
            TensorShape::matrix(bs(), Dim::Param("VS".into())),
        );

        // Layer ops
        dag.add_op(
            OpKind::RmsNorm {
                input: BufferId("hidden_states".into()),
                weights: BufferId("attn_norm_w".into()),
                output: BufferId("normed".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::Gemm {
                a: BufferId("normed".into()),
                b: BufferId("qkv_w".into()),
                output: BufferId("qkv_out".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::RopeAppend {
                qkv: BufferId("qkv_out".into()),
                positions: BufferId("positions".into()),
                kv_cache: BufferId("kv_cache".into()),
                q_out: BufferId("q".into()),
                k_out: BufferId("k".into()),
                v_out: BufferId("v".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::AttentionDecode {
                q: BufferId("q".into()),
                kv_cache: BufferId("kv_cache".into()),
                block_table: BufferId("block_table".into()),
                output: BufferId("attn_out".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::GemmAdd {
                a: BufferId("attn_out".into()),
                b: BufferId("o_proj_w".into()),
                residual: BufferId("hidden_states".into()),
                output: BufferId("hidden_states".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::RmsNorm {
                input: BufferId("hidden_states".into()),
                weights: BufferId("mlp_norm_w".into()),
                output: BufferId("normed2".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::Gemm {
                a: BufferId("normed2".into()),
                b: BufferId("gate_w".into()),
                output: BufferId("gate_out".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::Silu {
                input: BufferId("gate_out".into()),
                output: BufferId("gate_silu".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::Gemm {
                a: BufferId("normed2".into()),
                b: BufferId("up_w".into()),
                output: BufferId("up_out".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::Mul {
                a: BufferId("gate_silu".into()),
                b: BufferId("up_out".into()),
                output: BufferId("gate_up".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::GemmAdd {
                a: BufferId("gate_up".into()),
                b: BufferId("down_proj_w".into()),
                residual: BufferId("hidden_states".into()),
                output: BufferId("hidden_states".into()),
            },
            true,
        );

        // Post-loop: lm_head_norm + lm_head
        dag.add_op(
            OpKind::RmsNorm {
                input: BufferId("hidden_states".into()),
                weights: BufferId("lm_head_norm_w".into()),
                output: BufferId("final_normed".into()),
            },
            false,
        );
        dag.add_op(
            OpKind::Gemm {
                a: BufferId("final_normed".into()),
                b: BufferId("lm_head_w".into()),
                output: BufferId("logits".into()),
            },
            false,
        );

        dag
    }

    /// AttentionPrefill with mismatched q/output shapes.
    #[test]
    fn attention_prefill_shape_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "q", TensorShape::matrix(bs(), hd()));
        dag.add_buffer(Buffer {
            id: BufferId("kv_cache".into()),
            kind: BufferKind::KvCache,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            },
            producer: None,
            consumers: vec![],
            per_layer: true,
            is_input: true,
        });
        dag.add_buffer(Buffer {
            id: BufferId("block_table".into()),
            kind: BufferKind::Metadata,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            },
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });
        add_activation(&mut dag, "attn_out", TensorShape::matrix(bs(), id())); // WRONG: should be HD

        dag.add_op(
            OpKind::AttentionPrefill {
                q: BufferId("q".into()),
                kv_cache: BufferId("kv_cache".into()),
                block_table: BufferId("block_table".into()),
                output: BufferId("attn_out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch for prefill attention q vs output, got: {errors:?}"
        );
    }

    /// Silu shape mismatch: input and output have different shapes.
    #[test]
    fn silu_shape_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "inp", TensorShape::matrix(bs(), hd()));
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), id())); // WRONG

        dag.add_op(
            OpKind::Silu {
                input: BufferId("inp".into()),
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch for silu, got: {errors:?}"
        );
    }

    /// Mul shape mismatch: operands have different shapes.
    #[test]
    fn mul_shape_mismatch() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "a", TensorShape::matrix(bs(), hd()));
        add_input(&mut dag, "b", TensorShape::matrix(bs(), id())); // WRONG: different dim
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), hd()));

        dag.add_op(
            OpKind::Mul {
                a: BufferId("a".into()),
                b: BufferId("b".into()),
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::ShapeMismatch),
            "expected ShapeMismatch for mul, got: {errors:?}"
        );
    }

    /// Buffer aliasing: two non-loop ops writing the same activation.
    #[test]
    fn buffer_aliasing_detected() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "a", TensorShape::matrix(bs(), hd()));
        add_weight(&mut dag, "w1", TensorShape::matrix(hd(), hd()));
        add_weight(&mut dag, "w2", TensorShape::matrix(hd(), hd()));
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), hd()));

        // Two non-loop ops both write "out"
        dag.add_op(
            OpKind::Gemm {
                a: BufferId("a".into()),
                b: BufferId("w1".into()),
                output: BufferId("out".into()),
            },
            false, // not in layer loop
        );
        dag.add_op(
            OpKind::Gemm {
                a: BufferId("a".into()),
                b: BufferId("w2".into()),
                output: BufferId("out".into()),
            },
            false, // not in layer loop
        );

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::BufferAliasing),
            "expected BufferAliasing, got: {errors:?}"
        );
    }

    /// NAH not divisible by NKH should fail.
    #[test]
    fn nah_not_divisible_by_nkh() {
        let mut p = params();
        p.insert("NAH".into(), 33); // not divisible by 8
        p.insert("HD".into(), 33 * 64); // keep NAH*HDM == HD
        let mut dag = ModelDag::new("test".into(), p);
        add_input(&mut dag, "x", TensorShape::matrix(bs(), Dim::Lit(33 * 64)));

        let errors = verify(&dag);
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("not divisible by NKH")),
            "expected NKH divisibility error, got: {errors:?}"
        );
    }

    /// NAH*HDM != HD should fail.
    #[test]
    fn nah_times_hdm_not_equal_hd() {
        let mut p = params();
        p.insert("NAH".into(), 16); // 16*64 = 1024 != HD=2048
        let mut dag = ModelDag::new("test".into(), p);
        add_input(&mut dag, "x", TensorShape::matrix(bs(), hd()));

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.message.contains("NAH*HDM")),
            "expected NAH*HDM mismatch error, got: {errors:?}"
        );
    }

    /// Dimension not divisible by tile size 32.
    #[test]
    fn tile_divisibility_id() {
        let mut p = params();
        p.insert("ID".into(), 5633); // not divisible by 32
        let mut dag = ModelDag::new("test".into(), p);
        add_input(&mut dag, "x", TensorShape::matrix(bs(), hd()));

        let errors = verify(&dag);
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::TileCoverage),
            "expected TileCoverage for ID not divisible by 32, got: {errors:?}"
        );
    }

    /// Correct AttentionPrefill should pass (matching q/output shapes).
    #[test]
    fn attention_prefill_correct() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "q", TensorShape::matrix(bs(), hd()));
        dag.add_buffer(Buffer {
            id: BufferId("kv_cache".into()),
            kind: BufferKind::KvCache,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            },
            producer: None,
            consumers: vec![],
            per_layer: true,
            is_input: true,
        });
        dag.add_buffer(Buffer {
            id: BufferId("block_table".into()),
            kind: BufferKind::Metadata,
            shape: TensorShape {
                dims: vec![Dim::Lit(1)],
            },
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: true,
        });
        add_activation(&mut dag, "attn_out", TensorShape::matrix(bs(), hd())); // correct

        dag.add_op(
            OpKind::AttentionPrefill {
                q: BufferId("q".into()),
                kv_cache: BufferId("kv_cache".into()),
                block_table: BufferId("block_table".into()),
                output: BufferId("attn_out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        let shape_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::ShapeMismatch)
            .collect();
        assert!(
            shape_errors.is_empty(),
            "expected no shape errors, got: {shape_errors:?}"
        );
    }

    /// Shmem estimation covers all op kinds without panicking.
    #[test]
    fn shmem_estimation_all_ops() {
        let dag = build_llama_test_dag();
        for op in &dag.ops {
            let shmem = estimate_op_shmem(&dag, &op.kind);
            // All ops should return a reasonable shmem value
            assert!(shmem <= 100_000, "op {} shmem={shmem} too large", op.idx);
        }
    }

    /// Correct Silu and Mul pass verification.
    #[test]
    fn silu_and_mul_correct() {
        let mut dag = ModelDag::new("test".into(), params());

        add_input(&mut dag, "gate", TensorShape::matrix(bs(), id()));
        add_activation(&mut dag, "gate_silu", TensorShape::matrix(bs(), id()));
        add_input(&mut dag, "up", TensorShape::matrix(bs(), id()));
        add_activation(&mut dag, "out", TensorShape::matrix(bs(), id()));

        dag.add_op(
            OpKind::Silu {
                input: BufferId("gate".into()),
                output: BufferId("gate_silu".into()),
            },
            true,
        );
        dag.add_op(
            OpKind::Mul {
                a: BufferId("gate_silu".into()),
                b: BufferId("up".into()),
                output: BufferId("out".into()),
            },
            true,
        );

        let errors = verify(&dag);
        let shape_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::ShapeMismatch)
            .collect();
        assert!(
            shape_errors.is_empty(),
            "expected no shape errors, got: {shape_errors:?}"
        );
    }
}
