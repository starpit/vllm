// SPDX-License-Identifier: Apache-2.0
//! Typed DAG IR for the megakernel.
//!
//! Every buffer and edge carries its tensor shape. The verification pass
//! proves that all producer/consumer pairs agree on dimensions, that shared
//! memory fits within budget, and that barriers are balanced.

use std::collections::HashMap;
use std::fmt;

/// A symbolic dimension — either a concrete value or a named const generic.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Lit(usize),
    Param(String), // e.g. "HD", "NL"
}

impl Dim {
    pub fn resolve(&self, params: &HashMap<String, usize>) -> Result<usize, String> {
        match self {
            Dim::Lit(v) => Ok(*v),
            Dim::Param(name) => params
                .get(name)
                .copied()
                .ok_or_else(|| format!("undefined dimension parameter: {name}")),
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Lit(v) => write!(f, "{v}"),
            Dim::Param(s) => write!(f, "{s}"),
        }
    }
}

/// Tensor shape: [rows, cols]. For a weight with layers: [layers, rows, cols].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<Dim>,
}

impl TensorShape {
    pub fn matrix(rows: Dim, cols: Dim) -> Self {
        Self {
            dims: vec![rows, cols],
        }
    }

    #[allow(dead_code)]
    pub fn resolve(&self, params: &HashMap<String, usize>) -> Result<Vec<usize>, String> {
        self.dims.iter().map(|d| d.resolve(params)).collect()
    }

    /// Number of elements (product of all dims).
    #[allow(dead_code)]
    pub fn numel(&self, params: &HashMap<String, usize>) -> Result<usize, String> {
        self.resolve(params).map(|ds| ds.iter().product())
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// A named buffer (activation or weight) in the DAG.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub String);

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Buffer kind — distinguishes activations (read/write) from weights (read-only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// Activation: allocated in GPU global memory, written by one op, read by others.
    Activation,
    /// Weight: read-only, loaded from safetensors.
    Weight,
    /// KV cache: special paged structure.
    KvCache,
    /// Metadata: positions, block tables, etc.
    Metadata,
}

/// A buffer declaration with its shape and kind.
#[derive(Clone, Debug)]
pub struct Buffer {
    pub id: BufferId,
    pub kind: BufferKind,
    pub shape: TensorShape,
    /// Which op produces this buffer (None for weights/inputs).
    pub producer: Option<OpIdx>,
    /// Which ops consume this buffer.
    pub consumers: Vec<OpIdx>,
    /// Whether this buffer is per-layer (indexed by layer in the loop).
    #[allow(dead_code)]
    pub per_layer: bool,
    /// Whether this buffer is an external input (provided by the caller, not produced by any op).
    pub is_input: bool,
}

pub type OpIdx = usize;

/// The known megakernel op types. Each carries typed input/output references.
#[derive(Clone, Debug)]
pub enum OpKind {
    /// Token-id → hidden state lookup: `output[i] = embed_tokens[input_ids[i]]`.
    /// Produces the initial `hidden_states` activation at the top of the
    /// forward pass. The `weights` buffer is classified as a weight of
    /// Rust type `Embedding` in the field extractor.
    Embed {
        input_ids: BufferId,
        weights: BufferId,
        output: BufferId,
    },
    /// RMS normalization: input[BS, D] * weights[D] -> output[BS, D]
    RmsNorm {
        input: BufferId,
        weights: BufferId,
        output: BufferId,
    },
    /// General matrix multiply: A[BS, K] @ B[N, K]^T -> output[BS, N].
    Gemm {
        a: BufferId,
        b: BufferId,
        output: BufferId,
    },
    /// GEMM + residual add: A[BS, K] @ B[N, K]^T + residual[BS, N] -> output[BS, N]
    GemmAdd {
        a: BufferId,
        b: BufferId,
        residual: BufferId,
        output: BufferId,
    },
    /// RoPE + KV cache append. Takes separate Q, K, V projections
    /// (the solver may have fused the upstream GEMMs, but the DAG
    /// always stores separate inputs). The `rotary` buffer is the
    /// pre-computed cos/sin cache (classified as a global weight of
    /// Rust type `RotaryCache` in the field extractor).
    RopeAppend {
        q_in: BufferId,
        k_in: BufferId,
        v_in: BufferId,
        positions: BufferId,
        rotary: BufferId,
        kv_cache: BufferId,
        q_out: BufferId,
        k_out: BufferId,
        v_out: BufferId,
    },
    /// Paged grouped-query attention decode (single token per sequence)
    AttentionDecode {
        q: BufferId,
        kv_cache: BufferId,
        block_table: BufferId,
        output: BufferId,
    },
    /// Paged grouped-query attention prefill (variable-length sequences)
    AttentionPrefill {
        q: BufferId,
        kv_cache: BufferId,
        block_table: BufferId,
        output: BufferId,
    },
    /// SiLU activation: x * sigmoid(x)
    Silu { input: BufferId, output: BufferId },
    /// Element-wise multiply: a * b -> output
    Mul {
        a: BufferId,
        b: BufferId,
        output: BufferId,
    },
    /// Per-column bias add: input[BS, N] + bias[N] -> output[BS, N].
    /// Expressed explicitly in the DSL for architectures that use bias
    /// (e.g. Qwen2 QKV projections). The solver decides whether to
    /// fuse this with the upstream GEMM or dispatch it standalone.
    BiasAdd {
        input: BufferId,
        bias: BufferId,
        output: BufferId,
    },
}

/// A single operation in the DAG.
#[derive(Clone, Debug)]
pub struct Op {
    pub idx: OpIdx,
    pub kind: OpKind,
    /// True if this op is inside the layer loop.
    pub in_layer_loop: bool,
}

impl Op {
    /// All buffer IDs read by this op.
    pub fn inputs(&self) -> Vec<&BufferId> {
        match &self.kind {
            OpKind::Embed {
                input_ids, weights, ..
            } => vec![input_ids, weights],
            OpKind::RmsNorm { input, weights, .. } => vec![input, weights],
            OpKind::Gemm { a, b, .. } => vec![a, b],
            OpKind::GemmAdd { a, b, residual, .. } => vec![a, b, residual],
            OpKind::RopeAppend {
                q_in,
                k_in,
                v_in,
                positions,
                rotary,
                kv_cache,
                ..
            } => vec![q_in, k_in, v_in, positions, rotary, kv_cache],
            OpKind::AttentionDecode {
                q,
                kv_cache,
                block_table,
                ..
            }
            | OpKind::AttentionPrefill {
                q,
                kv_cache,
                block_table,
                ..
            } => vec![q, kv_cache, block_table],
            OpKind::Silu { input, .. } => vec![input],
            OpKind::Mul { a, b, .. } => vec![a, b],
            OpKind::BiasAdd { input, bias, .. } => vec![input, bias],
        }
    }

    /// All buffer IDs written by this op.
    pub fn outputs(&self) -> Vec<&BufferId> {
        match &self.kind {
            OpKind::Embed { output, .. } => vec![output],
            OpKind::RmsNorm { output, .. } => vec![output],
            OpKind::Gemm { output, .. } => vec![output],
            OpKind::GemmAdd { output, .. } => vec![output],
            OpKind::RopeAppend {
                q_out,
                k_out,
                v_out,
                ..
            } => vec![q_out, k_out, v_out],
            OpKind::AttentionDecode { output, .. } | OpKind::AttentionPrefill { output, .. } => {
                vec![output]
            }
            OpKind::Silu { output, .. } => vec![output],
            OpKind::Mul { output, .. } => vec![output],
            OpKind::BiasAdd { output, .. } => vec![output],
        }
    }
}

/// Shared memory requirement for an op.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ShmemRequirement {
    /// Bytes needed for tile data (A, B buffers for GEMM; K, V for attention).
    pub tile_bytes: usize,
    /// Bytes needed for scratch (semaphores, temporaries).
    pub scratch_bytes: usize,
    /// Total = tile_bytes + scratch_bytes.
    pub total_bytes: usize,
}

/// The complete model DAG with all verification data.
#[derive(Clone, Debug)]
pub struct ModelDag {
    pub name: String,
    /// Dimension parameters: NL, HD, ID, HDM, NAH, NKH, VS, etc.
    pub params: HashMap<String, usize>,
    /// All buffers (activations, weights, caches).
    pub buffers: HashMap<BufferId, Buffer>,
    /// Ops in topological (execution) order.
    pub ops: Vec<Op>,
    /// SM count (may be resolved later at runtime).
    #[allow(dead_code)]
    pub sm_count: Option<usize>,
}

impl ModelDag {
    pub fn new(name: String, params: HashMap<String, usize>) -> Self {
        Self {
            name,
            params,
            buffers: HashMap::new(),
            ops: Vec::new(),
            sm_count: None,
        }
    }

    pub fn add_buffer(&mut self, buf: Buffer) {
        self.buffers.insert(buf.id.clone(), buf);
    }

    pub fn add_op(&mut self, kind: OpKind, in_layer_loop: bool) -> OpIdx {
        let idx = self.ops.len();
        // Register this op as producer/consumer in its buffers.
        let op = Op {
            idx,
            kind,
            in_layer_loop,
        };
        for out_id in op.outputs() {
            if let Some(buf) = self.buffers.get_mut(out_id) {
                if buf.producer.is_some() && buf.kind == BufferKind::Activation {
                    // Will be caught by verify() — record it for now.
                }
                buf.producer = Some(idx);
            }
        }
        for in_id in op.inputs() {
            if let Some(buf) = self.buffers.get_mut(in_id) {
                buf.consumers.push(idx);
            }
        }
        self.ops.push(op);
        idx
    }

    /// Resolve a dimension parameter.
    #[allow(dead_code)]
    pub fn dim(&self, name: &str) -> Result<usize, String> {
        self.params
            .get(name)
            .copied()
            .ok_or_else(|| format!("undefined parameter: {name}"))
    }
}

/// Barrier between ops with expected signal/wait counts.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Barrier {
    pub name: String,
    /// Op that signals completion.
    pub producer_op: OpIdx,
    /// Ops that wait on this barrier.
    pub consumer_ops: Vec<OpIdx>,
    /// Number of SMs that will signal (producer tile count).
    pub signal_count: usize,
    /// Number of signals each consumer expects.
    pub wait_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_params() -> HashMap<String, usize> {
        let mut p = HashMap::new();
        p.insert("NL".into(), 16);
        p.insert("HD".into(), 2048);
        p.insert("ID".into(), 5632);
        p.insert("HDM".into(), 64);
        p.insert("NAH".into(), 32);
        p.insert("NKH".into(), 8);
        p.insert("VS".into(), 128256);
        p
    }

    #[test]
    fn dim_resolution() {
        let params = llama_params();
        assert_eq!(Dim::Lit(42).resolve(&params).unwrap(), 42);
        assert_eq!(Dim::Param("HD".into()).resolve(&params).unwrap(), 2048);
        assert!(Dim::Param("NOPE".into()).resolve(&params).is_err());
    }

    #[test]
    fn tensor_shape_display() {
        let s = TensorShape::matrix(Dim::Param("BS".into()), Dim::Param("HD".into()));
        assert_eq!(format!("{s}"), "[BS, HD]");
    }

    #[test]
    fn op_input_output_tracking() {
        let params = llama_params();
        let mut dag = ModelDag::new("test".into(), params);

        dag.add_buffer(Buffer {
            id: BufferId("hidden".into()),
            kind: BufferKind::Activation,
            shape: TensorShape::matrix(Dim::Param("BS".into()), Dim::Param("HD".into())),
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: false,
        });
        dag.add_buffer(Buffer {
            id: BufferId("norm_w".into()),
            kind: BufferKind::Weight,
            shape: TensorShape {
                dims: vec![Dim::Param("HD".into())],
            },
            producer: None,
            consumers: vec![],
            per_layer: true,
            is_input: false,
        });
        dag.add_buffer(Buffer {
            id: BufferId("normed".into()),
            kind: BufferKind::Activation,
            shape: TensorShape::matrix(Dim::Param("BS".into()), Dim::Param("HD".into())),
            producer: None,
            consumers: vec![],
            per_layer: false,
            is_input: false,
        });

        let op_idx = dag.add_op(
            OpKind::RmsNorm {
                input: BufferId("hidden".into()),
                weights: BufferId("norm_w".into()),
                output: BufferId("normed".into()),
            },
            true,
        );

        assert_eq!(dag.ops[op_idx].inputs().len(), 2);
        assert_eq!(dag.ops[op_idx].outputs().len(), 1);
        assert_eq!(dag.buffers[&BufferId("normed".into())].producer, Some(0));
        assert!(
            dag.buffers[&BufferId("hidden".into())]
                .consumers
                .contains(&0)
        );
    }
}
