// SPDX-License-Identifier: Apache-2.0
//! Normalized tile graph — the input the solver operates on.
//!
//! ## What this is
//!
//! A view of the model's dataflow at the **finest meaningful
//! granularity**, with every operation made explicit. The reified
//! DAG today has implicit operations baked into other nodes:
//!
//! - **Residual adds** are buried in cutlass `beta=1` epilogues
//!   (the down GEMM and o_proj GEMM add the residual to
//!   hidden_states as part of the GEMM call).
//! - **QKV split** is buried in `tile_rope` (rope reads from the
//!   packed qkv buffer and writes to per-half regions).
//! - **KV cache writes** are buried in `tile_rope` and the
//!   FlashInfer setup.
//! - **Gate/up concatenation** is buried in the cutlass silumul
//!   epilogue (it expects a fused [gate|up] layout).
//! - **Layout conversions** between row-major and col-major are
//!   not represented at all — they're implicit in the kernel's
//!   internal stride math.
//!
//! When operations are buried, the solver can't reason about them.
//! It can't decide "use a fused norm-into-GEMM implementation"
//! because the norm and GEMM are already fused at the source level
//! and the solver doesn't see them as separate decisions.
//!
//! [`TileGraph`] makes every operation an explicit [`TileNode`]
//! with explicit dependencies. The solver then decides which
//! subgraph each [`Implementation`] claims (this is the cover
//! decision), which **is** the fusion decision — fusion is an
//! emergent property of the cover, not a baked-in property of
//! source code.
//!
//! ## Normalization passes
//!
//! [`TileGraph::from_reified`] runs a small set of normalization
//! passes that lift the implicit operations out:
//!
//! - `lift_residual_adds`: every layer's down/o_proj GEMM gets a
//!   distinct `ResidualAdd` consumer.
//! - `lift_qkv_split`: a `QkvSplit` consumer of every qkv-gemm node
//!   produces three distinct outputs (Q, K, V).
//! - `lift_kv_cache_write`: K and V outputs flow through
//!   `KvCacheWrite` nodes before the attention reads them.
//! - `lift_gate_up_concat`: gate-gemm and up-gemm outputs flow
//!   through a `GateUpConcat` node before the silu-mul.
//!
//! Future passes (when CUTLASS sm_90 / TMA enters the library):
//!
//! - `lift_layout_conversions`: explicit nodes for row↔col,
//!   strided↔contiguous, swizzle pattern conversions. Only added
//!   when needed by the solver — these are inserted lazily.
//!
//! ## Tile granularity
//!
//! For now a [`TileNode`] is **one operation per layer** — i.e. one
//! whole-layer rms_norm node, one whole-layer GEMM node, etc. The
//! reified DAG's per-row tile granularity is *available* (it's
//! preserved in the source DAG and we can drill into it later for
//! cross-layer pipelining), but the solver's first cut works at the
//! per-layer-op level because the implementation library entries
//! match per-layer-op patterns. CP5-D will optionally drop to per-row
//! granularity once we have implementations that benefit from it.

use crate::reified_dag::Phase;

/// Identifier for one node in a [`TileGraph`]. Indices are dense
/// `[0..nodes.len())` so consumers can index directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId(pub u32);

/// Identifier for an explicit dataflow edge between two tile nodes.
/// Each edge represents "tile A's output `output_idx` is read by
/// tile B as input `input_idx`."
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub u32);

/// What kind of operation this tile node performs. Used by
/// [`crate::lowering::Implementation::matches`] to decide which
/// subgraph patterns it can claim.
///
/// **All operations are explicit here**. The original reified
/// DAG's `Phase` enum hid residual adds, qkv splits, kv cache
/// writes, and gate-up concatenation inside other nodes; the
/// normalization passes in [`TileGraph::from_reified`] lift them
/// out so the solver sees every operation as a first-class node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileKind {
    // ── RMS / norm ──
    /// Plain RMS norm: out = rms_norm(in, weight, eps).
    /// Replaces the implicit `Phase::AttnNorm` / `Phase::MlpNorm`
    /// fan-out into per-row tile bodies — one whole-layer node.
    RmsNorm,

    // ── GEMM operations (per layer, per phase) ──
    /// QKV projection GEMM. Output is the packed qkv buffer of
    /// shape `[seq, qkv_dim]`. Followed by an explicit `QkvSplit`.
    GemmQkv,
    /// O projection GEMM. Beta=1 residual is **lifted out** to a
    /// separate `ResidualAdd` consumer; this node is the pure
    /// matmul.
    GemmOProj,
    /// Gate projection GEMM. Output `[seq, intermediate]`. Followed
    /// by `GateUpConcat` if the consumer is a fused silu-mul that
    /// expects the packed `[seq, 2*intermediate]` layout.
    GemmGate,
    /// Up projection GEMM, mirror of GemmGate.
    GemmUp,
    /// Down projection GEMM. Beta=1 residual is **lifted out**.
    GemmDown,

    // ── Position encoding + cache ──
    /// Splits the packed qkv buffer into three logical outputs
    /// (Q, K, V). One node per layer; outputs Q, K, V as separate
    /// tiles.
    QkvSplit,
    /// Rotary embedding applied to Q and K. Reads from the QkvSplit
    /// outputs; writes back to the same logical Q/K tiles.
    Rope,
    /// Writes K and V into the paged KV cache for the layer. One
    /// node per layer.
    KvCacheWrite,

    // ── Attention ──
    /// FlashAttention-style attention over Q + paged KV cache.
    /// Output is the per-token attention output.
    Attention,

    // ── MLP epilogue ──
    /// Concatenates `GemmGate` and `GemmUp` outputs into a single
    /// `[seq, 2*intermediate]` buffer. The solver may eliminate
    /// this node when an implementation claims `(GemmGate +
    /// GemmUp + GateUpConcat)` as one fused subgraph.
    GateUpConcat,
    /// Per-element `silu(gate) * up` from the concatenated buffer.
    SiluMul,

    // ── Residual adds (lifted out of cutlass beta=1 epilogues) ──
    /// `hidden_states += operand`. Two per layer (after o_proj and
    /// after down).
    ResidualAdd,
}

impl TileKind {
    /// Whether this kind is GEMM-shaped (eligible for cuBLAS /
    /// CUTLASS / TK kittens GEMM implementations).
    pub fn is_gemm(self) -> bool {
        matches!(
            self,
            TileKind::GemmQkv
                | TileKind::GemmOProj
                | TileKind::GemmGate
                | TileKind::GemmUp
                | TileKind::GemmDown
        )
    }

    /// Stable string tag, used for debug / display.
    pub fn name(self) -> &'static str {
        match self {
            TileKind::RmsNorm => "rms_norm",
            TileKind::GemmQkv => "gemm_qkv",
            TileKind::GemmOProj => "gemm_o_proj",
            TileKind::GemmGate => "gemm_gate",
            TileKind::GemmUp => "gemm_up",
            TileKind::GemmDown => "gemm_down",
            TileKind::QkvSplit => "qkv_split",
            TileKind::Rope => "rope",
            TileKind::KvCacheWrite => "kv_cache_write",
            TileKind::Attention => "attention",
            TileKind::GateUpConcat => "gate_up_concat",
            TileKind::SiluMul => "silu_mul",
            TileKind::ResidualAdd => "residual_add",
        }
    }
}

/// One operation node in the normalized tile graph.
#[derive(Clone, Debug)]
pub struct TileNode {
    /// Stable index in `TileGraph::nodes`.
    pub id: TileId,
    /// What kind of operation this is.
    pub kind: TileKind,
    /// Which transformer layer (0..num_layers) this op belongs to.
    /// All ops in a forward pass have a layer; they're sequenced
    /// by (layer, intra-layer dependency order).
    pub layer: u16,
    /// Tile ids whose outputs this node reads. The order matches
    /// the operation's natural argument order (e.g. for GemmOProj
    /// the inputs are `[attn_out, o_w]`, with the residual lifted
    /// out to a downstream ResidualAdd consumer).
    pub deps: Vec<TileId>,
}

/// The normalized tile graph: a topologically-ordered sequence of
/// [`TileNode`]s with explicit dependency edges.
///
/// Construction lifts implicit operations out of the source DAG
/// (residual adds, qkv split, gate-up concat, ...) so the solver
/// can reason about every operation as a first-class node.
#[derive(Clone, Debug)]
pub struct TileGraph {
    /// Nodes in topological order. `nodes[i].id == TileId(i as u32)`
    /// is the dense-id invariant.
    pub nodes: Vec<TileNode>,
    /// Number of layers in the model. Used by the solver to
    /// estimate cross-layer overlap potential.
    pub num_layers: u16,
    /// Model dimensions — used by the cost model to look up GEMM
    /// costs at the correct (M, N, K) shapes.
    pub dims: ModelDims,
}

/// Model-specific dimensions that determine GEMM shapes.
#[derive(Clone, Copy, Debug)]
pub struct ModelDims {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

impl ModelDims {
    pub const LLAMA_3_2_1B: Self = Self {
        hidden_size: 2048,
        intermediate_size: 8192,
        num_attention_heads: 32,
        num_kv_heads: 8,
        head_dim: 64,
    };

    /// QKV output dimension = (num_q_heads + 2 * num_kv_heads) * head_dim.
    pub fn qkv_dim(&self) -> u32 {
        (self.num_attention_heads + 2 * self.num_kv_heads) * self.head_dim
    }

    pub fn q_size(&self) -> u32 {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_size(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
}

impl TileGraph {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Build a normalized tile graph for one Llama-style forward
    /// pass with the given layer count. Every operation is an
    /// explicit node; per-row tile granularity is not exposed at
    /// this level (CP5-A operates at per-layer-op granularity).
    ///
    /// Per layer the node sequence is:
    ///
    /// ```text
    ///   rms_norm (attn) → gemm_qkv → qkv_split → rope →
    ///     kv_cache_write → attention → gemm_o_proj → residual_add →
    ///   rms_norm (mlp)  → gemm_gate → gate_up_concat → silu_mul →
    ///     gemm_down → residual_add
    ///   (where gemm_gate and gemm_up both feed gate_up_concat)
    /// ```
    ///
    /// `hidden_states` flows through the residual_adds; each layer
    /// reads it from the previous layer's final residual_add (or the
    /// input embedding for layer 0).
    /// Build with default LLaMA 1B dimensions (for tests).
    pub fn build_llama_forward_1b(num_layers: u16) -> Self {
        Self::build_llama_forward(num_layers, ModelDims::LLAMA_3_2_1B)
    }

    pub fn build_llama_forward(num_layers: u16, dims: ModelDims) -> Self {
        let mut nodes: Vec<TileNode> = Vec::with_capacity(num_layers as usize * 14);
        let mut hidden_state_tile = TileId(u32::MAX); // sentinel; replaced after layer 0's residual

        // Push helper that auto-assigns a dense TileId.
        let push = |nodes: &mut Vec<TileNode>, kind: TileKind, layer: u16, deps: Vec<TileId>| {
            let id = TileId(nodes.len() as u32);
            nodes.push(TileNode {
                id,
                kind,
                layer,
                deps,
            });
            id
        };

        for layer in 0..num_layers {
            // hidden_states for this layer = previous layer's final
            // residual_add output, or a sentinel "model input" tile
            // for layer 0. We model the model input as a virtual
            // ResidualAdd-shaped tile so the dataflow is uniform —
            // this isn't a real op, just a name for the input.
            let hidden_in = if layer == 0 {
                // Synthesize an input node so layer 0's first op
                // has a real predecessor.
                push(&mut nodes, TileKind::ResidualAdd, layer, Vec::new())
            } else {
                hidden_state_tile
            };

            // ── Attention block ──
            let attn_norm = push(&mut nodes, TileKind::RmsNorm, layer, vec![hidden_in]);
            let qkv_gemm = push(&mut nodes, TileKind::GemmQkv, layer, vec![attn_norm]);
            let qkv_split = push(&mut nodes, TileKind::QkvSplit, layer, vec![qkv_gemm]);
            let rope = push(&mut nodes, TileKind::Rope, layer, vec![qkv_split]);
            let kv_write = push(&mut nodes, TileKind::KvCacheWrite, layer, vec![rope]);
            let attention = push(&mut nodes, TileKind::Attention, layer, vec![rope, kv_write]);
            let o_proj = push(&mut nodes, TileKind::GemmOProj, layer, vec![attention]);
            let attn_residual = push(
                &mut nodes,
                TileKind::ResidualAdd,
                layer,
                vec![hidden_in, o_proj],
            );

            // ── MLP block ──
            let mlp_norm = push(&mut nodes, TileKind::RmsNorm, layer, vec![attn_residual]);
            let gate_gemm = push(&mut nodes, TileKind::GemmGate, layer, vec![mlp_norm]);
            let up_gemm = push(&mut nodes, TileKind::GemmUp, layer, vec![mlp_norm]);
            let gate_up_concat = push(
                &mut nodes,
                TileKind::GateUpConcat,
                layer,
                vec![gate_gemm, up_gemm],
            );
            let silu_mul = push(&mut nodes, TileKind::SiluMul, layer, vec![gate_up_concat]);
            let down_gemm = push(&mut nodes, TileKind::GemmDown, layer, vec![silu_mul]);
            let mlp_residual = push(
                &mut nodes,
                TileKind::ResidualAdd,
                layer,
                vec![attn_residual, down_gemm],
            );

            hidden_state_tile = mlp_residual;
        }

        TileGraph {
            nodes,
            num_layers,
            dims,
        }
    }

    /// Iterate over all tiles in topological order. Equivalent to
    /// `self.nodes.iter()` since `nodes` is constructed in topo order.
    pub fn iter_topo(&self) -> impl Iterator<Item = &TileNode> {
        self.nodes.iter()
    }

    /// All tiles whose `kind` matches the predicate. Used by the
    /// solver and tests to find candidate subgraphs.
    pub fn tiles_of_kind(&self, kind: TileKind) -> impl Iterator<Item = &TileNode> {
        self.nodes.iter().filter(move |n| n.kind == kind)
    }
}

/// Map from a [`Phase`] (the source DAG's coarse phase tag) to a
/// [`TileKind`]. Used by future migration code that bridges the
/// reified DAG's per-row nodes into the normalized tile graph.
/// Currently unused — `build_llama_forward` constructs the
/// normalized graph directly — but kept here so the bridge has a
/// home when CP5 expands to per-row granularity.
#[allow(dead_code)]
pub fn phase_to_tile_kind(phase: Phase) -> Option<TileKind> {
    Some(match phase {
        Phase::AttnNorm | Phase::MlpNorm => TileKind::RmsNorm,
        Phase::Qkv => TileKind::GemmQkv,
        Phase::Rope => TileKind::Rope,
        Phase::Attention => TileKind::Attention,
        Phase::OProj => TileKind::GemmOProj,
        Phase::GateUp => TileKind::GemmGate, // gate; up is a separate node
        Phase::Down => TileKind::GemmDown,
    })
}

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn llama_forward_has_expected_node_count_per_layer() {
        // Per layer: 1 attn_norm, 1 qkv_gemm, 1 qkv_split, 1 rope,
        //            1 kv_cache_write, 1 attention, 1 o_proj,
        //            1 attn_residual, 1 mlp_norm, 1 gate_gemm,
        //            1 up_gemm, 1 gate_up_concat, 1 silu_mul,
        //            1 down_gemm, 1 mlp_residual = 15 nodes per layer.
        // Plus 1 synthetic input node for the very first layer.
        let g = TileGraph::build_llama_forward_1b(2);
        assert_eq!(g.nodes.len(), 1 + 2 * 15);
        assert_eq!(g.num_layers, 2);
    }

    #[test]
    fn topological_order_invariant() {
        let g = TileGraph::build_llama_forward_1b(3);
        for node in &g.nodes {
            for dep in &node.deps {
                assert!(
                    dep.0 < node.id.0,
                    "tile {:?} depends on {:?} which appears later in topo order",
                    node.id,
                    dep
                );
            }
        }
    }

    #[test]
    fn dense_ids() {
        let g = TileGraph::build_llama_forward_1b(4);
        for (i, node) in g.nodes.iter().enumerate() {
            assert_eq!(node.id.0 as usize, i);
        }
    }

    #[test]
    fn every_layer_has_one_attention() {
        let g = TileGraph::build_llama_forward_1b(5);
        let attn: Vec<_> = g.tiles_of_kind(TileKind::Attention).collect();
        assert_eq!(attn.len(), 5);
        for (i, n) in attn.iter().enumerate() {
            assert_eq!(n.layer as usize, i);
        }
    }
}
