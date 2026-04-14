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
    /// Q projection GEMM: `[seq, hidden] → [seq, q_size]`.
    GemmQ,
    /// K projection GEMM: `[seq, hidden] → [seq, kv_size]`.
    GemmK,
    /// V projection GEMM: `[seq, hidden] → [seq, kv_size]`.
    GemmV,
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
    /// Final lm_head projection: `logits = hidden @ lm_head^T`.
    /// Output `[seq, vocab_size]`. Runs once at the end of the
    /// forward pass, after all decoder layers and the final norm.
    GemmLmHead,

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

    // ── Bias add ──
    /// Per-column bias broadcast add: `out[row, col] += bias[col]`.
    /// Expressed explicitly in the DSL via `bias_add(input, weights)`.
    /// The solver can fuse `Gemm* + BiasAdd` into one kernel (e.g.
    /// cuBLAS bias epilogue) or dispatch them separately.
    BiasAdd,

    // ── Embedding lookup (pre-loop, runs once per forward) ──
    /// `hidden_states[i] = embed_tokens[input_ids[i]]`. Emitted by
    /// the DSL op `hidden_states = embed(input_ids, embed_tokens)`
    /// at the top of a `forward!()` body. Lives in the pre-loop
    /// phase (tagged with `layer == PRE_LOOP_LAYER`) so the
    /// generated `Model::forward` runs it once before the
    /// per-layer dispatch match, not once per layer.
    Embed,
}

/// Sentinel layer index for pre-loop tiles (`Embed` today). Picked
/// as `u16::MAX` so it's clearly out of the actual layer range
/// `0..num_layers` and can't collide with a real layer, while still
/// fitting in the existing `u16 layer` field on `TileNode`.
pub const PRE_LOOP_LAYER: u16 = u16::MAX;

impl TileKind {
    /// Whether this kind is GEMM-shaped (eligible for cuBLAS /
    /// CUTLASS / TK kittens GEMM implementations).
    pub fn is_gemm(self) -> bool {
        matches!(
            self,
            TileKind::GemmQ
                | TileKind::GemmK
                | TileKind::GemmV
                | TileKind::GemmOProj
                | TileKind::GemmGate
                | TileKind::GemmUp
                | TileKind::GemmDown
                | TileKind::GemmLmHead
        )
    }

    /// Stable string tag, used for debug / display.
    pub fn name(self) -> &'static str {
        match self {
            TileKind::RmsNorm => "rms_norm",
            TileKind::GemmQ => "gemm_q",
            TileKind::GemmK => "gemm_k",
            TileKind::GemmV => "gemm_v",
            TileKind::GemmOProj => "gemm_o_proj",
            TileKind::GemmGate => "gemm_gate",
            TileKind::GemmUp => "gemm_up",
            TileKind::GemmDown => "gemm_down",
            TileKind::GemmLmHead => "gemm_lm_head",
            TileKind::QkvSplit => "qkv_split",
            TileKind::Rope => "rope",
            TileKind::KvCacheWrite => "kv_cache_write",
            TileKind::Attention => "attention",
            TileKind::GateUpConcat => "gate_up_concat",
            TileKind::SiluMul => "silu_mul",
            TileKind::ResidualAdd => "residual_add",
            TileKind::BiasAdd => "bias_add",
            TileKind::Embed => "embed",
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
    /// Metadata for cost model and debug — not semantic in the FUF.
    pub layer: u16,
    /// Tile ids whose outputs this node reads. The order matches
    /// the operation's natural argument order (e.g. for GemmOProj
    /// the inputs are `[attn_out, o_w]`, with the residual lifted
    /// out to a downstream ResidualAdd consumer).
    pub deps: Vec<TileId>,
    /// DSL weight buffer name for this tile's weight operand.
    /// E.g. "input_layernorm", "self_attn.o_proj", "mlp.gate_proj".
    /// Set for RmsNorm, Gemm, and Embed tiles. None for non-weight
    /// tiles (Attention, SiluMul, ResidualAdd, etc.).
    pub weight_name: Option<String>,
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

/// Model-specific dimensions that determine GEMM shapes **and** the
/// structural shape of the tile graph.
///
/// Per-instance numeric dims (hidden / intermediate / heads / …)
/// parameterize the cost model.
#[derive(Clone, Copy, Debug)]
pub struct ModelDims {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// Vocabulary size — the output dim of the lm_head projection.
    pub vocab_size: u32,
}

impl ModelDims {
    pub const LLAMA_3_2_1B: Self = Self {
        hidden_size: 2048,
        intermediate_size: 8192,
        num_attention_heads: 32,
        num_kv_heads: 8,
        head_dim: 64,
        vocab_size: 128256,
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
    /// Build with default LLaMA 1B dimensions. Test-only convenience.
    ///
    /// Routes through the canonical CFG → FUF pipeline, so tests
    /// using this helper exercise the same code path as the real
    /// `forward!()` macro. Produces a LLaMA tile graph with a
    /// synthetic-input `ResidualAdd` seed (no `embed`) plus
    /// `num_layers` × 17 loop-body tiles (no post-loop).
    #[cfg(test)]
    pub fn build_llama_forward_1b(num_layers: u16) -> Self {
        Self::build_llama_forward(num_layers, ModelDims::LLAMA_3_2_1B)
    }

    /// Test-only helper — builds a LLaMA tile graph by parsing a
    /// canonical DSL and running it through `fuf::build_fuf`.
    ///
    /// This replaces the legacy hardcoded builder; the DSL body is
    /// identical to the real LLaMA DSL minus `embed` and the
    /// post-loop (final norm + lm_head) so the synthetic-input
    /// `ResidualAdd` seed + 17 × num_layers loop-body tiles shape
    /// matches what the hardcoded builder used to emit.
    #[cfg(test)]
    pub fn build_llama_forward(num_layers: u16, dims: ModelDims) -> Self {
        let src = format!(
            r#"
            kernel llama<NL={num_layers}, HD={hd}, ID={id}, HDM={hdm}, NAH={nah}, NKH={nkh}, VS={vs}> {{
                for layer in 0..NL {{
                    let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    let q = gemm(normed, self_attn.q_proj[layer]);
                    let k = gemm(normed, self_attn.k_proj[layer]);
                    let v = gemm(normed, self_attn.v_proj[layer]);
                    let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                    let attn = attention(q, k, v, kv_cache[layer], block_table);
                    let oproj = gemm(attn, self_attn.o_proj[layer]);
                    hidden_states = add(oproj, hidden_states);

                    let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                    let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                    let up = gemm(normed2, mlp.up_proj[layer]);
                    let down = gemm(gate * up, mlp.down_proj[layer]);
                    hidden_states = add(down, hidden_states);
                }}
            }}
            "#,
            hd = dims.hidden_size,
            id = dims.intermediate_size,
            hdm = dims.head_dim,
            nah = dims.num_attention_heads,
            nkh = dims.num_kv_heads,
            vs = dims.vocab_size,
        );
        let tokens: proc_macro2::TokenStream = src.parse().expect("hardcoded DSL tokenizes");
        let def: crate::parse::MegakernelDef = syn::parse2(tokens).expect("hardcoded DSL parses");
        let cfg = crate::cfg::build_cfg(&def);
        crate::fuf::build_fuf(&cfg, dims).expect("hardcoded DSL builds FUF")
    }

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

#[cfg(test)]
mod self_tests {
    use super::*;

    #[test]
    fn llama_forward_has_expected_node_count_per_layer() {
        // Per layer: 1 attn_norm, 3 qkv_gemm (q,k,v), 1 qkv_split,
        //            1 rope, 1 kv_cache_write, 1 attention, 1 o_proj,
        //            1 attn_residual, 1 mlp_norm, 1 gate_gemm,
        //            1 up_gemm, 1 gate_up_concat, 1 silu_mul,
        //            1 down_gemm, 1 mlp_residual = 17 nodes per layer.
        // Plus 1 synthetic input node for the very first layer.
        let g = TileGraph::build_llama_forward_1b(2);
        assert_eq!(g.nodes.len(), 1 + 2 * 17);
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

    /// Build a TileGraph from the LLaMA DSL and verify it has the
    /// same structure as the hardcoded build_llama_forward.
    #[test]
    fn from_model_dag_matches_hardcoded() {
        let dsl = r#"
            kernel llama_test<NL=2, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                    let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                    let attn = attention(q, k, v, kv_cache[layer], block_table);
                    let oproj = gemm(attn, self_attn.o_proj[layer]);
                    hidden_states = add(oproj, hidden_states);

                    let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                    let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                    let up = gemm(normed2, mlp.up_proj[layer]);
                    let down = gemm(gate * up, mlp.down_proj[layer]);
                    hidden_states = add(down, hidden_states);
                }
                let normed = rmsnorm(hidden_states, lm_head_norm);
                logits = gemm(normed, lm_head);
            }
        "#;
        let tokens: proc_macro2::TokenStream = dsl.parse().unwrap();
        let def: crate::parse::MegakernelDef = syn::parse2(tokens).unwrap();
        let cfg = crate::cfg::build_cfg(&def);

        let from_cfg = crate::fuf::build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).unwrap();
        let hardcoded = TileGraph::build_llama_forward_1b(2);

        // from_cfg includes the post-loop (rmsnorm + lm_head) ops
        // tagged with layer=num_layers; the hardcoded
        // build_llama_forward_1b doesn't emit them. Count only
        // in-loop nodes for comparison.
        let in_loop = |tg: &TileGraph, num_layers: u16| -> usize {
            tg.nodes.iter().filter(|n| n.layer < num_layers).count()
        };
        assert_eq!(
            in_loop(&from_cfg, 2),
            in_loop(&hardcoded, 2),
            "node count mismatch: from_cfg={}, hardcoded={}",
            in_loop(&from_cfg, 2),
            in_loop(&hardcoded, 2),
        );

        // Same tile kinds per layer.
        for layer in 0..2u16 {
            let cfg_kinds: Vec<_> = from_cfg
                .nodes
                .iter()
                .filter(|n| n.layer == layer)
                .map(|n| n.kind)
                .collect();
            let hc_kinds: Vec<_> = hardcoded
                .nodes
                .iter()
                .filter(|n| n.layer == layer)
                .map(|n| n.kind)
                .collect();
            assert_eq!(
                cfg_kinds, hc_kinds,
                "layer {layer} tile kinds differ:\n  from_cfg: {cfg_kinds:?}\n  hardcoded: {hc_kinds:?}"
            );
        }
    }

    /// Verify the DAG-built TileGraph passes topological order invariant.
    #[test]
    fn from_model_dag_topo_order() {
        let dsl = r#"
            kernel test<NL=3, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    let q = gemm(normed, self_attn.q_proj[layer]);
                let k = gemm(normed, self_attn.k_proj[layer]);
                let v = gemm(normed, self_attn.v_proj[layer]);
                    let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                    let attn = attention(q, k, v, kv_cache[layer], block_table);
                    let oproj = gemm(attn, self_attn.o_proj[layer]);
                    hidden_states = add(oproj, hidden_states);

                    let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                    let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                    let up = gemm(normed2, mlp.up_proj[layer]);
                    let down = gemm(gate * up, mlp.down_proj[layer]);
                    hidden_states = add(down, hidden_states);
                }
                let normed = rmsnorm(hidden_states, lm_head_norm);
                logits = gemm(normed, lm_head);
            }
        "#;
        let tokens: proc_macro2::TokenStream = dsl.parse().unwrap();
        let def: crate::parse::MegakernelDef = syn::parse2(tokens).unwrap();
        let cfg = crate::cfg::build_cfg(&def);
        let g = crate::fuf::build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).unwrap();

        for node in &g.nodes {
            for dep in &node.deps {
                assert!(
                    dep.0 < node.id.0,
                    "tile {:?} depends on {:?} which appears later",
                    node.id,
                    dep
                );
            }
        }
    }

    /// LLaMA DSL using explicit `add` instead of `gemm_add`.
    /// The tile graph should produce the same ResidualAdd tiles.
    #[test]
    fn llama_with_explicit_add() {
        let dsl = r#"
            kernel llama_add<NL=2, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    let q = gemm(normed, self_attn.q_proj[layer]);
                    let k = gemm(normed, self_attn.k_proj[layer]);
                    let v = gemm(normed, self_attn.v_proj[layer]);
                    let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                    let attn = attention(q, k, v, kv_cache[layer], block_table);
                    let oproj = gemm(attn, self_attn.o_proj[layer]);
                    hidden_states = add(oproj, hidden_states);

                    let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                    let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                    let up = gemm(normed2, mlp.up_proj[layer]);
                    let down = gemm(gate * up, mlp.down_proj[layer]);
                    hidden_states = add(down, hidden_states);
                }
                logits = gemm(hidden_states, lm_head);
            }
        "#;
        let tokens: proc_macro2::TokenStream = dsl.parse().unwrap();
        let def: crate::parse::MegakernelDef = syn::parse2(tokens).unwrap();
        let cfg = crate::cfg::build_cfg(&def);
        let tg = crate::fuf::build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).unwrap();

        // Should have ResidualAdd tiles (from explicit add ops).
        let residuals: Vec<_> = tg.tiles_of_kind(TileKind::ResidualAdd).collect();
        assert!(
            residuals.len() >= 4, // 2 per layer × 2 layers (oproj + down)
            "expected ≥4 ResidualAdd tiles, got {}",
            residuals.len()
        );

        // Should still have the same GEMM phases.
        assert!(tg.tiles_of_kind(TileKind::GemmOProj).count() >= 2);
        assert!(tg.tiles_of_kind(TileKind::GemmDown).count() >= 2);
        assert!(tg.tiles_of_kind(TileKind::Attention).count() >= 2);
    }

    /// Gemma2 DSL body — 4 norms per layer, explicit residual adds.
    #[test]
    fn gemma2_tile_graph() {
        let dsl = r#"
            kernel gemma2<NL=2, HD=2048, ID=16384, HDM=256, NAH=8, NKH=4, VS=256000> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    let q = gemm(normed, self_attn.q_proj[layer]);
                    let k = gemm(normed, self_attn.k_proj[layer]);
                    let v = gemm(normed, self_attn.v_proj[layer]);
                    let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                    let attn = attention(q, k, v, kv_cache[layer], block_table);
                    let attn_out = gemm(attn, self_attn.o_proj[layer]);
                    let attn_normed = rmsnorm(attn_out, post_attention_layernorm[layer]);
                    hidden_states = add(attn_normed, hidden_states);
                    let ff_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer]);
                    let gate = silu(gemm(ff_normed, mlp.gate_proj[layer]));
                    let up = gemm(ff_normed, mlp.up_proj[layer]);
                    let mlp_out = gemm(gate * up, mlp.down_proj[layer]);
                    hidden_states = rmsnorm(mlp_out, post_feedforward_layernorm[layer]);
                }
                logits = gemm(hidden_states, lm_head);
            }
        "#;
        let tokens: proc_macro2::TokenStream = dsl.parse().unwrap();
        let def: crate::parse::MegakernelDef = syn::parse2(tokens).unwrap();
        let cfg = crate::cfg::build_cfg(&def);
        let tg = crate::fuf::build_fuf(
            &cfg,
            ModelDims {
                hidden_size: 2048,
                intermediate_size: 16384,
                num_attention_heads: 8,
                num_kv_heads: 4,
                head_dim: 256,
                vocab_size: 256000,
            },
        )
        .unwrap();

        // 4 norms per layer × 2 layers = 8 RmsNorm tiles.
        let norms: Vec<_> = tg.tiles_of_kind(TileKind::RmsNorm).collect();
        assert_eq!(
            norms.len(),
            8,
            "expected 8 RmsNorm tiles (4 per layer × 2 layers), got {}",
            norms.len()
        );

        // 1 explicit add per layer (after post_attention_layernorm).
        let residuals: Vec<_> = tg
            .tiles_of_kind(TileKind::ResidualAdd)
            .filter(|n| n.layer < tg.num_layers)
            .collect();
        assert!(
            residuals.len() >= 2,
            "expected ≥2 ResidualAdd tiles, got {}",
            residuals.len()
        );

        // Attention + OProj + Gate + Up + Down GEMMs present.
        assert_eq!(tg.tiles_of_kind(TileKind::Attention).count(), 2);
        assert_eq!(tg.tiles_of_kind(TileKind::GemmOProj).count(), 2);
        assert_eq!(tg.tiles_of_kind(TileKind::GemmGate).count(), 2);
        assert_eq!(tg.tiles_of_kind(TileKind::GemmDown).count(), 2);

        // Post-loop: lm_head.
        assert_eq!(tg.tiles_of_kind(TileKind::GemmLmHead).count(), 1);
    }

    /// FUF: fully unrolled forward — no layers, just flat tiles.
    #[test]
    fn fuf_has_cross_layer_edges() {
        let dsl = r#"
            kernel llama_fuf<NL=3, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
                hidden_states = embed(input_ids, embed_tokens);
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, input_layernorm[layer]);
                    let q = gemm(normed, self_attn.q_proj[layer]);
                    let k = gemm(normed, self_attn.k_proj[layer]);
                    let v = gemm(normed, self_attn.v_proj[layer]);
                    let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                    let attn = attention(q, k, v, kv_cache[layer], block_table);
                    let oproj = gemm(attn, self_attn.o_proj[layer]);
                    hidden_states = add(oproj, hidden_states);

                    let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                    let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
                    let up = gemm(normed2, mlp.up_proj[layer]);
                    let down = gemm(gate * up, mlp.down_proj[layer]);
                    hidden_states = add(down, hidden_states);
                }
                hidden_states = rmsnorm(hidden_states, norm);
                logits = gemm(hidden_states, lm_head);
            }
        "#;
        let tokens: proc_macro2::TokenStream = dsl.parse().unwrap();
        let def: crate::parse::MegakernelDef = syn::parse2(tokens).unwrap();
        let cfg = crate::cfg::build_cfg(&def);

        let fuf = crate::fuf::build_fuf(&cfg, ModelDims::LLAMA_3_2_1B).unwrap();

        // 3 layers of tiles + embed + final norm + lm_head.
        // Each layer has ~17 tiles (same as from_model_dag).
        // Plus embed (1) + final norm (1) + lm_head (1) = 3.
        let per_layer = fuf.nodes.iter().filter(|n| n.layer == 0).count();
        assert!(
            per_layer > 10,
            "expected >10 tiles for layer 0, got {per_layer}"
        );

        let total = fuf.nodes.len();
        assert!(
            total > 40,
            "expected >40 total tiles (3 layers + pre/post), got {total}"
        );

        // Topological order: every dep has a lower id than its consumer.
        for node in &fuf.nodes {
            for dep in &node.deps {
                assert!(
                    dep.0 < node.id.0,
                    "tile {:?} depends on {:?} which appears later",
                    node.id,
                    dep
                );
            }
        }

        // Cross-layer edges: layer 1's first tile should have a dep
        // on a layer 0 tile. This proves unrolling created cross-layer edges.
        let layer1_tiles: Vec<_> = fuf.nodes.iter().filter(|n| n.layer == 1).collect();
        assert!(!layer1_tiles.is_empty(), "no layer 1 tiles");
        let first_layer1 = layer1_tiles[0];
        let has_cross_layer_dep = first_layer1
            .deps
            .iter()
            .any(|d| fuf.nodes[d.0 as usize].layer == 0);
        assert!(
            has_cross_layer_dep,
            "layer 1's first tile should depend on a layer 0 tile"
        );

        // Weight names should be set for norm and gemm tiles.
        let norms_with_weight: Vec<_> = fuf
            .nodes
            .iter()
            .filter(|n| n.kind == TileKind::RmsNorm && n.weight_name.is_some())
            .collect();
        // 2 norms per layer × 3 layers + 1 final norm = 7
        assert!(
            norms_with_weight.len() >= 7,
            "expected ≥7 norms with weight_name, got {}",
            norms_with_weight.len()
        );

        // Post-loop lm_head should be present.
        assert_eq!(fuf.tiles_of_kind(TileKind::GemmLmHead).count(), 1);
    }
}
