// SPDX-License-Identifier: Apache-2.0
//! FUF → Stencil IR lowering, v1.
//!
//! Consumes the solver's `Assignment` (tile → subgraph → Impl) and
//! produces a `Megakernel` of stencil `Region`s. For v1 we handle
//! the two attention impls — `AttentionPrefillContiguousImpl` and
//! `AttentionViaCacheImpl` — which both claim a singleton
//! `OpKind::Attention` tile. Everything else is out of scope and
//! returns an error: the point of this commit is to prove the
//! pipeline connects end-to-end on the target stress case, not to
//! replace codegen.
//!
//! The lowering is pattern-match on `Implementation::name()`. Param
//! inference (head_dim, tile_q, tile_k, pipe) is passed in by the
//! caller for now; shape-driven inference from the FUF arrives when
//! we extend beyond singleton attention.
//!
//! Integration: slot is between `solver::solve()` and `codegen::emit_model()`
//! in `lib.rs`. This commit adds the pass itself + unit tests; the
//! wire-up into the macro drive is deferred to the next commit,
//! because changing codegen to consume `Megakernel` is a bigger
//! edit than the lowering itself.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use ferrite_stencil::{
    AttnParams, ControlEdge, DepKind, EmbedParams, GateUpSiluMulParams, GemmParams, Megakernel,
    PagedDecodeParams, QkvRopeParams, Region, RegionId, ResidualAddParams, RmsNormParams,
    UnaryInplaceParams, Window, attn_region, attn_region_paged_decode, embed_region,
    gate_up_silu_mul_region, gemm_region, qkv_rope_region, residual_add_region, rmsnorm_region,
    unary_inplace_region,
};

use crate::classified::ExternKind;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::ImplementationLibrary;
use crate::solver::{Assignment, SubgraphId};

/// Outcome of a tolerant pass over an `Assignment`: every subgraph
/// whose Impl has a region template is lowered; every subgraph whose
/// Impl does not is skipped (not an error). Used by the parallel
/// wire-up in `lib.rs` so the new pipeline runs on real models
/// without blocking on templates for every non-attention Impl.
#[derive(Debug)]
pub struct LowerReport {
    pub mk: Megakernel,
    pub supported_subgraphs: usize,
    /// Subgraphs that had no region template, with the offending
    /// impl name for telemetry.
    pub skipped: Vec<(SubgraphId, &'static str)>,
}

/// Hardware-facing tile/pipe parameters the lowering can't yet
/// infer from the FUF. Pass-through for v1; later commits will
/// derive these from FUF shapes + solver cost hints.
#[derive(Clone, Copy, Debug)]
pub struct LowerHints {
    // Attention.
    pub head_dim: u32,
    pub num_head_groups: u32,
    pub tile_q: u32,
    pub tile_k: u32,
    pub pipe: u32,
    pub tokens_per_page: u32,
    // GEMM tile dims — picked once per lowering for all projections.
    // Refined per-Impl once calibrated tile sizes land on the Impl.
    pub gemm_m_tile: u32,
    pub gemm_n_tile: u32,
    pub gemm_k_tile: u32,
    // Token-wise element-wise ops (rmsnorm, residual add). `hidden_dim`
    // reaches here via `from_model_bounds`; token_tile is pass-through.
    pub hidden_dim: u32,
    pub token_tile: u32,
    // MLP: intermediate dim and the inter-axis tile. `intermediate_dim`
    // comes from bounds["intermediate_size"] via from_model_bounds.
    pub intermediate_dim: u32,
    pub inter_tile: u32,
}

impl Default for LowerHints {
    fn default() -> Self {
        Self {
            head_dim: 128,
            num_head_groups: 8,
            tile_q: 128,
            tile_k: 64,
            pipe: 3,
            tokens_per_page: 256,
            gemm_m_tile: 128,
            gemm_n_tile: 128,
            gemm_k_tile: 32,
            hidden_dim: 4096,
            token_tile: 64,
            intermediate_dim: 14336,
            inter_tile: 128,
        }
    }
}

impl LowerHints {
    /// Pull the two bounds-derivable fields (`head_dim`,
    /// `num_head_groups` = num_q_heads / num_kv_heads) off the
    /// model's bounds table. The remaining fields — `tile_q`,
    /// `tile_k`, `pipe`, `tokens_per_page` — still come from
    /// `Default`; they belong on the Impl (tile tuning) and on the
    /// KvCachePool config respectively. See `STENCIL_IR_STATUS.md`
    /// item #2 for the remaining work.
    pub fn from_model_bounds(bounds: &std::collections::BTreeMap<String, u64>) -> Self {
        let default = Self::default();
        let head_dim = bounds
            .get("head_dim")
            .copied()
            .map(|v| v as u32)
            .unwrap_or(default.head_dim);
        let num_q = bounds.get("num_attention_heads").copied();
        let num_kv = bounds.get("num_key_value_heads").copied();
        let num_head_groups = match (num_q, num_kv) {
            (Some(q), Some(kv)) if kv > 0 => ((q / kv).max(1)) as u32,
            _ => default.num_head_groups,
        };
        let hidden_dim = bounds
            .get("hidden_size")
            .copied()
            .map(|v| v as u32)
            .unwrap_or(default.hidden_dim);
        let intermediate_dim = bounds
            .get("intermediate_size")
            .copied()
            .map(|v| v as u32)
            .unwrap_or(default.intermediate_dim);
        Self {
            head_dim,
            num_head_groups,
            hidden_dim,
            intermediate_dim,
            ..default
        }
    }
}

#[derive(Debug)]
pub enum LowerError {
    UnsupportedImpl { name: &'static str },
    EmptyAssignment,
    SubgraphWithoutTiles { sg: SubgraphId },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedImpl { name } => {
                write!(f, "lowering has no region template for impl {:?}", name)
            }
            LowerError::EmptyAssignment => write!(f, "assignment has no subgraphs"),
            LowerError::SubgraphWithoutTiles { sg } => {
                write!(f, "subgraph {:?} claims no tiles", sg)
            }
        }
    }
}

impl std::error::Error for LowerError {}

pub fn lower_assignment(
    fuf: &Fuf,
    assignment: &Assignment,
    library: &ImplementationLibrary,
    hints: &LowerHints,
) -> Result<Megakernel, LowerError> {
    if assignment.num_subgraphs() == 0 {
        return Err(LowerError::EmptyAssignment);
    }

    let mut subgraphs: Vec<SubgraphId> = assignment.subgraphs().collect();
    subgraphs.sort();

    let mut regions: Vec<Region> = Vec::new();
    let mut sg_to_rid: HashMap<SubgraphId, RegionId> = HashMap::new();

    for sg in subgraphs {
        let tiles = assignment.tiles_in_subgraph(sg);
        if tiles.is_empty() {
            return Err(LowerError::SubgraphWithoutTiles { sg });
        }
        let impl_id = assignment
            .impl_of(sg)
            .expect("subgraph with tiles has impl");
        let imp = library.get(impl_id);
        let mut region = lower_impl(imp.name(), &tiles, fuf, hints)?;
        stamp_gmem_bindings(&mut region, imp.name(), fuf, &tiles);
        let rid = regions.len() as RegionId;
        region.id = rid;
        sg_to_rid.insert(sg, rid);
        regions.push(region);
    }

    let control = derive_control_edges(fuf, assignment, &sg_to_rid);

    Ok(Megakernel { regions, control })
}

/// Tolerant variant: lowers every supported subgraph, records the
/// rest as skipped. Used by the macro drive while non-attention
/// region templates are still being written.
pub fn lower_assignment_partial(
    fuf: &Fuf,
    assignment: &Assignment,
    library: &ImplementationLibrary,
    hints: &LowerHints,
) -> LowerReport {
    let mut regions: Vec<Region> = Vec::new();
    let mut sg_to_rid: HashMap<SubgraphId, RegionId> = HashMap::new();
    let mut skipped: Vec<(SubgraphId, &'static str)> = Vec::new();
    let mut supported = 0usize;

    let mut subgraphs: Vec<SubgraphId> = assignment.subgraphs().collect();
    subgraphs.sort();

    for sg in subgraphs {
        let tiles = assignment.tiles_in_subgraph(sg);
        if tiles.is_empty() {
            continue;
        }
        let Some(impl_id) = assignment.impl_of(sg) else {
            continue;
        };
        let imp = library.get(impl_id);
        match lower_impl(imp.name(), &tiles, fuf, hints) {
            Ok(mut region) => {
                stamp_gmem_bindings(&mut region, imp.name(), fuf, &tiles);
                let rid = regions.len() as RegionId;
                region.id = rid;
                sg_to_rid.insert(sg, rid);
                regions.push(region);
                supported += 1;
            }
            Err(LowerError::UnsupportedImpl { name }) => {
                skipped.push((sg, name));
            }
            Err(_) => {}
        }
    }

    // Skipped subgraphs have no region, so ControlEdges touching them
    // are dropped (there's nothing to synchronize to/from). Telemetry
    // lives on `skipped` — the megakernel is simply incomplete until
    // every subgraph gets a region template.
    let control = derive_control_edges(fuf, assignment, &sg_to_rid);

    LowerReport {
        mk: Megakernel { regions, control },
        supported_subgraphs: supported,
        skipped,
    }
}

/// Walk the FUF's per-tile input edges; every dep that crosses a
/// subgraph boundary becomes one `ControlEdge` (deduped on
/// (src_region, dst_region)). DepKind::Barrier is the conservative
/// default.
///
/// If a subgraph on the dep path has no region (its Impl wasn't
/// templated — e.g. `reshape_ref`, which is purely structural), the
/// walker traverses through it and attributes the edge to the
/// nearest regioned ancestor. Otherwise reshape-only subgraphs would
/// silently break the dep chain between every region they connect.
fn derive_control_edges(
    fuf: &Fuf,
    assignment: &Assignment,
    sg_to_rid: &HashMap<SubgraphId, RegionId>,
) -> Vec<ControlEdge> {
    // Tile index for O(1) lookups during the BFS.
    let tile_to_node: HashMap<TileId, &crate::fuf::FufNode> =
        fuf.nodes.iter().map(|n| (n.id, n)).collect();
    let mut seen: HashSet<(RegionId, RegionId)> = HashSet::new();
    let mut out: Vec<ControlEdge> = Vec::new();

    for node in &fuf.nodes {
        let Some(dst_sg) = assignment.subgraph_of(node.id) else {
            continue;
        };
        let Some(&dst_rid) = sg_to_rid.get(&dst_sg) else {
            continue;
        };

        // BFS back through inputs. If we hit a regioned ancestor,
        // emit the edge and stop walking that path; if we hit a
        // skipped-subgraph ancestor, keep walking through its inputs.
        let mut stack: Vec<TileId> = Vec::new();
        let mut visited: HashSet<TileId> = HashSet::new();
        for input in &node.inputs {
            if let FufInput::Tile { id, .. } = input {
                stack.push(*id);
            }
        }
        while let Some(src_tile) = stack.pop() {
            if !visited.insert(src_tile) {
                continue;
            }
            let Some(src_sg) = assignment.subgraph_of(src_tile) else {
                continue;
            };
            if src_sg == dst_sg {
                continue;
            }
            if let Some(&src_rid) = sg_to_rid.get(&src_sg) {
                if seen.insert((src_rid, dst_rid)) {
                    out.push(ControlEdge {
                        src: src_rid,
                        dst: dst_rid,
                        kind: DepKind::Barrier,
                    });
                }
                // Regioned ancestor — the edge stops here; we don't
                // traverse further back, because upstream deps belong
                // to the ancestor's own inbound edges.
            } else {
                // Skipped ancestor — walk through it.
                if let Some(anc) = tile_to_node.get(&src_tile) {
                    for input in &anc.inputs {
                        if let FufInput::Tile { id, .. } = input {
                            stack.push(*id);
                        }
                    }
                }
            }
        }
    }
    out
}

fn lower_impl(
    impl_name: &'static str,
    _tiles: &[TileId],
    _fuf: &Fuf,
    hints: &LowerHints,
) -> Result<Region, LowerError> {
    match impl_name {
        // ── Attention ──
        "attention_prefill_contiguous" => Ok(attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: hints.head_dim,
            tile_q: hints.tile_q,
            tile_k: hints.tile_k,
            num_head_groups: hints.num_head_groups,
            pipe: hints.pipe,
        })),
        // `flashinfer_attention_decode` has the same stencil shape as
        // `attention_via_cache` — paged-KV decode. The Impl name
        // differs only because the current runtime dispatches a
        // pre-built FlashInfer kernel for it; from the Stencil IR's
        // POV they're the same region, and whether the emitted
        // megakernel inlines the compute or calls FlashInfer is an
        // emit_ops / resource-mapping concern, not a template one.
        "attention_via_cache" | "flashinfer_attention_decode" => {
            Ok(attn_region_paged_decode(&PagedDecodeParams {
                head_dim: hints.head_dim,
                tile_k: hints.tile_k,
                num_head_groups: hints.num_head_groups,
                pipe: hints.pipe,
                tokens_per_page: hints.tokens_per_page,
                window: Window::Infinite,
            }))
        }
        "sliding_attention_via_cache" => Ok(attn_region_paged_decode(&PagedDecodeParams {
            head_dim: hints.head_dim,
            tile_k: hints.tile_k,
            num_head_groups: hints.num_head_groups,
            pipe: hints.pipe,
            tokens_per_page: hints.tokens_per_page,
            window: Window::Finite(4096),
        })),
        "sliding_attention_prefill_contiguous" => Ok(attn_region(&AttnParams {
            window: Window::Finite(4096),
            head_dim: hints.head_dim,
            tile_q: hints.tile_q,
            tile_k: hints.tile_k,
            num_head_groups: hints.num_head_groups,
            pipe: hints.pipe,
        })),
        // ── GEMM-class (full-precision + quantized variants) ──
        // All lower to the same gemm_region stencil; the difference is
        // the Load's addressing for quantized weights, which belongs in
        // emit_ops's per-tag expansion, not in the region template.
        "fused_gemm_bias" | "gemm_ref" | "cutlass_gemv" | "marlin_gemm" | "bnb4_gemm" => {
            Ok(gemm_region(&GemmParams {
                m_tile: hints.gemm_m_tile,
                n_tile: hints.gemm_n_tile,
                k_tile: hints.gemm_k_tile,
                pipe: hints.pipe,
            }))
        }
        // ── RMSNorm (with and without a fused residual add) ──
        // The residual-add fuses in at the Compute node's expansion; the
        // stencil shape (1 parallel axis, straight-line) is identical.
        "fused_add_rms_norm"
        | "fused_add_rms_norm_with_offset"
        | "scalar_offset_rms_norm"
        | "rmsnorm_ref"
        | "layer_norm_ref" => Ok(rmsnorm_region(&RmsNormParams {
            hidden_dim: hints.hidden_dim,
            token_tile: hints.token_tile,
        })),
        // ── Element-wise add (standalone, not fused with a norm) ──
        "add_ref" => Ok(residual_add_region(&ResidualAddParams {
            hidden_dim: hints.hidden_dim,
            token_tile: hints.token_tile,
        })),
        // ── QKV projection + RoPE (prefill writes-out-direct; cache
        //    variant writes into the paged KV cache). qk_norm variant
        //    shares the same stencil shape — the extra norm lives in
        //    the Compute node's expansion, not the region template.
        //    Quantized variants (marlin_, bnb4_) all lower to the same
        //    stencil; the Load addressing differs only at emit_ops.
        "fused_qkv_rope_cache"
        | "fused_qkv_qk_norm_rope_cache"
        | "marlin_fused_qkv_rope_cache"
        | "bnb4_fused_qkv_rope_cache" => Ok(qkv_rope_region(&QkvRopeParams {
            hidden_dim: hints.hidden_dim,
            head_dim: hints.head_dim,
            num_q_heads: hints.num_head_groups.saturating_mul(1).max(1),
            num_kv_heads: 1,
            token_tile: hints.token_tile,
            k_tile: hints.gemm_k_tile,
            pipe: hints.pipe,
            writes_kv_cache: true,
        })),
        "fused_qkv_rope_prefill"
        | "marlin_fused_qkv_rope_prefill"
        | "bnb4_fused_qkv_rope_prefill" => Ok(qkv_rope_region(&QkvRopeParams {
            hidden_dim: hints.hidden_dim,
            head_dim: hints.head_dim,
            num_q_heads: hints.num_head_groups.saturating_mul(1).max(1),
            num_kv_heads: 1,
            token_tile: hints.token_tile,
            k_tile: hints.gemm_k_tile,
            pipe: hints.pipe,
            writes_kv_cache: false,
        })),
        // ── RoPE append only (no projection): tiny variant of the
        //    qkv_rope region with projection-GEMM nodes elided. For
        //    now share the full template — the emit_ops expansion
        //    for a qkv projection whose weights are Wq=I / Wk=I /
        //    Wv=I degenerates to a pass-through, and that's what the
        //    intrinsic should do for rope_append. Refinement into a
        //    dedicated, smaller template lands if this proves costly.
        "rope_append_ref" => Ok(qkv_rope_region(&QkvRopeParams {
            hidden_dim: hints.hidden_dim,
            head_dim: hints.head_dim,
            num_q_heads: hints.num_head_groups.saturating_mul(1).max(1),
            num_kv_heads: 1,
            token_tile: hints.token_tile,
            k_tile: hints.gemm_k_tile,
            pipe: hints.pipe,
            writes_kv_cache: true,
        })),
        // ── Gate+Up+SiLU/GeLU+Mul (MLP input; silu and gelu share
        //    stencil shape — activation picks at emit_ops). Quantized
        //    variants pass through unchanged.
        "fused_gate_up_silu_mul"
        | "fused_gate_up_gelu_mul"
        | "marlin_fused_gate_up_silu_mul"
        | "marlin_fused_gate_up_gelu_mul"
        | "bnb4_fused_gate_up_silu_mul"
        | "bnb4_fused_gate_up_gelu_mul" => Ok(gate_up_silu_mul_region(&GateUpSiluMulParams {
            hidden_dim: hints.hidden_dim,
            intermediate_dim: hints.intermediate_dim,
            token_tile: hints.token_tile,
            inter_tile: hints.inter_tile,
            k_tile: hints.gemm_k_tile,
            pipe: hints.pipe,
        })),
        // ── Unary in-place ops (single Load → Compute → Store). Same
        //    stencil shape; the op tag picks the intrinsic expansion.
        "scalar_mul_inplace" => Ok(unary_inplace_region(&UnaryInplaceParams {
            hidden_dim: hints.hidden_dim,
            token_tile: hints.token_tile,
            op_tag: "scalar_mul",
        })),
        "tanh_softcap_inplace" => Ok(unary_inplace_region(&UnaryInplaceParams {
            hidden_dim: hints.hidden_dim,
            token_tile: hints.token_tile,
            op_tag: "tanh_softcap",
        })),
        // ── Embedding lookup (gather-only; no Compute). ──
        "embed_ref" => Ok(embed_region(&EmbedParams {
            hidden_dim: hints.hidden_dim,
            token_tile: hints.token_tile,
        })),
        other => Err(LowerError::UnsupportedImpl { name: other }),
    }
}

// ─── gmem bindings (STENCIL_IR_STATUS.md item 4b) ─────────────────
//
// After `lower_impl` instantiates a template's Region, we walk the
// subgraph's FUF tile(s) to resolve each template canonical gmem name
// (from `emit_ops::gmem_refs`) to an FUF-derived unique identity.
// Effect: llama-3-8b's kernel signature grows from ~18 canonical
// pointers to per-layer-per-tensor identities — each layer's Wqkv,
// Wo, Wgate, Wup, Wdown, RMSnorm weights become distinct params, and
// each attention region reads from the specific upstream tile output
// its FUF edge points at. Identities dedupe on true FUF identity:
// two regions sharing an upstream tile share the pointer.
//
// Single-tile subgraphs only (the overwhelming common case for the
// Impls we ship templates for). Multi-tile subgraphs leave bindings
// empty; their canonical names still work as before and can be
// promoted in a follow-up when we hit an Impl that needs it.

fn stamp_gmem_bindings(region: &mut Region, impl_name: &'static str, fuf: &Fuf, tiles: &[TileId]) {
    // Collect external inputs = the subgraph's interface. Inputs that
    // reference other tiles in the SAME subgraph are internal edges
    // and don't appear as gmem pointers. For fused_qkv_rope_cache
    // (4 tiles: gemm → rope → kv_write, plus an intermediate) we
    // keep the upstream hidden-tile ref, the Wqkv weight, the rotary
    // extern, and the kv_cache extern; the intra-subgraph tile
    // connections drop out.
    let subgraph_set: std::collections::HashSet<TileId> = tiles.iter().copied().collect();
    let mut external_inputs: Vec<FufInput> = Vec::new();
    for t in tiles {
        let node = fuf.get(*t);
        for inp in &node.inputs {
            match inp {
                FufInput::Tile { id, .. } if subgraph_set.contains(id) => continue,
                _ => external_inputs.push(inp.clone()),
            }
        }
    }
    // Output tile: for multi-tile subgraphs the tile whose outputs
    // flow outside the subgraph is typically the last-declared one
    // (FUF unroll assigns TileIds in topo order). For single-tile
    // subgraphs it's just that tile.
    let output_tile_id = tiles.iter().map(|t| t.0).max().unwrap_or(0);
    let bindings = bindings_for_impl(impl_name, &external_inputs, output_tile_id);
    if std::env::var("FERRITE_STENCIL_4B_TRACE").is_ok() {
        eprintln!(
            "[4b] impl={} tiles={:?} ext_inputs={:?} -> bindings={:?}",
            impl_name,
            tiles.iter().map(|t| t.0).collect::<Vec<_>>(),
            external_inputs.iter().map(input_brief).collect::<Vec<_>>(),
            bindings,
        );
    }
    region.gmem_bindings = bindings;
}

fn input_brief(input: &FufInput) -> String {
    match input {
        FufInput::Tile { id, slot } => format!("tile({},{})", id.0, slot),
        FufInput::Weight { id, index, .. } => match index {
            Some(i) => format!("weight({},{})", id.0, i),
            None => format!("weight({},_)", id.0),
        },
        FufInput::Extern { kind, index } => {
            format!(
                "extern({:?},{})",
                kind,
                index.map_or("_".to_string(), |i| i.to_string())
            )
        }
        FufInput::Scalar(v) => format!("scalar({})", v),
    }
}

/// Per-Impl map of template canonical gmem names → FUF-derived
/// identities. The Impl name decides the FUF input layout (which slot
/// is A vs B, where the weight lives, whether there's a Rotary extern,
/// etc.). Output tensors identify via `t{tile_id}_{slot}` so
/// downstream regions that read from this tile get matching pointer
/// names.
fn bindings_for_impl(
    impl_name: &'static str,
    external_inputs: &[FufInput],
    output_tile_id: u32,
) -> Vec<(&'static str, &'static str)> {
    let own_output =
        |slot: u8| -> &'static str { leak_str(format!("t{}_{}", output_tile_id, slot)) };
    // Pick the `i`-th non-weight-non-extern external input — typically
    // a FufInput::Tile from upstream (the "data" input, e.g. hidden
    // state flowing into this subgraph). Used for A/X style mappings.
    let data_input = |i: usize| -> Option<&'static str> {
        external_inputs
            .iter()
            .filter(|inp| matches!(inp, FufInput::Tile { .. } | FufInput::Scalar(_)))
            .nth(i)
            .map(|inp| leak_str(fuf_input_identity(inp)))
    };
    let find_extern = |want: ExternKind| -> Option<&'static str> {
        external_inputs.iter().find_map(|inp| match inp {
            FufInput::Extern { kind, index } if *kind == want => Some(leak_str(format!(
                "x_{}_{}",
                extern_name(*kind),
                index.unwrap_or(0)
            ))),
            _ => None,
        })
    };
    let find_weight = |nth: usize| -> Option<&'static str> {
        let mut count = 0usize;
        for inp in external_inputs {
            if let FufInput::Weight { id, index, .. } = inp {
                if count == nth {
                    return Some(leak_str(match index {
                        Some(i) => format!("w{}_{}", id.0, i),
                        None => format!("w{}", id.0),
                    }));
                }
                count += 1;
            }
        }
        None
    };

    let mut b = Vec::new();
    match impl_name {
        "attention_prefill_contiguous" | "sliding_attention_prefill_contiguous" => {
            if let Some(q) = data_input(0) {
                b.push(("Q_gmem", q));
            }
            if let Some(k) = data_input(1) {
                b.push(("K_gmem", k));
            }
            if let Some(v) = data_input(2) {
                b.push(("V_gmem", v));
            }
            b.push(("O_gmem", own_output(0)));
        }
        "attention_via_cache" | "flashinfer_attention_decode" | "sliding_attention_via_cache" => {
            if let Some(q) = data_input(0) {
                b.push(("Q_gmem", q));
            }
            // K/V come from the paged KV cache; both canonical names
            // share the KvCache extern pointer — the runtime handles
            // the k vs v offset per page.
            if let Some(kv) = find_extern(ExternKind::KvCache) {
                b.push(("K_gmem", kv));
                b.push(("V_gmem", kv));
            }
            b.push(("O_gmem", own_output(0)));
        }
        "fused_gemm_bias" | "gemm_ref" | "cutlass_gemv" | "marlin_gemm" | "bnb4_gemm" => {
            if let Some(a) = data_input(0) {
                b.push(("A_gmem", a));
            }
            if let Some(w) = find_weight(0) {
                b.push(("B_gmem", w));
            }
            b.push(("C_gmem", own_output(0)));
        }
        "fused_add_rms_norm"
        | "fused_add_rms_norm_with_offset"
        | "scalar_offset_rms_norm"
        | "rmsnorm_ref"
        | "layer_norm_ref" => {
            if let Some(x) = data_input(0) {
                b.push(("X_gmem", x));
            }
            if let Some(w) = find_weight(0) {
                b.push(("W_gmem", w));
            }
            b.push(("Y_gmem", own_output(0)));
        }
        "add_ref" => {
            if let Some(a) = data_input(0) {
                b.push(("A_gmem", a));
            }
            if let Some(b1) = data_input(1) {
                b.push(("B_gmem", b1));
            }
            b.push(("Sum_gmem", own_output(0)));
        }
        "fused_qkv_rope_cache"
        | "fused_qkv_qk_norm_rope_cache"
        | "marlin_fused_qkv_rope_cache"
        | "bnb4_fused_qkv_rope_cache"
        | "rope_append_ref" => {
            if let Some(x) = data_input(0) {
                b.push(("X_gmem", x));
            }
            if let Some(w) = find_weight(0) {
                b.push(("Wqkv_gmem", w));
            }
            if let Some(r) = find_extern(ExternKind::Rotary) {
                b.push(("RopeCoef_gmem", r));
            }
            b.push(("Q_gmem", own_output(0)));
            b.push(("K_gmem", own_output(1)));
            b.push(("V_gmem", own_output(2)));
            if let Some(kv) = find_extern(ExternKind::KvCache) {
                b.push(("K_cache_gmem", kv));
                b.push(("V_cache_gmem", kv));
            }
        }
        "fused_qkv_rope_prefill"
        | "marlin_fused_qkv_rope_prefill"
        | "bnb4_fused_qkv_rope_prefill" => {
            if let Some(x) = data_input(0) {
                b.push(("X_gmem", x));
            }
            if let Some(w) = find_weight(0) {
                b.push(("Wqkv_gmem", w));
            }
            if let Some(r) = find_extern(ExternKind::Rotary) {
                b.push(("RopeCoef_gmem", r));
            }
            b.push(("Q_gmem", own_output(0)));
            b.push(("K_gmem", own_output(1)));
            b.push(("V_gmem", own_output(2)));
        }
        "fused_gate_up_silu_mul"
        | "fused_gate_up_gelu_mul"
        | "marlin_fused_gate_up_silu_mul"
        | "marlin_fused_gate_up_gelu_mul"
        | "bnb4_fused_gate_up_silu_mul"
        | "bnb4_fused_gate_up_gelu_mul" => {
            if let Some(x) = data_input(0) {
                b.push(("X_gmem", x));
            }
            if let Some(wg) = find_weight(0) {
                b.push(("Wgate_gmem", wg));
            }
            if let Some(wu) = find_weight(1) {
                b.push(("Wup_gmem", wu));
            }
            b.push(("Inter_gmem", own_output(0)));
        }
        "scalar_mul_inplace" | "tanh_softcap_inplace" => {
            // Unary in-place: the template reads X_gmem but has no
            // paired store_*_gmem tag, so there's only one gmem name
            // to bind. The output tensor identity is the same
            // pointer — in-place writes stay in place.
            if let Some(x) = data_input(0) {
                b.push(("X_gmem", x));
            }
        }
        "embed_ref" => {
            if let Some(ids) = find_extern(ExternKind::InputIds) {
                b.push(("TokenIds_gmem", ids));
            }
            if let Some(w) = find_weight(0) {
                b.push(("Embed_gmem", w));
            }
            b.push(("Y_gmem", own_output(0)));
        }
        _ => {} // Unknown Impl: canonical names stay (backward-compat).
    }
    b
}

fn fuf_input_identity(input: &FufInput) -> String {
    match input {
        FufInput::Tile { id, slot } => format!("t{}_{}", id.0, slot),
        FufInput::Weight { id, index, .. } => match index {
            Some(i) => format!("w{}_{}", id.0, i),
            None => format!("w{}", id.0),
        },
        FufInput::Extern { kind, index } => {
            format!("x_{}_{}", extern_name(*kind), index.unwrap_or(0))
        }
        // Scalar inline constants never appear as gmem pointers — they
        // go in kernel body expressions. An input slot holding a
        // scalar is a template/Impl mismatch; produce a distinct-enough
        // placeholder so identity dedupes don't collide.
        FufInput::Scalar(v) => format!("s_{:x}", v.to_bits()),
    }
}

fn extern_name(kind: ExternKind) -> &'static str {
    match kind {
        ExternKind::InputIds => "input_ids",
        ExternKind::Positions => "positions",
        ExternKind::Rotary => "rotary",
        ExternKind::RotaryLocal => "rotary_local",
        ExternKind::BlockTable => "block_table",
        ExternKind::KvCache => "kv_cache",
    }
}

/// Move a freshly-built identity string into a 'static slot. Called
/// only at proc-macro time (lowering runs once per user-macro
/// invocation), so the leak is one-shot and bounded by the model's
/// tensor count.
fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ferrite_stencil::ir;

    use super::*;
    use crate::classified::OpKind;
    use crate::fuf::{FufNode, TileId};
    use crate::impl_lib::{
        AttentionPrefillContiguousImpl, AttentionViaCacheImpl, ImplementationLibrary,
    };
    use crate::solver::{Assignment, SubgraphId};

    fn singleton_attention_fuf() -> Fuf {
        Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Attention,
                inputs: vec![],
                outputs: vec![],
            }],
        }
    }

    fn singleton_assignment(impl_id: crate::impl_lib::ImplId) -> Assignment {
        let sg = SubgraphId(0);
        let mut cover = HashMap::new();
        cover.insert(TileId(0), sg);
        let mut impls = HashMap::new();
        impls.insert(sg, impl_id);
        Assignment {
            cover,
            impls,
            predicted_us: 0.0,
        }
    }

    #[test]
    fn lowers_attention_prefill_to_fa2_region() {
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let assignment = singleton_assignment(impl_id);

        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default())
            .expect("lowering succeeds");

        assert_eq!(mk.regions.len(), 1);
        let r = &mk.regions[0];
        assert_eq!(r.name, "fa2_prefill");
        assert_eq!(r.nodes.len(), 7);
        ir::validate(r).expect("lowered region validates");

        // Matches the hand-authored template byte-for-byte on structure.
        let hand = attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        assert_eq!(r.nodes.len(), hand.nodes.len());
        assert_eq!(r.edges.len(), hand.edges.len());
        assert_eq!(r.domain.axes.len(), hand.domain.axes.len());
        assert_eq!(r.domain.predicates.len(), hand.domain.predicates.len());
    }

    #[test]
    fn lowers_attention_via_cache_to_paged_decode() {
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionViaCacheImpl));
        let assignment = singleton_assignment(impl_id);

        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default())
            .expect("lowering succeeds");
        assert_eq!(mk.regions.len(), 1);
        assert_eq!(mk.regions[0].name, "paged_decode");
        ir::validate(&mk.regions[0]).expect("decode region validates");
    }

    #[test]
    fn unsupported_impl_returns_error() {
        // Use the name of an impl we haven't templated yet to force
        // the error path. The impl itself doesn't need to exist in
        // the library for this test — we construct a tiny stub.
        use crate::impl_lib::Implementation;

        #[derive(Debug, Default)]
        struct FakeGemm;
        impl Implementation for FakeGemm {
            fn name(&self) -> &'static str {
                "gemm_rowmajor"
            }
            fn target_compatible(&self, _: &crate::target::TargetProfile) -> bool {
                true
            }
            fn workload_constraint(&self) -> crate::impl_lib::WorkloadConstraint {
                crate::impl_lib::WorkloadConstraint::Any
            }
            fn matches(
                &self,
                _: &Fuf,
                _: TileId,
                _: &crate::target::TargetProfile,
            ) -> Option<crate::impl_lib::MatchInfo> {
                None
            }
            fn cost_us(&self, _: &crate::impl_lib::MatchInfo, _: &crate::impl_lib::CostCtx) -> f64 {
                0.0
            }
            fn resources(&self, _: &crate::impl_lib::MatchInfo) -> crate::impl_lib::Resources {
                crate::impl_lib::Resources::ZERO
            }
            fn launch_kind(&self) -> crate::impl_lib::LaunchKind {
                crate::impl_lib::LaunchKind::HostCallback
            }
            fn supported_input_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn supported_output_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn input_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn output_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn is_compute_bound(&self) -> bool {
                false
            }
            fn emit_call(&self, _: &crate::emit::EmitCtx) -> proc_macro2::TokenStream {
                proc_macro2::TokenStream::new()
            }
        }

        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(FakeGemm));
        let assignment = singleton_assignment(impl_id);

        let err = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default())
            .expect_err("unsupported impl must fail");
        match err {
            LowerError::UnsupportedImpl { name } => {
                assert_eq!(name, "gemm_rowmajor");
            }
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn lowered_region_has_valid_structure_for_scheduling() {
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let assignment = singleton_assignment(impl_id);
        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default()).unwrap();
        let r = &mk.regions[0];

        // The axis classifier + topo-sort primitives run cleanly on
        // a lowered region — proves the output feeds the scheduler
        // substrate we already have.
        let classes = ferrite_stencil::classify_axes(r);
        assert_eq!(classes.len(), 3);
        assert_eq!(ferrite_stencil::region_pipeline_depth(r), 3);
        let order = ferrite_stencil::topo_order_within_iter(r);
        assert_eq!(order.len(), r.nodes.len());
    }

    #[test]
    fn lowers_gemm_impls_to_gemm_region() {
        use ferrite_stencil::ir;
        let fuf = Fuf { nodes: vec![] };
        let hints = LowerHints::default();
        for name in [
            "fused_gemm_bias",
            "cutlass_gemv",
            "marlin_gemm",
            "bnb4_gemm",
        ] {
            let r = lower_impl(name, &[], &fuf, &hints)
                .unwrap_or_else(|e| panic!("{} should lower: {:?}", name, e));
            assert_eq!(r.name, "gemm", "{}", name);
            ir::validate(&r).expect("gemm region validates");
            // gemm has M, N, K axes; K is the serial accumulation axis.
            assert_eq!(r.domain.axes.len(), 3);
            assert_eq!(r.nodes.len(), 4);
        }
    }

    #[test]
    fn lowers_rmsnorm_impls_to_rmsnorm_region() {
        use ferrite_stencil::ir;
        let fuf = Fuf { nodes: vec![] };
        let hints = LowerHints::default();
        for name in [
            "fused_add_rms_norm",
            "fused_add_rms_norm_with_offset",
            "scalar_offset_rms_norm",
        ] {
            let r = lower_impl(name, &[], &fuf, &hints)
                .unwrap_or_else(|e| panic!("{} should lower: {:?}", name, e));
            assert_eq!(r.name, "rmsnorm", "{}", name);
            ir::validate(&r).expect("rmsnorm region validates");
        }
    }

    #[test]
    fn populates_control_edges_from_fuf_deps() {
        // FUF with two tiles: tile1 consumes tile0's output. Both map
        // onto distinct subgraphs with an attention Impl (any impl that
        // has a region template works for this test). Expect one
        // ControlEdge(0 → 1, Barrier) in the emitted Megakernel.
        use crate::impl_lib::AttentionPrefillContiguousImpl;
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Attention,
                    inputs: vec![],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Attention,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let mut cover = HashMap::new();
        cover.insert(TileId(0), SubgraphId(0));
        cover.insert(TileId(1), SubgraphId(1));
        let mut impls = HashMap::new();
        impls.insert(SubgraphId(0), impl_id);
        impls.insert(SubgraphId(1), impl_id);
        let assignment = Assignment {
            cover,
            impls,
            predicted_us: 0.0,
        };

        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default()).unwrap();
        assert_eq!(mk.regions.len(), 2);
        assert_eq!(mk.regions[0].id, 0);
        assert_eq!(mk.regions[1].id, 1);
        assert_eq!(mk.control.len(), 1, "one inter-subgraph edge");
        assert_eq!(mk.control[0].src, 0);
        assert_eq!(mk.control[0].dst, 1);
        assert!(matches!(
            mk.control[0].kind,
            ferrite_stencil::DepKind::Barrier
        ));
    }

    #[test]
    fn control_edges_traverse_skipped_subgraphs() {
        // FUF: A → S (skipped impl: no region template) → B. Both
        // A and B have regioned subgraphs. Expect a direct A → B
        // ControlEdge — the skipped S must not silently break the
        // dep chain, since that'd misrepresent the megakernel.
        use crate::impl_lib::{AttentionPrefillContiguousImpl, Implementation};

        #[derive(Debug, Default)]
        struct SkippedImpl;
        impl Implementation for SkippedImpl {
            fn name(&self) -> &'static str {
                "reshape_ref"
            }
            fn target_compatible(&self, _: &crate::target::TargetProfile) -> bool {
                true
            }
            fn workload_constraint(&self) -> crate::impl_lib::WorkloadConstraint {
                crate::impl_lib::WorkloadConstraint::Any
            }
            fn matches(
                &self,
                _: &Fuf,
                _: TileId,
                _: &crate::target::TargetProfile,
            ) -> Option<crate::impl_lib::MatchInfo> {
                None
            }
            fn cost_us(&self, _: &crate::impl_lib::MatchInfo, _: &crate::impl_lib::CostCtx) -> f64 {
                0.0
            }
            fn resources(&self, _: &crate::impl_lib::MatchInfo) -> crate::impl_lib::Resources {
                crate::impl_lib::Resources::ZERO
            }
            fn launch_kind(&self) -> crate::impl_lib::LaunchKind {
                crate::impl_lib::LaunchKind::HostCallback
            }
            fn supported_input_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn supported_output_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn input_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn output_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn is_compute_bound(&self) -> bool {
                false
            }
            fn emit_call(&self, _: &crate::emit::EmitCtx) -> proc_macro2::TokenStream {
                proc_macro2::TokenStream::new()
            }
        }

        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Attention,
                    inputs: vec![],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Attention,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(2),
                    op: OpKind::Attention,
                    inputs: vec![FufInput::Tile {
                        id: TileId(1),
                        slot: 0,
                    }],
                    outputs: vec![],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let attn_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let skip_id = lib.push(Box::new(SkippedImpl));
        let mut cover = HashMap::new();
        cover.insert(TileId(0), SubgraphId(0));
        cover.insert(TileId(1), SubgraphId(1));
        cover.insert(TileId(2), SubgraphId(2));
        let mut impls = HashMap::new();
        impls.insert(SubgraphId(0), attn_id);
        impls.insert(SubgraphId(1), skip_id);
        impls.insert(SubgraphId(2), attn_id);
        let assignment = Assignment {
            cover,
            impls,
            predicted_us: 0.0,
        };

        let report = lower_assignment_partial(&fuf, &assignment, &lib, &LowerHints::default());
        assert_eq!(report.mk.regions.len(), 2, "A and B; S skipped");
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.skipped[0].1, "reshape_ref");
        // Direct A → B edge through S.
        assert_eq!(report.mk.control.len(), 1);
        assert_eq!(report.mk.control[0].src, 0);
        assert_eq!(report.mk.control[0].dst, 1);
    }

    #[test]
    fn dedupes_multiple_tile_edges_between_same_subgraphs() {
        // Two tiles in SG1 each depend on the same tile in SG0 → still
        // one ControlEdge. Dedupe is (src_region, dst_region)-keyed.
        use crate::impl_lib::AttentionPrefillContiguousImpl;
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Attention,
                    inputs: vec![],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Attention,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(2),
                    op: OpKind::Attention,
                    inputs: vec![FufInput::Tile {
                        id: TileId(0),
                        slot: 0,
                    }],
                    outputs: vec![],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let mut cover = HashMap::new();
        cover.insert(TileId(0), SubgraphId(0));
        cover.insert(TileId(1), SubgraphId(1));
        cover.insert(TileId(2), SubgraphId(1));
        let mut impls = HashMap::new();
        impls.insert(SubgraphId(0), impl_id);
        impls.insert(SubgraphId(1), impl_id);
        let assignment = Assignment {
            cover,
            impls,
            predicted_us: 0.0,
        };

        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default()).unwrap();
        assert_eq!(mk.regions.len(), 2);
        assert_eq!(
            mk.control.len(),
            1,
            "(0,1) edge deduped across two tile-tile edges"
        );
    }

    #[test]
    fn lowers_add_ref_to_residual_add_region() {
        use ferrite_stencil::ir;
        let fuf = Fuf { nodes: vec![] };
        let hints = LowerHints::default();
        let r = lower_impl("add_ref", &[], &fuf, &hints).expect("add_ref lowers");
        assert_eq!(r.name, "residual_add");
        ir::validate(&r).expect("residual_add region validates");
    }

    #[test]
    fn hints_from_bounds_derives_hidden_size() {
        use std::collections::BTreeMap;
        let mut b = BTreeMap::new();
        b.insert("hidden_size".into(), 8192);
        let h = LowerHints::from_model_bounds(&b);
        assert_eq!(h.hidden_dim, 8192);
    }

    #[test]
    fn hints_from_bounds_derives_head_dim_and_gqa_groups() {
        use std::collections::BTreeMap;

        // Llama-2-7B: MHA, 32 Q heads = 32 KV heads → groups = 1.
        let mut mha = BTreeMap::new();
        mha.insert("head_dim".into(), 128);
        mha.insert("num_attention_heads".into(), 32);
        mha.insert("num_key_value_heads".into(), 32);
        let h = LowerHints::from_model_bounds(&mha);
        assert_eq!(h.head_dim, 128);
        assert_eq!(h.num_head_groups, 1);

        // Llama-3-70B: GQA 64 Q / 8 KV → groups = 8.
        let mut gqa = BTreeMap::new();
        gqa.insert("head_dim".into(), 128);
        gqa.insert("num_attention_heads".into(), 64);
        gqa.insert("num_key_value_heads".into(), 8);
        let h = LowerHints::from_model_bounds(&gqa);
        assert_eq!(h.num_head_groups, 8);

        // Missing bounds fall back to Default without panicking.
        let empty = BTreeMap::new();
        let h = LowerHints::from_model_bounds(&empty);
        assert_eq!(h.head_dim, LowerHints::default().head_dim);
        assert_eq!(h.num_head_groups, LowerHints::default().num_head_groups);
    }

    #[test]
    fn partial_lowering_skips_unsupported_subgraphs() {
        // Assignment mixes one supported (attention) and one unsupported
        // (fake gemm) subgraph. Partial lowering should lower the
        // attention one and record the other as skipped.
        use crate::impl_lib::Implementation;

        #[derive(Debug, Default)]
        struct FakeGemm;
        impl Implementation for FakeGemm {
            fn name(&self) -> &'static str {
                "gemm_rowmajor"
            }
            fn target_compatible(&self, _: &crate::target::TargetProfile) -> bool {
                true
            }
            fn workload_constraint(&self) -> crate::impl_lib::WorkloadConstraint {
                crate::impl_lib::WorkloadConstraint::Any
            }
            fn matches(
                &self,
                _: &Fuf,
                _: TileId,
                _: &crate::target::TargetProfile,
            ) -> Option<crate::impl_lib::MatchInfo> {
                None
            }
            fn cost_us(&self, _: &crate::impl_lib::MatchInfo, _: &crate::impl_lib::CostCtx) -> f64 {
                0.0
            }
            fn resources(&self, _: &crate::impl_lib::MatchInfo) -> crate::impl_lib::Resources {
                crate::impl_lib::Resources::ZERO
            }
            fn launch_kind(&self) -> crate::impl_lib::LaunchKind {
                crate::impl_lib::LaunchKind::HostCallback
            }
            fn supported_input_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn supported_output_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn input_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn output_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn is_compute_bound(&self) -> bool {
                false
            }
            fn emit_call(&self, _: &crate::emit::EmitCtx) -> proc_macro2::TokenStream {
                proc_macro2::TokenStream::new()
            }
        }

        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Attention,
                    inputs: vec![],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Attention,
                    inputs: vec![],
                    outputs: vec![],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let attn_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let gemm_id = lib.push(Box::new(FakeGemm));
        let sg_attn = SubgraphId(0);
        let sg_gemm = SubgraphId(1);
        let mut cover = HashMap::new();
        cover.insert(TileId(0), sg_attn);
        cover.insert(TileId(1), sg_gemm);
        let mut impls = HashMap::new();
        impls.insert(sg_attn, attn_id);
        impls.insert(sg_gemm, gemm_id);
        let assignment = Assignment {
            cover,
            impls,
            predicted_us: 0.0,
        };

        let report = lower_assignment_partial(&fuf, &assignment, &lib, &LowerHints::default());
        assert_eq!(report.supported_subgraphs, 1);
        assert_eq!(report.mk.regions.len(), 1);
        assert_eq!(report.mk.regions[0].name, "fa2_prefill");
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.skipped[0].1, "gemm_rowmajor");
    }

    #[test]
    fn end_to_end_lower_and_schedule_on_sm90() {
        // Capstone: prove lowering's output feeds the wavefront
        // scheduler, so the whole pipeline from solver Assignment to
        // per-CTA preamble/body/epilogue works without a hand-authored
        // Region in the middle.
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let assignment = singleton_assignment(impl_id);
        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default()).unwrap();

        let region = &mk.regions[0];
        let sched = ferrite_stencil::schedule_wavefront(region, &ferrite_stencil::sm90_fa2())
            .expect("schedule succeeds on lowered region");

        assert_eq!(sched.pipeline_depth, 3);
        assert_eq!(sched.preamble.len(), 1, "preamble = load_q");
        assert_eq!(sched.body.len(), 5, "body = load_k, load_v, qk, sm, pv");
        assert_eq!(sched.epilogue.len(), 1, "epilogue = store_o");

        let tag = |n: u16| region.nodes[n as usize].op.tag;
        assert_eq!(tag(sched.preamble[0].node), "load_q_tile");
        assert_eq!(tag(sched.epilogue[0].node), "store_o_tile");

        // Pipeline loads in body carry iter_offset = P.
        for step in &sched.body {
            let t = tag(step.node);
            let expected = if t == "load_k_tile" || t == "load_v_tile" {
                3
            } else {
                0
            };
            assert_eq!(
                step.iter_offset, expected,
                "step {} iter_offset mismatch",
                t
            );
        }
    }

    // ── gmem binding stamping (item 4b) ────────────────────────────

    #[test]
    fn stamps_attention_inputs_and_output_identities() {
        // Build a FUF where an attention tile has three upstream tile
        // inputs (Q, K, V slots) — stamping should bind each canonical
        // gmem name to the corresponding FufInput identity, plus the
        // output O_gmem to `t{attn_tile}_0`.
        use ferrite_stencil::ir;
        let dummy = |id: u32| FufNode {
            id: TileId(id),
            op: OpKind::Attention,
            inputs: vec![],
            outputs: vec![],
        };
        let fuf = Fuf {
            nodes: vec![
                dummy(0),
                dummy(1),
                dummy(2),
                dummy(3),
                dummy(4),
                FufNode {
                    id: TileId(5),
                    op: OpKind::Attention,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(2),
                            slot: 0,
                        },
                        FufInput::Tile {
                            id: TileId(2),
                            slot: 1,
                        },
                        FufInput::Tile {
                            id: TileId(2),
                            slot: 2,
                        },
                    ],
                    outputs: vec![],
                },
            ],
        };
        let mut region = super::attn_region(&ferrite_stencil::AttnParams {
            window: ferrite_stencil::Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        stamp_gmem_bindings(
            &mut region,
            "attention_prefill_contiguous",
            &fuf,
            &[TileId(5)],
        );
        let b = &region.gmem_bindings;
        assert!(b.iter().any(|(c, id)| *c == "Q_gmem" && *id == "t2_0"));
        assert!(b.iter().any(|(c, id)| *c == "K_gmem" && *id == "t2_1"));
        assert!(b.iter().any(|(c, id)| *c == "V_gmem" && *id == "t2_2"));
        assert!(b.iter().any(|(c, id)| *c == "O_gmem" && *id == "t5_0"));
        ir::validate(&region).unwrap();
    }

    #[test]
    fn stamps_multi_tile_fused_qkv_rope_cache() {
        // Simulate the 4-tile fused_qkv_rope_cache shape llama uses:
        // tile 10 (gemm X⋅Wqkv), tile 11 (rope+split), tile 12 (kv cache
        // write), tile 13 (trailing). External inputs = X (upstream
        // tile 9), Wqkv weight, Rotary extern, KvCache extern.
        // Intra-subgraph tile-tile edges must be dropped.
        use crate::quantization::StorageFormat;
        let dummy = |id: u32| FufNode {
            id: TileId(id),
            op: OpKind::Attention,
            inputs: vec![],
            outputs: vec![],
        };
        let fuf = Fuf {
            nodes: vec![
                dummy(0),
                dummy(1),
                dummy(2),
                dummy(3),
                dummy(4),
                dummy(5),
                dummy(6),
                dummy(7),
                dummy(8),
                FufNode {
                    id: TileId(9),
                    op: OpKind::RmsNorm,
                    inputs: vec![],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(10),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(9),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: crate::classified::WeightId(42),
                            index: Some(5),
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(11),
                    op: OpKind::RopeAppend,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(10),
                            slot: 0,
                        },
                        FufInput::Extern {
                            kind: ExternKind::Rotary,
                            index: None,
                        },
                    ],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(12),
                    op: OpKind::Attention,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(11),
                            slot: 1,
                        },
                        FufInput::Extern {
                            kind: ExternKind::KvCache,
                            index: Some(5),
                        },
                    ],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(13),
                    op: OpKind::Attention,
                    inputs: vec![FufInput::Tile {
                        id: TileId(12),
                        slot: 0,
                    }],
                    outputs: vec![],
                },
            ],
        };
        let mut region = super::qkv_rope_region(&ferrite_stencil::QkvRopeParams {
            hidden_dim: 4096,
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            token_tile: 64,
            k_tile: 32,
            pipe: 3,
            writes_kv_cache: true,
        });
        stamp_gmem_bindings(
            &mut region,
            "fused_qkv_rope_cache",
            &fuf,
            &[TileId(10), TileId(11), TileId(12), TileId(13)],
        );
        let b = &region.gmem_bindings;
        // X comes from upstream tile 9 (not in subgraph).
        assert!(b.iter().any(|(c, id)| *c == "X_gmem" && *id == "t9_0"));
        // Wqkv identified by weight id + layer index.
        assert!(b.iter().any(|(c, id)| *c == "Wqkv_gmem" && *id == "w42_5"));
        // Rotary extern.
        assert!(
            b.iter()
                .any(|(c, id)| *c == "RopeCoef_gmem" && *id == "x_rotary_0")
        );
        // KvCache extern (layer-scoped by index).
        assert!(
            b.iter()
                .any(|(c, id)| *c == "K_cache_gmem" && *id == "x_kv_cache_5")
        );
        assert!(
            b.iter()
                .any(|(c, id)| *c == "V_cache_gmem" && *id == "x_kv_cache_5")
        );
        // Output tile_id = max in subgraph = 13; slots 0/1/2 = Q/K/V.
        assert!(b.iter().any(|(c, id)| *c == "Q_gmem" && *id == "t13_0"));
        assert!(b.iter().any(|(c, id)| *c == "K_gmem" && *id == "t13_1"));
        assert!(b.iter().any(|(c, id)| *c == "V_gmem" && *id == "t13_2"));
    }

    #[test]
    fn unknown_impl_leaves_bindings_empty() {
        let fuf = singleton_attention_fuf();
        let mut region = super::attn_region(&ferrite_stencil::AttnParams {
            window: ferrite_stencil::Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        stamp_gmem_bindings(
            &mut region,
            "some_impl_not_in_the_table",
            &fuf,
            &[TileId(0)],
        );
        assert!(region.gmem_bindings.is_empty());
    }
}
