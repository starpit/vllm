// SPDX-License-Identifier: Apache-2.0
//! Sub-tiling pass: walk the FUF and annotate every tile with its
//! natural tile-axis set + per-input dep vectors.
//!
//! This is the first pass on the road to Stencil IR (see
//! `STENCIL_IR_V2_DESIGN.md` §5). Output is pure *annotation* of
//! FUF — no rewrite, no new DAG, no consumers yet. The point is
//! to pin down per-op axis assignments concretely so the later
//! Region-formation and periodicity-detection passes have a stable
//! substrate.
//!
//! Axis vocabulary is frozen to seven names (design §4.1). New
//! axes join the vocabulary here; impls don't invent their own.
//!
//! Current scope limits:
//! - One aggregate Compute node per FUF op. Decomposition into
//!   Load / Compute / Store happens later, during Stencil IR
//!   construction.
//! - Tile sizes are symbolic (`TILE_T`, `TILE_D_INTER`, …). No
//!   size instantiation here.
//! - Gemm role (qkv / gate-up / down / o / lm_head) inferred from
//!   downstream usage: what op consumes the gemm output decides
//!   which `tile_d_*` axis applies.
//! - Attention is marked as a stencil boundary; its own axes are
//!   assigned but cross-region sharing is disallowed in the
//!   Region-formation pass.
//!
//! See `STENCIL_IR_V2_DESIGN.md` for the full picture.

#![allow(dead_code)]

use crate::classified::OpKind;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};

/// Frozen axis vocabulary. New axes join this enum; don't invent
/// ad-hoc names. Extending the enum is a design change, not an
/// impl-level detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TileAxis {
    /// Token-tile. Primary spatial axis. Shared across essentially
    /// every Region in a transformer forward — every per-token op,
    /// plus the activation side of every GEMM.
    TileT,
    /// MLP intermediate-dim tile. Present only in MLP Regions;
    /// enables gate/up/silu·mul/down fusion along this axis.
    TileDInter,
    /// Head-dim tile. Attention-internal.
    TileDHead,
    /// Head-group axis. Shared within attention sub-Regions,
    /// and carried on rope_append outputs (per-head Q/K/V).
    HeadGroup,
    /// Attention-internal query-tile axis.
    QTile,
    /// Attention-internal key/value-tile axis.
    KvTile,
    /// LM-head vocabulary-dim tile. Intra-Region only for now.
    TileDVocab,
}

/// Dep vector on the iteration domain. `Δ` per axis; axes not
/// mentioned are assumed `0`.
///
/// Stored as a sparse list so most deps (which touch one or two
/// axes) stay small. Represented with `i32` because dep vectors can
/// be negative (upstream tile at offset `-1`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DepVec {
    pub entries: Vec<(TileAxis, i32)>,
}

impl DepVec {
    pub fn zero() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn from_axis(axis: TileAxis, delta: i32) -> Self {
        Self {
            entries: vec![(axis, delta)],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.entries.iter().all(|(_, d)| *d == 0)
    }
}

/// Per-FUF-tile annotation produced by the sub-tiling pass.
///
/// `axes` is the axis set this tile iterates over at the outer
/// stencil level. `input_deps[i]` is the dep vector on this
/// tile's `inputs[i]` — describes how this tile's iteration
/// coordinates relate to the upstream tile's coordinates.
///
/// For non-tile inputs (weights, externs, scalars), `input_deps`
/// holds a zero vector — those inputs don't iterate on the
/// stencil's outer axes.
#[derive(Clone, Debug)]
pub struct SubtiledTile {
    pub tile: TileId,
    /// The axes this tile's computation iterates over.
    pub axes: Vec<TileAxis>,
    /// One dep vector per entry in `FufNode::inputs`, in the same
    /// order.
    pub input_deps: Vec<DepVec>,
    /// True iff this op is a stencil boundary — its dep pattern
    /// isn't affine on the spatial axes (today: attention).
    pub is_boundary: bool,
    /// Kind of gemm, if `op == Gemm`. `None` for non-gemm ops.
    /// Drives which `tile_d_*` axis applies.
    pub gemm_role: Option<GemmRole>,
}

/// Which role a `Gemm` plays in the forward. Determines its
/// `tile_d_out` axis.
///
/// Inferred from downstream consumers:
/// - Consumed by `RopeAppend*` → `QkvProjection`.
/// - Consumed by `Silu` / `Gelu` → `MlpGateProjection` (feeds the
///   activation of the gate·up multiply).
/// - Consumed by `Mul` (the gate·up mul) without upstream Silu/Gelu
///   → `MlpUpProjection`.
/// - Consumed by `Add` (residual stream) → `AttnOutput` or
///   `MlpDown` — distinguished by the upstream side of the add.
/// - No consumer (DAG sink) → `LmHead`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmRole {
    /// Q / K / V projection. Output `tile_d_out` = `HeadGroup *
    /// TileDHead` — but at the outer stencil level we just carry
    /// `HeadGroup` and let the Region sub-tile further internally.
    QkvProjection,
    /// MLP gate projection (consumed by Silu/Gelu then Mul).
    /// Output axis: `TileDInter`.
    MlpGateProjection,
    /// MLP up projection (consumed by Mul directly). Output axis:
    /// `TileDInter`.
    MlpUpProjection,
    /// MLP down projection (reduces `TileDInter` → hidden). Output
    /// consumed by residual Add. No outer `tile_d_*` axis — the
    /// output is `TileT`-only at the outer stencil level.
    MlpDown,
    /// Attention output projection (reduces head_dim*num_heads →
    /// hidden). Output consumed by residual Add. Same shape as
    /// `MlpDown` at the outer stencil level.
    AttnOutput,
    /// LM head (final gemm). Output axis: `TileDVocab`.
    LmHead,
    /// Couldn't be classified — emitted but marked. Consumers log
    /// and fall back to conservative axis assignment. Shouldn't
    /// happen on well-formed transformer FUFs; a safety valve.
    Unknown,
}

/// Sub-tiled FUF: the original FUF + one annotation per tile.
#[derive(Clone, Debug)]
pub struct SubtiledFuf<'a> {
    pub fuf: &'a Fuf,
    pub tiles: Vec<SubtiledTile>,
}

/// Walk the FUF, annotate every tile. Infallible: unknown ops fall
/// back to `(TileT,)` with zero dep vectors — correct for any
/// strictly-per-token op even if we haven't special-cased it.
pub fn subtile(fuf: &Fuf) -> SubtiledFuf<'_> {
    // Precompute downstream consumers so Gemm role inference is
    // O(|inputs|) per consumer rather than O(|nodes|²).
    let consumers = build_consumer_map(fuf);

    let mut tiles: Vec<SubtiledTile> = Vec::with_capacity(fuf.len());
    for node in &fuf.nodes {
        tiles.push(subtile_one(fuf, node, &consumers));
    }
    SubtiledFuf { fuf, tiles }
}

/// Classify one FUF op.
fn subtile_one(fuf: &Fuf, node: &FufNode, consumers: &ConsumerMap) -> SubtiledTile {
    let (axes, is_boundary, gemm_role) = match node.op {
        // Elementwise / per-token ops. All live on `(TileT,)`. Dep
        // vectors to upstream tiles are zero on TileT (same tile
        // reads same tile).
        OpKind::RmsNorm
        | OpKind::LayerNorm
        | OpKind::Silu
        | OpKind::Gelu
        | OpKind::Add
        | OpKind::BiasAdd
        | OpKind::Mul
        | OpKind::TanhSoftCap => (vec![TileAxis::TileT], false, None),

        // Embed: gather into `[tile_t, hidden]` from token_ids.
        // Outer axis is TileT; the gather inside the Region is
        // sub-tile.
        OpKind::Embed => (vec![TileAxis::TileT], false, None),

        // Gemm: axes depend on role. Activation side always carries
        // TileT; the `tile_d_out` axis depends on consumer.
        OpKind::Gemm => {
            let role = infer_gemm_role(fuf, node, consumers);
            let axes = gemm_axes_for_role(role);
            (axes, false, Some(role))
        }

        // RopeAppend: per-token per-head rotation + cache write.
        // Outer axes: (TileT, HeadGroup). Multi-output (q, k, v)
        // — each output carries the same axes.
        OpKind::RopeAppend | OpKind::RopeAppendInterleaved => {
            (vec![TileAxis::TileT, TileAxis::HeadGroup], false, None)
        }

        // Attention: cross-token reduction → stencil boundary.
        // Axes describe the Region's *internal* iteration; they
        // don't share with non-attention Regions (HeadGroup might
        // share with RopeAppend's HeadGroup — Region-formation
        // pass decides).
        OpKind::Attention | OpKind::SlidingAttention => (
            vec![TileAxis::HeadGroup, TileAxis::QTile, TileAxis::KvTile],
            true,
            None,
        ),

        // Reshape: metadata-only, no math. Axes passthrough from
        // upstream tile input (if any). Safe default: inherit the
        // first tile input's axes; fallback to TileT.
        OpKind::Reshape => {
            let axes = first_tile_input_axes(fuf, node, consumers)
                .unwrap_or_else(|| vec![TileAxis::TileT]);
            (axes, false, None)
        }
    };

    // Dep vectors per input: zero on the op's own axes for
    // same-coord reads. Upstream tile reads have `(0, …)` on the
    // outer axes for now; the Region-formation pass refines
    // intra-Region deps (e.g. pipelined gate → silu·mul) once
    // fusion boundaries are known.
    //
    // Weights / externs / scalars get the zero vector — they
    // don't iterate on outer axes.
    let input_deps: Vec<DepVec> = node.inputs.iter().map(|_| DepVec::zero()).collect();

    SubtiledTile {
        tile: node.id,
        axes,
        input_deps,
        is_boundary,
        gemm_role,
    }
}

fn gemm_axes_for_role(role: GemmRole) -> Vec<TileAxis> {
    match role {
        GemmRole::QkvProjection => vec![TileAxis::TileT, TileAxis::HeadGroup],
        GemmRole::MlpGateProjection | GemmRole::MlpUpProjection => {
            vec![TileAxis::TileT, TileAxis::TileDInter]
        }
        GemmRole::MlpDown | GemmRole::AttnOutput => vec![TileAxis::TileT],
        GemmRole::LmHead => vec![TileAxis::TileT, TileAxis::TileDVocab],
        // Unknown falls back to TileT — the safest
        // non-tile-disrupting choice. Region formation will log
        // these so we can tighten over time.
        GemmRole::Unknown => vec![TileAxis::TileT],
    }
}

/// Map every `(producer_tile, slot)` to the list of downstream
/// tiles that reference it via `FufInput::Tile`.
type ConsumerMap = Vec<Vec<TileId>>;

fn build_consumer_map(fuf: &Fuf) -> ConsumerMap {
    let mut consumers: ConsumerMap = vec![Vec::new(); fuf.len()];
    for node in &fuf.nodes {
        for input in &node.inputs {
            if let FufInput::Tile { id, .. } = input {
                consumers[id.0 as usize].push(node.id);
            }
        }
    }
    consumers
}

/// Infer a gemm's role by looking at which op consumes its output.
///
/// Multi-consumer case: take the "most informative" consumer —
/// RopeAppend > Silu/Gelu > Mul > Add. If none match, it's
/// probably lm_head (no consumer) or Unknown.
fn infer_gemm_role(fuf: &Fuf, node: &FufNode, consumers: &ConsumerMap) -> GemmRole {
    let downstream = &consumers[node.id.0 as usize];
    if downstream.is_empty() {
        return GemmRole::LmHead;
    }
    // Scan consumers; prefer the most specific signal.
    let mut saw_rope = false;
    let mut saw_silu_or_gelu = false;
    let mut saw_mul = false;
    let mut saw_add = false;
    for consumer_id in downstream {
        match fuf.get(*consumer_id).op {
            OpKind::RopeAppend | OpKind::RopeAppendInterleaved => saw_rope = true,
            OpKind::Silu | OpKind::Gelu => saw_silu_or_gelu = true,
            OpKind::Mul => saw_mul = true,
            OpKind::Add | OpKind::BiasAdd => saw_add = true,
            _ => {}
        }
    }
    if saw_rope {
        return GemmRole::QkvProjection;
    }
    if saw_silu_or_gelu {
        return GemmRole::MlpGateProjection;
    }
    if saw_mul {
        return GemmRole::MlpUpProjection;
    }
    if saw_add {
        // AttnOutput vs MlpDown — distinguish by which residual
        // this add closes. For v1 both map to `(TileT,)`, so the
        // distinction doesn't affect axes. Use a heuristic:
        // AttnOutput is typically upstream of the first rmsnorm
        // after attention; MlpDown is downstream of the MLP
        // block's rmsnorm. Walk a bounded distance to
        // disambiguate. For v1 we accept MlpDown when any
        // ancestor is a Silu/Gelu/Mul; AttnOutput otherwise.
        if gemm_has_mlp_ancestor(fuf, node.id) {
            return GemmRole::MlpDown;
        }
        return GemmRole::AttnOutput;
    }
    GemmRole::Unknown
}

/// True if any tile-input ancestor of `tile` (bounded search
/// depth) is a Silu/Gelu/Mul — the MLP signature.
fn gemm_has_mlp_ancestor(fuf: &Fuf, tile: TileId) -> bool {
    use std::collections::{HashSet, VecDeque};
    const MAX_DEPTH: usize = 16;
    let mut q: VecDeque<(TileId, usize)> = VecDeque::new();
    let mut seen: HashSet<TileId> = HashSet::new();
    q.push_back((tile, 0));
    seen.insert(tile);
    while let Some((t, depth)) = q.pop_front() {
        if depth > MAX_DEPTH {
            continue;
        }
        let n = fuf.get(t);
        match n.op {
            OpKind::Silu | OpKind::Gelu | OpKind::Mul => return true,
            _ => {}
        }
        for input in &n.inputs {
            if let FufInput::Tile { id, .. } = input
                && seen.insert(*id)
            {
                q.push_back((*id, depth + 1));
            }
        }
    }
    false
}

/// Axis set of the first tile-input ancestor (for Reshape
/// passthrough). Returns `None` if the node has no tile input.
fn first_tile_input_axes(
    fuf: &Fuf,
    node: &FufNode,
    consumers: &ConsumerMap,
) -> Option<Vec<TileAxis>> {
    for input in &node.inputs {
        if let FufInput::Tile { id, .. } = input {
            // Recursively subtile the producer to inherit its axes.
            // For v1, bounded: just use the producer's op directly;
            // deeper passthroughs are rare and Reshape rarely
            // chains. If it does chain, the recursion will happen
            // naturally through the top-level walk.
            let producer = fuf.get(*id);
            let one = subtile_one(fuf, producer, consumers);
            return Some(one.axes);
        }
    }
    None
}

/// Pretty-print the sub-tiled FUF for inspection. Format: one
/// line per tile with `id`, `op`, axis set, and gemm role if any.
///
/// Intended for eyeballing a llama-size FUF during development;
/// not a format we want to parse.
pub fn pretty_print(st: &SubtiledFuf<'_>) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(out, "sub-tiled FUF: {} tiles", st.tiles.len()).unwrap();
    for (idx, t) in st.tiles.iter().enumerate() {
        let node = &st.fuf.nodes[idx];
        let axes_str = t
            .axes
            .iter()
            .map(|a| format!("{a:?}"))
            .collect::<Vec<_>>()
            .join(",");
        let role_str = t.gemm_role.map(|r| format!(" [{r:?}]")).unwrap_or_default();
        let boundary_str = if t.is_boundary { " *BOUNDARY*" } else { "" };
        writeln!(
            out,
            "  t{:>3} {:?} axes=({axes_str}){role_str}{boundary_str}",
            node.id.0, node.op
        )
        .unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuf::Fuf;

    /// Build a tiny synthetic FUF by hand. Avoids the full unroll
    /// pipeline and keeps the test self-contained.
    fn fuf_from_ops(ops: Vec<(OpKind, Vec<FufInput>, usize)>) -> Fuf {
        use crate::fuf::FufNode;
        let nodes = ops
            .into_iter()
            .enumerate()
            .map(|(i, (op, inputs, num_outputs))| FufNode {
                id: TileId(i as u32),
                op,
                inputs,
                outputs: (0..num_outputs).map(|_| Vec::new()).collect(),
            })
            .collect();
        Fuf { nodes }
    }

    fn tile(id: u32) -> FufInput {
        FufInput::Tile {
            id: TileId(id),
            slot: 0,
        }
    }

    #[test]
    fn rmsnorm_is_tile_t() {
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),
            (OpKind::RmsNorm, vec![tile(0)], 1),
        ]);
        let st = subtile(&fuf);
        assert_eq!(st.tiles[1].axes, vec![TileAxis::TileT]);
        assert!(!st.tiles[1].is_boundary);
    }

    #[test]
    fn attention_is_boundary() {
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),
            (OpKind::Attention, vec![tile(0)], 1),
        ]);
        let st = subtile(&fuf);
        assert!(st.tiles[1].is_boundary);
        assert_eq!(
            st.tiles[1].axes,
            vec![TileAxis::HeadGroup, TileAxis::QTile, TileAxis::KvTile]
        );
    }

    #[test]
    fn gemm_to_rope_is_qkv() {
        // tile 0: embed, tile 1: gemm, tile 2: rope_append consuming gemm.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),
            (OpKind::Gemm, vec![tile(0)], 1),
            (OpKind::RopeAppend, vec![tile(1), tile(1), tile(1)], 3),
        ]);
        let st = subtile(&fuf);
        assert_eq!(st.tiles[1].gemm_role, Some(GemmRole::QkvProjection));
        assert_eq!(st.tiles[1].axes, vec![TileAxis::TileT, TileAxis::HeadGroup]);
    }

    #[test]
    fn gemm_to_silu_is_gate() {
        // tile 0: embed, tile 1: gemm, tile 2: silu consuming gemm.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),
            (OpKind::Gemm, vec![tile(0)], 1),
            (OpKind::Silu, vec![tile(1)], 1),
        ]);
        let st = subtile(&fuf);
        assert_eq!(st.tiles[1].gemm_role, Some(GemmRole::MlpGateProjection));
        assert_eq!(
            st.tiles[1].axes,
            vec![TileAxis::TileT, TileAxis::TileDInter]
        );
    }

    #[test]
    fn gemm_to_mul_is_up() {
        // tile 0: embed, tile 1: gemm, tile 2: mul consuming gemm (not through silu).
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),
            (OpKind::Gemm, vec![tile(0)], 1),
            (OpKind::Mul, vec![tile(1), tile(0)], 1),
        ]);
        let st = subtile(&fuf);
        assert_eq!(st.tiles[1].gemm_role, Some(GemmRole::MlpUpProjection));
    }

    #[test]
    fn gemm_no_consumer_is_lm_head() {
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),
            (OpKind::Gemm, vec![tile(0)], 1),
        ]);
        let st = subtile(&fuf);
        assert_eq!(st.tiles[1].gemm_role, Some(GemmRole::LmHead));
        assert_eq!(
            st.tiles[1].axes,
            vec![TileAxis::TileT, TileAxis::TileDVocab]
        );
    }

    #[test]
    fn gemm_to_add_after_mlp_is_down() {
        // embed → gemm(gate) → silu → mul → gemm(down) → add.
        // Down gemm's consumer is Add; MlpDown should win via
        // MLP-ancestor heuristic.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),               // 0
            (OpKind::Gemm, vec![tile(0)], 1),         // 1 (gate)
            (OpKind::Silu, vec![tile(1)], 1),         // 2
            (OpKind::Mul, vec![tile(2), tile(0)], 1), // 3
            (OpKind::Gemm, vec![tile(3)], 1),         // 4 (down)
            (OpKind::Add, vec![tile(4), tile(0)], 1), // 5
        ]);
        let st = subtile(&fuf);
        assert_eq!(st.tiles[4].gemm_role, Some(GemmRole::MlpDown));
    }

    #[test]
    fn gemm_to_add_without_mlp_is_attn_output() {
        // embed → gemm(o) → add. No silu/gelu/mul ancestors → AttnOutput.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![], 1),
            (OpKind::Gemm, vec![tile(0)], 1),
            (OpKind::Add, vec![tile(1), tile(0)], 1),
        ]);
        let st = subtile(&fuf);
        assert_eq!(st.tiles[1].gemm_role, Some(GemmRole::AttnOutput));
    }
}
