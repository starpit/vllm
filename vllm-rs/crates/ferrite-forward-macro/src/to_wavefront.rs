// SPDX-License-Identifier: Apache-2.0
//! PD-wavefront task **T2b** — the macro→wavefront bridge.
//!
//! Translate a *solved* decode FUF (`Fuf` + solver [`Assignment`]) into a
//! [`ferrite_wavefront::lower::LoweringInput`] and hand it to
//! [`ferrite_wavefront::lower::lower`], so a real Llama-3.2-1B forward
//! flows through the host-validated subtile pipeline (DAG eval + tape
//! player + wavefront scheduler).
//!
//! This is the one piece that *cannot* be `cargo test`-ed on a Mac (the
//! macro crate's test suite is cuda-coupled), so it is kept deliberately
//! thin and mechanical: every structural decision it makes — embed
//! becomes a read-only `Source`, `rope_append` splits into a Q-side
//! `RopeRotate` + a K-side `RopeAppend` (the GPU cache-write) with the
//! un-roped V aliasing the V-proj output, `attention` reads the prefix KV
//! cache as `Source` segments + the new token as `Sub` edges, `lm_head` is
//! the result — is mirrored exactly
//! by the Mac-testable `full_forward_bit_exact` test in
//! `ferrite_wavefront::lower`. The bridge resolves shapes and wires
//! edges; the decomposition *semantics* it targets are already proven
//! bit-exact vs `cpu_golden` there.
//!
//! Coarse granularity (one subtile per op) per the plan: split-K /
//! N-block tiling and the inter-op slicing it needs are layered on at
//! the scheduler, not here.
//!
//! Scope: the Llama-3.2 decode op set (`Embed`, `RmsNorm`, `Gemm`,
//! `RopeAppend`, `Attention`, `Silu`, `Mul`, `Add`). Anything else
//! (MoE, MLA, sliding-window / interleaved rope, vision, TP collectives)
//! is surfaced as [`BridgeError::UnsupportedOp`] — never silently
//! skipped. The drive treats any error as "no wavefront lowering for
//! this model" and continues; this never gates a normal build.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use ferrite_wavefront::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
use ferrite_wavefront::mega::SourceDesc;
use ferrite_wavefront::subtile::SourceShape;
use ferrite_wavefront::metal_tape::{BufferRef, WeightBundle, WeightLoc, WeightRole};

use crate::classified::{ExternKind, OpKind, Program, WeightId};
use crate::codegen::{split_base_layer, weight_kind_accessor_method};
use crate::config::ModelParams;
use crate::emit::weight_field_name;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::impl_lib::{
    WeightKind, WeightSlot, attention_scale_for, eval_shape_with, gemm_nk_from_fuf,
};
use crate::quantization::StorageFormat;
use crate::shape::Inferred;
use crate::solver::Assignment;

/// Why a FUF couldn't be lowered to a wavefront `LoweringInput`. Always
/// surfaced (logged by the drive), never papered over.
#[derive(Debug)]
pub enum BridgeError {
    /// FUF op kind outside the coarse Llama-3.2 decode set.
    UnsupportedOp { tile: TileId, op: OpKind },
    /// A tile's (or an upstream tile's) output didn't close to a
    /// concrete shape under `bounds` — shape inference left a `Var`, or
    /// a dim referenced a missing config bound.
    UnresolvedShape { tile: TileId, what: &'static str },
    /// A tile referenced a `(TileId, slot)` the walk hadn't produced —
    /// FUF not in ascending-id topological order, or a bad slot.
    DanglingInput { tile: TileId, dep: TileId, slot: u8 },
    /// An op's inputs didn't match the arity / kinds the bridge expects
    /// for that `OpKind`.
    MalformedOp {
        tile: TileId,
        op: OpKind,
        detail: &'static str,
    },
    /// A required model scalar (e.g. `rms_norm_eps`) was absent.
    MissingScalar { key: &'static str },
    /// A required integer bound (e.g. `head_dim`) was absent.
    MissingBound { key: &'static str },
    /// The FUF produced no result op (empty, or the last tile wasn't a
    /// value-producing op).
    NoResult,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOp { tile, op } => {
                write!(
                    f,
                    "tile {} op {op:?} is outside the coarse decode set",
                    tile.0
                )
            }
            Self::UnresolvedShape { tile, what } => {
                write!(
                    f,
                    "tile {} {what} did not close to a concrete shape",
                    tile.0
                )
            }
            Self::DanglingInput { tile, dep, slot } => write!(
                f,
                "tile {} reads ({}, slot {}) which was not produced yet",
                tile.0, dep.0, slot
            ),
            Self::MalformedOp { tile, op, detail } => {
                write!(f, "tile {} op {op:?}: {detail}", tile.0)
            }
            Self::MissingScalar { key } => write!(f, "model scalar `{key}` missing"),
            Self::MissingBound { key } => write!(f, "config bound `{key}` missing"),
            Self::NoResult => write!(f, "FUF produced no result op"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// What real tensor each wavefront `Source` is bound to at run time —
/// parallel to [`LoweringInput::sources`]. The bridge anonymizes sources
/// to bare shapes; this manifest is how the Tier-B host executor (and
/// later the GPU tape player) maps `SourceId(i)` back to a concrete
/// buffer. `Weight` carries the raw [`crate::classified::WeightId`] +
/// unrolled index (resolve to a tensor name via the same `Program`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceBinding {
    /// A model weight: the `index` is the unrolled (former loop-var)
    /// integer, so `(id, index)` uniquely names a per-layer tensor.
    Weight {
        id: u32,
        index: Option<u64>,
    },
    /// The embedded hidden-state row the runtime gathers from
    /// `embed_tokens[input_id]` before the kernel — embed is a cheap
    /// host lookup, not a megakernel op.
    EmbeddedHidden,
    /// The new token's rotary `cos` / `sin` row `[1, head_dim]` (shared
    /// across layers at one decode position).
    Cos,
    Sin,
    /// The read-only prefix KV cache for a layer, `[prefix_len, kvdim]`.
    PrefixK {
        layer: u64,
    },
    PrefixV {
        layer: u64,
    },
}

/// The bridge's output: a coarse subtile `LoweringInput` plus the
/// per-source binding manifest the executor needs to fill it.
#[derive(Debug, Clone)]
pub struct LoweredDecode {
    pub input: LoweringInput,
    /// Parallel to `input.sources`: what each source is bound to.
    pub bindings: Vec<SourceBinding>,
}

/// Lightweight stats for the macro-time dump.
#[derive(Debug, Clone)]
pub struct BridgeStats {
    pub fuf_tiles: usize,
    pub subgraphs: usize,
    pub sources: usize,
    pub ops: usize,
    /// `(LoweredOp kind name, count)`, sorted by name.
    pub op_histogram: Vec<(&'static str, usize)>,
    /// Source bindings bucketed: weights, prefix-KV pairs, and the
    /// fixed singletons (embed/cos/sin).
    pub weight_sources: usize,
    pub prefix_sources: usize,
}

/// One producer of a FUF `(tile, slot)`: either a wavefront op output,
/// or a graph `Source` (embed becomes a source; a `rope_append`'s
/// un-roped V slot aliases the V-proj op output, which is itself an
/// `Op`, so that case stays `Op` here).
#[derive(Clone, Copy)]
enum Producer {
    Op(usize),
    Ext(usize),
}

impl Producer {
    fn as_input_ref(self) -> InputRef {
        match self {
            Producer::Op(j) => InputRef::Op(j),
            Producer::Ext(e) => InputRef::Ext(e),
        }
    }
}

/// Mutable bridge state threaded through the FUF walk.
struct Builder<'a> {
    fuf: &'a Fuf,
    inferred: &'a Inferred,
    bounds: BTreeMap<String, u64>, // model bounds + num_tokens=1
    sources: Vec<SourceShape>,
    bindings: Vec<SourceBinding>,
    ops: Vec<OpDesc>,
    /// FUF `(tile_id, slot)` → its producer in the wavefront graph.
    produced: HashMap<(u32, u8), Producer>,
    /// Dedup weight sources by `(WeightId, index)`.
    weight_src: HashMap<(u32, Option<u64>), usize>,
    /// Shared cos / sin `[1, head_dim]` source indices (decode: one
    /// position; every layer's rope reads the same rotary row).
    cos_sin: Option<(usize, usize)>,
    /// Prefix KV-cache source pair `(prefix_k, prefix_v)` per kv-cache
    /// extern index (one per layer).
    prefix: HashMap<u64, (usize, usize)>,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    eps: f32,
    scale: f32,
    m: u32,
    /// Modeled prefix length for the KV-cache `Source` rows (structural;
    /// the host eval / GPU player binds the real length at run time).
    prefix_len: u32,
}

impl<'a> Builder<'a> {
    fn push_op(&mut self, op: LoweredOp, inputs: Vec<InputRef>) -> usize {
        let idx = self.ops.len();
        self.ops.push(OpDesc {
            op,
            m: self.m,
            inputs,
        });
        idx
    }

    fn push_source(&mut self, rows: u32, cols: u32, binding: SourceBinding) -> usize {
        let e = self.sources.len();
        self.sources.push(SourceShape { rows, cols });
        self.bindings.push(binding);
        e
    }

    /// Resolve a Tile / Weight input to an `InputRef`. Externs and
    /// scalars are op-specific and must be handled by the caller.
    fn resolve(&mut self, tile: TileId, inp: &FufInput) -> Result<InputRef, BridgeError> {
        match inp {
            FufInput::Tile { id, slot } => match self.produced.get(&(id.0, *slot)) {
                Some(p) => Ok(p.as_input_ref()),
                None => Err(BridgeError::DanglingInput {
                    tile,
                    dep: *id,
                    slot: *slot,
                }),
            },
            FufInput::Weight { id, index, .. } => {
                let e = self.weight_source(tile, id.0, *index)?;
                Ok(InputRef::Ext(e))
            }
            FufInput::Extern { .. } => Err(BridgeError::MalformedOp {
                tile,
                op: self.fuf.get(tile).op,
                detail: "extern input not valid here",
            }),
            FufInput::Scalar(_) => Err(BridgeError::MalformedOp {
                tile,
                op: self.fuf.get(tile).op,
                detail: "scalar input not valid here",
            }),
        }
    }

    /// Get-or-create the dedup'd source for a weight reference.
    fn weight_source(
        &mut self,
        tile: TileId,
        id: u32,
        index: Option<u64>,
    ) -> Result<usize, BridgeError> {
        if let Some(&e) = self.weight_src.get(&(id, index)) {
            return Ok(e);
        }
        let shape = self
            .inferred
            .weights
            .get(&crate::classified::WeightId(id))
            .ok_or(BridgeError::UnresolvedShape {
                tile,
                what: "weight (no inferred shape)",
            })?;
        let dims = eval_shape_with(shape, &self.bounds).ok_or(BridgeError::UnresolvedShape {
            tile,
            what: "weight shape has unresolved dim",
        })?;
        let (rows, cols) = match dims.as_slice() {
            [n] => (1u32, *n as u32),
            [r, c] => (*r as u32, *c as u32),
            _ => {
                return Err(BridgeError::UnresolvedShape {
                    tile,
                    what: "weight rank not 1 or 2",
                });
            }
        };
        let e = self.push_source(rows, cols, SourceBinding::Weight { id, index });
        self.weight_src.insert((id, index), e);
        Ok(e)
    }

    /// The single Tile or Weight input matching `pred`, resolved.
    fn input_at(&mut self, tile: TileId, idx: usize) -> Result<InputRef, BridgeError> {
        let inp = self
            .fuf
            .get(tile)
            .inputs
            .get(idx)
            .ok_or(BridgeError::MalformedOp {
                tile,
                op: self.fuf.get(tile).op,
                detail: "missing input at expected index",
            })?
            .clone();
        self.resolve(tile, &inp)
    }

    /// Output dims `(rows, cols)` of one tile slot, resolved to concrete
    /// integers. Rank-1 shapes are treated as `[1, n]`.
    fn out_cols(&self, tile: TileId, slot: usize, what: &'static str) -> Result<u32, BridgeError> {
        let node = self.fuf.get(tile);
        let shape = node
            .outputs
            .get(slot)
            .ok_or(BridgeError::UnresolvedShape { tile, what })?;
        let dims = eval_shape_with(shape, &self.bounds)
            .ok_or(BridgeError::UnresolvedShape { tile, what })?;
        dims.last()
            .copied()
            .map(|c| c as u32)
            .ok_or(BridgeError::UnresolvedShape { tile, what })
    }

    fn cos_sin(&mut self) -> (usize, usize) {
        if let Some(cs) = self.cos_sin {
            return cs;
        }
        let cos = self.push_source(1, self.head_dim, SourceBinding::Cos);
        let sin = self.push_source(1, self.head_dim, SourceBinding::Sin);
        self.cos_sin = Some((cos, sin));
        (cos, sin)
    }

    fn prefix_for(&mut self, layer: u64) -> (usize, usize) {
        if let Some(pp) = self.prefix.get(&layer) {
            return *pp;
        }
        let kvdim = self.num_kv_heads * self.head_dim;
        let pk = self.push_source(self.prefix_len, kvdim, SourceBinding::PrefixK { layer });
        let pv = self.push_source(self.prefix_len, kvdim, SourceBinding::PrefixV { layer });
        self.prefix.insert(layer, (pk, pv));
        (pk, pv)
    }
}

/// Pull the kv-cache extern's layer index out of an attention /
/// rope_append input list. Returns the first `Extern{KvCache, Some(i)}`.
fn kv_cache_index(node: &FufNode) -> Option<u64> {
    node.inputs.iter().find_map(|inp| match inp {
        FufInput::Extern {
            kind: ExternKind::KvCache,
            index,
        } => *index,
        _ => None,
    })
}

/// Translate a solved decode FUF into a wavefront `LoweringInput`.
///
/// `asn` is the `num_tokens = 1` (decode) [`Assignment`]; `bounds` is the
/// per-model integer bounds the solver used (tp-sharded if tp>1).
/// `prefix_len` models the KV-cache prefix rows for the attention
/// `Source` segments (structural only — the real length binds at run
/// time).
pub fn lower_decode_to_wavefront(
    fuf: &Fuf,
    asn: &Assignment,
    inferred: &Inferred,
    bounds: &BTreeMap<String, u64>,
    model: &ModelParams,
    prefix_len: u32,
) -> Result<LoweredDecode, BridgeError> {
    let eps = *model
        .scalars
        .get("rms_norm_eps")
        .ok_or(BridgeError::MissingScalar {
            key: "rms_norm_eps",
        })? as f32;
    let scale = attention_scale_for(model);
    let mut b = bounds.clone();
    b.insert("num_tokens".into(), 1);
    let m = 1u32;
    let head_dim = b
        .get("head_dim")
        .copied()
        .ok_or(BridgeError::MissingBound { key: "head_dim" })? as u32;
    let num_q_heads = b
        .get("num_attention_heads")
        .copied()
        .ok_or(BridgeError::MissingBound {
            key: "num_attention_heads",
        })? as u32;
    let num_kv_heads = b
        .get("num_key_value_heads")
        .copied()
        .ok_or(BridgeError::MissingBound {
            key: "num_key_value_heads",
        })? as u32;

    let mut bx = Builder {
        fuf,
        inferred,
        bounds: b,
        sources: Vec::new(),
        bindings: Vec::new(),
        ops: Vec::new(),
        produced: HashMap::new(),
        weight_src: HashMap::new(),
        cos_sin: None,
        prefix: HashMap::new(),
        head_dim,
        num_q_heads,
        num_kv_heads,
        eps,
        scale,
        m,
        prefix_len,
    };

    // The result is the last value-producing op (the lm_head GEMM).
    let mut result: Option<usize> = None;

    // Topological walk over `fuf.nodes`. The bridge's resolver looks
    // up each input's producer in `bx.produced`, so a tile must be
    // visited only AFTER every tile-input it references. Metal's
    // post-fusion FUF happens to be stored in ascending-id topo order
    // (which is why the original `for node in &fuf.nodes` loop
    // sufficed), but cuda's fusion pass reorders ids — so we must
    // sort. Standard Kahn: unmet-input count per tile, queue of
    // ready tiles, process in order, decrement successors. Result is
    // a permutation of `fuf.nodes` that respects every tile→tile
    // edge regardless of how fusion happened to assign ids.
    let topo: Vec<TileId> = {
        let n = fuf.nodes.len();
        let mut indeg: Vec<u32> = vec![0; n];
        let mut succs: Vec<Vec<u32>> = vec![Vec::new(); n];
        for node in &fuf.nodes {
            let me = node.id.0 as usize;
            for inp in &node.inputs {
                if let FufInput::Tile { id, .. } = inp {
                    let dep = id.0 as usize;
                    indeg[me] += 1;
                    succs[dep].push(node.id.0);
                }
            }
        }
        let mut queue: std::collections::VecDeque<u32> = (0..n as u32)
            .filter(|&i| indeg[i as usize] == 0)
            .collect();
        let mut out: Vec<TileId> = Vec::with_capacity(n);
        while let Some(t) = queue.pop_front() {
            out.push(TileId(t));
            for &s in &succs[t as usize] {
                indeg[s as usize] -= 1;
                if indeg[s as usize] == 0 {
                    queue.push_back(s);
                }
            }
        }
        if out.len() != n {
            // Cycle in the FUF — that's structurally impossible
            // (the FUF is a DAG built from straight-line DSL +
            // unrolled `for`), so surface it loudly.
            return Err(BridgeError::MalformedOp {
                tile: TileId(0),
                op: fuf.nodes[0].op,
                detail: "FUF has a cycle; cannot topologically order",
            });
        }
        out
    };

    for tile_id in topo {
        let node = fuf.get(tile_id);
        let tile = node.id;
        // Every tile must be claimed by the solve — otherwise codegen
        // would have errored. A coverage gap here is a real bug, so
        // surface it rather than lower a tile the solver rejected.
        if asn.subgraph_of(tile).is_none() {
            return Err(BridgeError::MalformedOp {
                tile,
                op: node.op,
                detail: "tile not covered by the assignment",
            });
        }

        match node.op {
            // Embed is a host-side row gather, not a megakernel op: the
            // runtime supplies the embedded hidden state, so the embed
            // tile's output is a read-only graph Source.
            OpKind::Embed => {
                let cols = bx.out_cols(tile, 0, "embed output")?;
                let e = bx.push_source(m, cols, SourceBinding::EmbeddedHidden);
                bx.produced.insert((tile.0, 0), Producer::Ext(e));
            }
            // MmEmbedSplice is the multimodal placeholder splice the
            // tp-lowering pass inserts after every Embed to overwrite
            // placeholder positions with vision-encoder rows. For
            // text-only forwards (no `ctx.embed_patches`) it's a
            // runtime no-op — the megakernel sees its input slot
            // verbatim. Pass the upstream (Embed's output) through to
            // the splice's output slot so downstream tiles read the
            // same `EmbeddedHidden` source. No `LoweredOp` is emitted
            // — this op is structurally absent from the megakernel.
            OpKind::MmEmbedSplice => {
                let upstream = bx.input_at(tile, 0)?;
                let producer = match upstream {
                    InputRef::Op(j) => Producer::Op(j),
                    InputRef::Ext(e) => Producer::Ext(e),
                };
                bx.produced.insert((tile.0, 0), producer);
            }
            OpKind::RmsNorm => {
                let x = bx.input_at(tile, 0)?;
                let w = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::RmsNorm { eps }, vec![x, w]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Gemm => {
                let (n, k) = gemm_nk_from_fuf(fuf, node, &bx.bounds).ok_or(
                    BridgeError::UnresolvedShape {
                        tile,
                        what: "gemm n/k",
                    },
                )?;
                let act = bx.input_at(tile, 0)?;
                let w = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::Gemm { n, k }, vec![act, w]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // rope_append(q, k, v, positions, rotary, kv_cache[layer]) →
            // (q', k', v): the Q slot rotates in place (`RopeRotate`); the K
            // slot is the GPU cache-write `RopeAppend` (rotate K + write the
            // rotated K / un-rotated V to the paged cache for `layer`), so it
            // takes the un-roped V as a fourth input. The un-roped V (slot 2)
            // still aliases the V-proj output for the attention dataflow edge;
            // the serializer maps attention onto the runtime paged cache, so
            // the new K/V are NOT a separate cache round-trip.
            OpKind::RopeAppend => {
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                // slot 2 (v) is the third input's producer, aliased.
                let v_inp = node.inputs.get(2).ok_or(BridgeError::MalformedOp {
                    tile,
                    op: node.op,
                    detail: "rope_append missing v input",
                })?;
                let v = bx.resolve(tile, v_inp)?;
                let layer = kv_cache_index(node).ok_or(BridgeError::MalformedOp {
                    tile,
                    op: node.op,
                    detail: "rope_append missing kv_cache extern index",
                })? as u32;
                let (cos, sin) = bx.cos_sin();
                // E.12 — RopeAppend writes rotated K and V into the
                // paged KV cache. Pull the per-layer PrefixK / PrefixV
                // source indices via the cached `prefix_for(layer)`
                // helper (same indices Attention will receive later).
                let (pk, pv) = bx.prefix_for(layer as u64);
                let qi = bx.push_op(
                    LoweredOp::RopeRotate { head_dim },
                    vec![q, InputRef::Ext(cos), InputRef::Ext(sin)],
                );
                let ki = bx.push_op(
                    LoweredOp::RopeAppend { head_dim, layer },
                    vec![
                        k,
                        InputRef::Ext(cos),
                        InputRef::Ext(sin),
                        v,
                        InputRef::Ext(pk),
                        InputRef::Ext(pv),
                    ],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(qi));
                bx.produced.insert((tile.0, 1), Producer::Op(ki));
                bx.produced.insert(
                    (tile.0, 2),
                    match v {
                        InputRef::Op(j) => Producer::Op(j),
                        InputRef::Ext(e) => Producer::Ext(e),
                    },
                );
            }
            // attention(q', k', v, kv_cache, block_table): AttnDecode
            // reading Q_rot, prefix-cache Source segments, then the new
            // token's (K_rot, V) as Sub edges.
            OpKind::Attention => {
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                let v = bx.input_at(tile, 2)?;
                let layer = kv_cache_index(node).ok_or(BridgeError::MalformedOp {
                    tile,
                    op: node.op,
                    detail: "attention missing kv_cache extern index",
                })?;
                let (pk, pv) = bx.prefix_for(layer);
                let idx = bx.push_op(
                    LoweredOp::AttnDecode {
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                    },
                    vec![q, InputRef::Ext(pk), InputRef::Ext(pv), k, v],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Silu => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::Silu, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Mul => {
                let a = bx.input_at(tile, 0)?;
                let b2 = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::Mul, vec![a, b2]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Add => {
                let a = bx.input_at(tile, 0)?;
                let b2 = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::Add, vec![a, b2]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            other => return Err(BridgeError::UnsupportedOp { tile, op: other }),
        }
    }

    let result = result.ok_or(BridgeError::NoResult)?;
    debug_assert_eq!(bx.sources.len(), bx.bindings.len());
    Ok(LoweredDecode {
        input: LoweringInput {
            sources: bx.sources,
            ops: bx.ops,
            result,
        },
        bindings: bx.bindings,
    })
}

/// A per-arch weight locator recovered from a lowered decode bucket's
/// `weight_slots`: the `(bucket, op_idx, slot)` triple the runtime
/// `WeightAccessors::<kind>_at` match table keys on, plus the [`WeightKind`]
/// selecting the accessor bundle. The unrolled `layer` is supplied
/// per-source (it is the FUF's former loop-var), so this is layer-agnostic.
#[derive(Clone, Debug)]
pub struct WeightLocInfo {
    pub bucket: u32,
    pub op_idx: u32,
    pub slot: u32,
    pub kind: WeightKind,
}

/// Build `accessor-base-name → WeightLocInfo` from a decode bucket's
/// backbone + lm_head `weight_slots`, reproducing
/// [`crate::codegen::emit_weight_accessors_impl`]'s per-(op_idx, kind)
/// slot-ordinal walk EXACTLY — both route the kind→method key through
/// [`weight_kind_accessor_method`] — so a wavefront weight source resolves
/// to the SAME `(bucket, op_idx, slot)` match arm the non-mega decode path
/// uses. `bb_bucket_id` is the backbone tape-index (`2*ci`); lm_head is
/// `bb_bucket_id + 1`.
///
/// First writer wins per base: within one (loop-compressed) decode body
/// each accessor base occurs once per `(op_idx, kind)`, and every arm for a
/// given base calls the same layer-parametric `self.<base>(layer)` getter,
/// so any one arm is interchangeable given the right `layer`.
pub fn build_base_to_loc(
    backbone: &[Vec<WeightSlot>],
    lm_head: &[Vec<WeightSlot>],
    bb_bucket_id: u32,
) -> HashMap<String, WeightLocInfo> {
    let mut map: HashMap<String, WeightLocInfo> = HashMap::new();
    let mut walk = |slots_arr: &[Vec<WeightSlot>], bucket: u32| {
        for (op_idx, slots) in slots_arr.iter().enumerate() {
            // Per-(op_idx, method) ordinal — identical to the runtime match
            // table's `slot` axis.
            let mut counts: HashMap<&'static str, u32> = HashMap::new();
            for slot in slots {
                let key = weight_kind_accessor_method(&slot.kind);
                let n = counts.entry(key).or_insert(0);
                let ordinal = *n;
                *n += 1;
                map.entry(slot.base.to_string())
                    .or_insert_with(|| WeightLocInfo {
                        bucket,
                        op_idx: op_idx as u32,
                        slot: ordinal,
                        kind: slot.kind.clone(),
                    });
            }
        }
    };
    walk(backbone, bb_bucket_id);
    walk(lm_head, bb_bucket_id + 1);
    map
}

/// Recover `(group_size, bits)` for a weight from its FUF `storage`
/// annotation (the per-model `annotate_storage_formats` pass). `None` for
/// dense / unquantized weights.
fn affine_gs_bits(fuf: &Fuf, id: u32, index: Option<u64>) -> Option<(u32, u32)> {
    for node in &fuf.nodes {
        for inp in &node.inputs {
            if let FufInput::Weight {
                id: wid,
                index: widx,
                storage,
            } = inp
                && wid.0 == id
                && *widx == index
            {
                return match storage {
                    StorageFormat::Affine { bits, group_size } => Some((*group_size, *bits)),
                    _ => None,
                };
            }
        }
    }
    None
}

/// Outcome of resolving the wavefront weight sources to real runtime
/// locators — the macro-time dump reports this to verify every weight keys
/// a real `WeightAccessors` match arm (the whole point of macro-emission
/// increment 2).
#[derive(Debug, Default, Clone)]
pub struct ResolutionReport {
    /// Total `SourceBinding::Weight` sources.
    pub weights_total: usize,
    /// How many resolved to a real `WeightLoc` (base found in the decode
    /// bucket's `weight_slots`).
    pub weights_resolved: usize,
    /// Accessor base names that did NOT resolve (sorted, deduped) — each
    /// fell back to a placeholder locator. Empty ⇒ every weight resolved.
    pub unresolved: Vec<String>,
}

/// Map the bridge's [`SourceBinding`] manifest to the serializer's
/// [`SourceDesc`] vector (parallel to `input.sources`), so a solved decode
/// FUF flows all the way to a [`ferrite_wavefront::mega::MegaProgram`] with
/// REAL per-weight [`WeightLoc`]s.
///
/// Each `SourceBinding::Weight{id, index}` is correlated to the non-mega
/// decode path's `WeightAccessors` match arm: `weight_field_name` +
/// `split_base_layer` recover the accessor base, looked up in `base_to_loc`
/// (built from the lowered decode bucket's `weight_slots`) for the real
/// `(bucket, op_idx, slot)`; `index` supplies the unrolled `layer`. Quant
/// `(group_size, bits)` come from the FUF weight's resolved `storage`.
/// Cos/sin resolve through the rotary `CosSin` accessor; the embedded
/// hidden state is a host-gathered activation (runtime input — locator is
/// a placeholder until increment 3 wires it).
pub fn build_source_descs(
    program: &Program,
    fuf: &Fuf,
    input: &LoweringInput,
    bindings: &[SourceBinding],
    base_to_loc: &HashMap<String, WeightLocInfo>,
) -> (Vec<SourceDesc>, ResolutionReport) {
    // A source read as a Gemm's weight (input 1) is a quantized linear
    // weight; any other `Weight` binding is a dense gain (rmsnorm).
    let mut gemm_weight: HashSet<usize> = HashSet::new();
    for od in &input.ops {
        if matches!(od.op, LoweredOp::Gemm { .. })
            && let Some(InputRef::Ext(e)) = od.inputs.get(1)
        {
            gemm_weight.insert(*e);
        }
    }

    // The rotary cache resolves through `cos_sin_at`; its base
    // (`rotary{,_local}`) was injected on the rope-consuming op in
    // `lower_bucket`. Cos and sin share this one locator.
    let cos_sin_loc = base_to_loc
        .values()
        .find(|li| li.kind == WeightKind::CosSin)
        .cloned();

    let mut report = ResolutionReport::default();
    let mut unresolved: BTreeSet<String> = BTreeSet::new();

    let descs = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| {
            // Fallback locator — used only when a real one can't be
            // recovered, so the pipeline still serializes for the dump.
            let placeholder = WeightLoc {
                layer: 0,
                bucket: 0,
                op_idx: i as u32,
                slot: 0,
            };
            let bref = |bundle, role, loc| BufferRef::Weight { bundle, role, loc };
            match b {
                // Embed is a host gather ("embed-as-source", plan decision):
                // the embedded hidden is the runtime-supplied first activation,
                // NOT the embedding weight. The Metal glue binds it to the
                // per-op forward's embed output.
                SourceBinding::EmbeddedHidden => SourceDesc::Dense {
                    buffer: BufferRef::EmbeddedHidden,
                    elem: 2,
                },
                SourceBinding::Cos | SourceBinding::Sin => {
                    let loc = cos_sin_loc
                        .as_ref()
                        .map(|li| WeightLoc {
                            layer: 0,
                            bucket: li.bucket,
                            op_idx: li.op_idx,
                            slot: li.slot,
                        })
                        .unwrap_or(placeholder);
                    SourceDesc::Dense {
                        buffer: bref(WeightBundle::CosSin, WeightRole::Weight, loc),
                        elem: 2,
                    }
                }
                SourceBinding::PrefixK { layer } => SourceDesc::PrefixK {
                    layer: *layer as u32,
                },
                SourceBinding::PrefixV { layer } => SourceDesc::PrefixV {
                    layer: *layer as u32,
                },
                SourceBinding::Weight { id, index } => {
                    report.weights_total += 1;
                    let name = weight_field_name(program, WeightId(*id), *index);
                    let (base, _layer) = split_base_layer(&name.to_string());
                    let layer = index.unwrap_or(0) as u32;
                    let loc = match base_to_loc.get(&base) {
                        Some(li) => {
                            report.weights_resolved += 1;
                            WeightLoc {
                                layer,
                                bucket: li.bucket,
                                op_idx: li.op_idx,
                                slot: li.slot,
                            }
                        }
                        None => {
                            // Fused-fallback: cuda's solver picks fused
                            // impls (FusedQkvRopeCacheImpl,
                            // FusedGateUpSiluMulImpl) whose accessor base
                            // is the `__fused__`-joined sorted list of
                            // constituent unfused bases. The unfused FUF
                            // (q/k/v_proj, gate/up_proj) refs miss in
                            // base_to_loc; look for any fused key that
                            // contains `base` as a `__fused__`-token and
                            // resolve to the fused locator. Per-position
                            // byte-offset slicing happens at dispatch
                            // time (the runtime dispatcher knows each
                            // constituent's row count from
                            // CanonicalParams::Q_SIZE / KV_SIZE /
                            // INTERMEDIATE_SIZE).
                            let fused_key = base_to_loc.keys().find(|k| {
                                k.contains("__fused__")
                                    && k.split("__fused__").any(|tok| tok == base)
                            });
                            match fused_key {
                                Some(k) => {
                                    let li = &base_to_loc[k];
                                    report.weights_resolved += 1;
                                    WeightLoc {
                                        layer,
                                        bucket: li.bucket,
                                        op_idx: li.op_idx,
                                        slot: li.slot,
                                    }
                                }
                                None => {
                                    unresolved.insert(base.clone());
                                    placeholder
                                }
                            }
                        }
                    };
                    if gemm_weight.contains(&i) {
                        // 4-bit affine linear: the qmv triple. gs/bits from the
                        // FUF weight's resolved storage; the three roles share
                        // the locator (one `linear_at` returns the LinearLayer,
                        // role selects packed / scales / biases).
                        let (group_size, bits) =
                            affine_gs_bits(fuf, *id, *index).unwrap_or((64, 4));
                        SourceDesc::QuantWeight {
                            weight: bref(WeightBundle::LinearLayer, WeightRole::Weight, loc),
                            scales: bref(WeightBundle::LinearLayer, WeightRole::AffineScales, loc),
                            biases: bref(WeightBundle::LinearLayer, WeightRole::AffineBiases, loc),
                            group_size,
                            bits,
                            scale_elem: 2,
                        }
                    } else {
                        // Dense rmsnorm gain.
                        SourceDesc::Dense {
                            buffer: bref(WeightBundle::RmsNorm, WeightRole::Weight, loc),
                            elem: 2,
                        }
                    }
                }
            }
        })
        .collect();

    report.unresolved = unresolved.into_iter().collect();
    (descs, report)
}

/// Compute dump stats for a lowered forward (the drive logs these).
pub fn stats(fuf: &Fuf, asn: &Assignment, lowered: &LoweredDecode) -> BridgeStats {
    let input = &lowered.input;
    let mut hist: HashMap<&'static str, usize> = HashMap::new();
    for od in &input.ops {
        let name = match od.op {
            LoweredOp::Gemm { .. } => "Gemm",
            LoweredOp::RmsNorm { .. } => "RmsNorm",
            LoweredOp::Silu => "Silu",
            LoweredOp::Mul => "Mul",
            LoweredOp::SiluMul => "SiluMul",
            LoweredOp::Add => "Add",
            LoweredOp::RopeRotate { .. } => "RopeRotate",
            LoweredOp::RopeAppend { .. } => "RopeAppend",
            LoweredOp::AttnDecode { .. } => "AttnDecode",
        };
        *hist.entry(name).or_default() += 1;
    }
    let mut op_histogram: Vec<(&'static str, usize)> = hist.into_iter().collect();
    op_histogram.sort_by_key(|(n, _)| *n);
    let weight_sources = lowered
        .bindings
        .iter()
        .filter(|b| matches!(b, SourceBinding::Weight { .. }))
        .count();
    let prefix_sources = lowered
        .bindings
        .iter()
        .filter(|b| {
            matches!(
                b,
                SourceBinding::PrefixK { .. } | SourceBinding::PrefixV { .. }
            )
        })
        .count();
    BridgeStats {
        fuf_tiles: fuf.len(),
        subgraphs: asn.num_subgraphs(),
        sources: input.sources.len(),
        ops: input.ops.len(),
        op_histogram,
        weight_sources,
        prefix_sources,
    }
}
