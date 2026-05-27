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
//! becomes a read-only `Source`, `rope_append` splits into two
//! `RopeRotate`s with the un-roped V aliasing the V-proj output,
//! `attention` reads the prefix KV cache as `Source` segments + the new
//! token as `Sub` edges, `lm_head` is the result — is mirrored exactly
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

use std::collections::{BTreeMap, HashMap};

use ferrite_wavefront::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
use ferrite_wavefront::mega::SourceDesc;
use ferrite_wavefront::subtile::SourceShape;
use ferrite_wavefront::subtile_ir::{BufferRef, WeightBundle, WeightLoc, WeightRole};

use crate::classified::{ExternKind, OpKind};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::impl_lib::{attention_scale_for, eval_shape_with, gemm_nk_from_fuf};
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

    for node in &fuf.nodes {
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
            // rope_append(q, k, v, positions, rotary, kv_cache) → (q', k', v):
            // two RopeRotate nodes; the un-roped V (slot 2) aliases the
            // V-proj output. The new K/V reach attention as Sub edges —
            // no cache round-trip (decision #4).
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
                let (cos, sin) = bx.cos_sin();
                let qi = bx.push_op(
                    LoweredOp::RopeRotate { head_dim },
                    vec![q, InputRef::Ext(cos), InputRef::Ext(sin)],
                );
                let ki = bx.push_op(
                    LoweredOp::RopeRotate { head_dim },
                    vec![k, InputRef::Ext(cos), InputRef::Ext(sin)],
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

/// Map the bridge's [`SourceBinding`] manifest to the serializer's
/// [`SourceDesc`] vector (parallel to `input.sources`), so a solved decode FUF
/// can flow all the way to a [`ferrite_wavefront::mega::MegaProgram`]. The
/// *structural* shape — which source is a quantized linear weight vs a dense
/// rmsnorm gain / rotary row / prefix-cache half — is recovered from how each
/// source is consumed. The per-weight runtime [`WeightLoc`] + quant params are
/// PLACEHOLDERS here (unique `op_idx` per source); the REAL locators come from
/// the lowered tape's `weight_slots` (codegen, where `linear_at` is built).
/// This is enough to prove the compile-time pipeline (`lower_region` →
/// `region_schedule` → `mega::serialize`) runs on the real FUF and yields a
/// structurally sound MegaProgram.
pub fn build_source_descs(input: &LoweringInput, bindings: &[SourceBinding]) -> Vec<SourceDesc> {
    // A source read as a Gemm's weight (input 1) is a quantized linear weight;
    // any other `Weight` binding is a dense gain (rmsnorm).
    let mut gemm_weight: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for od in &input.ops {
        if matches!(od.op, LoweredOp::Gemm { .. })
            && let Some(InputRef::Ext(e)) = od.inputs.get(1)
        {
            gemm_weight.insert(*e);
        }
    }
    bindings
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let bref = |bundle, role| BufferRef::Weight {
                bundle,
                role,
                loc: WeightLoc {
                    layer: 0,
                    bucket: 0,
                    op_idx: i as u32,
                    slot: 0,
                },
            };
            match b {
                SourceBinding::EmbeddedHidden => SourceDesc::Dense {
                    buffer: bref(WeightBundle::Embedding, WeightRole::Weight),
                    elem: 2,
                },
                SourceBinding::Cos | SourceBinding::Sin => SourceDesc::Dense {
                    buffer: bref(WeightBundle::CosSin, WeightRole::Weight),
                    elem: 2,
                },
                SourceBinding::PrefixK { layer } => SourceDesc::PrefixK {
                    layer: *layer as u32,
                },
                SourceBinding::PrefixV { layer } => SourceDesc::PrefixV {
                    layer: *layer as u32,
                },
                SourceBinding::Weight { .. } if gemm_weight.contains(&i) => {
                    SourceDesc::QuantWeight {
                        weight: bref(WeightBundle::LinearLayer, WeightRole::Weight),
                        scales: bref(WeightBundle::LinearLayer, WeightRole::AffineScales),
                        biases: bref(WeightBundle::LinearLayer, WeightRole::AffineBiases),
                        group_size: 64,
                        bits: 4,
                        scale_elem: 2,
                    }
                }
                SourceBinding::Weight { .. } => SourceDesc::Dense {
                    buffer: bref(WeightBundle::RmsNorm, WeightRole::Weight),
                    elem: 2,
                },
            }
        })
        .collect()
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
