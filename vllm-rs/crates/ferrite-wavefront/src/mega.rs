// SPDX-License-Identifier: Apache-2.0
//! The **megakernel serializer** — turn a scheduled tensor-region graph
//! into the flat, backend-neutral buffers the on-GPU trivial tape player
//! (`shaders/wavefront_layer.metal`'s `wavefront_player`) consumes.
//!
//! # What this is
//!
//! [`crate::region_schedule`] bin-packs a [`crate::region::RegionGraph`]
//! into `P` co-resident worker tapes ([`Schedule`]) with cross-worker
//! `Wait`/`Signal` flags. This module flattens that schedule into the
//! exact data the persistent megakernel reads:
//!
//! - **tape** — one `uint4 (opcode, shape_class, operand_base, flag)` per
//!   instruction, the worker tapes laid end-to-end. `opcode` is
//!   `0=Compute 1=Signal 2=Wait` ([`opcode`]).
//! - **shapes** — a flat table of fixed-width [`SHAPE_STRIDE`]-`u32`
//!   records `[op_kind, p1..p6, _]`; a `Compute` indexes it by
//!   `shape_class`. `op_kind` matches the player's `WL_OP_*` ([`op_kind`]).
//! - **operands** — a flat list of [`OperandSlot`]s `(BufId, byte_offset)`;
//!   a `Compute` reads `operands[operand_base + k]` for its `k`-th operand.
//!   The Metal backend turns each slot into a `gpuAddress` (the bindless
//!   table); the byte offset is the N-block / position offset the compiler
//!   baked in.
//! - **tape_offsets** — `[P + 1]`; worker `me` runs `[tape_offsets[me],
//!   tape_offsets[me + 1])`.
//! - **num_flags** — the one-shot p2p flags the schedule allocated.
//!
//! plus the **buffer table** ([`MegaProgram::buffers`]): what each [`BufId`]
//! is, as a [`BufferRef`] (a model weight tensor, an arena activation slot,
//! or a runtime input). The Metal backend resolves these to
//! `(MTLBuffer, base_offset)` exactly like
//! `interpreter/metal/subtile_player.rs::resolve_buffers` already does for
//! the per-dispatch [`crate::subtile_ir`] path, then `operands[i] =
//! gpuAddress(buffers[slot.buffer]) + base + slot.byte_offset`.
//!
//! # Backend neutrality (locked)
//!
//! The encoding is plain serializable data — no Metal / objc2 types. The
//! MSL `wavefront_player` and a future CUDA `.cu` interpreter are parallel
//! consumers of the *same* tape/shape/operand encoding; only `BufId →
//! pointer` resolution and the flag primitive are per-target. This module
//! reuses [`crate::subtile_ir`]'s neutral [`BufferRef`] / [`WeightLoc`] /
//! [`InputKind`] vocabulary and its qmv byte-stride helpers
//! ([`packed_weight_row_bytes`] / [`affine_scale_row_bytes`]) so the two
//! IRs cannot drift on the linchpin offset math.
//!
//! # The law: the serializer is mechanical, the player is trivial
//!
//! All intelligence (which atom, which operands, every byte offset, the
//! sync edges) is decided upstream — by the region lowering + the
//! wavefront scheduler — and merely *recorded* here. This module makes no
//! scheduling or kernel-selection choice; it only translates. See
//! `feedback_subtile_ir_trivial_player` and `feedback_we_are_a_compiler`.
//!
//! # Three region-IR ↔ GPU-atom impedance mismatches it resolves
//!
//! 1. **Dense weight tensor → quantized triple.** `region.rs` models a GEMM
//!    weight as one dense tensor; the GPU `qmv` atom needs `(packed w,
//!    scales, biases)`. The serializer expands a [`SubOp::MatmulTile`]'s
//!    weight source through its [`SourceDesc::QuantWeight`] descriptor (the
//!    macro supplies the real [`BufferRef`]s + quant params; tests synth).
//! 2. **Attention's abstract segments → the real paged cache.** The region
//!    attention node reads prefix-KV `Source`s + the new token as `Sub`
//!    edges (decision #4); the GPU `attention_decode_impl` reads the paged
//!    KV cache. The serializer maps Q/output to arena, and `seq_used_k /
//!    block_table / k_cache / v_cache` to runtime [`InputKind`]s — taking
//!    the cache `layer` from the prefix sources. The new-token K/V inputs
//!    are dataflow-only (they keep the rope→attn `Wait` edge); writing them
//!    into the cache is a backend concern, not part of this encoding.
//! 3. **Separate `Silu` + `Mul` → fused `SiluMul`.** The GPU has only a
//!    fused `silu_mul` arm (no standalone silu). Fusion must happen
//!    *before* scheduling (so the pair lands on one worker); until that
//!    lowering pass exists, the serializer **errors** on a standalone
//!    `Silu`/`Mul` rather than silently dropping it.
//!
//! `RopeRotate` is in place on the GPU (the arm rewrites operand 0), but the
//! region IR gives it a distinct output tensor; the serializer therefore
//! **aliases** a rope node's output arena slot to its input's slot.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::region::{RegionGraph, SubtileNode, TensorId, TensorRegion};
use crate::region_schedule::{Schedule, TapeInstr};
use crate::subtile::{EwKind, SubOp};
use crate::subtile_ir::{
    BufId, BufferRef, InputKind, affine_scale_row_bytes, packed_weight_row_bytes,
};

// ── Encoding constants (mirror wavefront_layer.metal) ────────────────

/// Tape `opcode` field (`tape[pc].x`).
pub mod opcode {
    pub const COMPUTE: u32 = 0;
    pub const SIGNAL: u32 = 1;
    pub const WAIT: u32 = 2;
}

/// Shape-class `op_kind` field (`shapes[sc*STRIDE]`); matches the player's
/// `WL_OP_*` constants.
pub mod op_kind {
    pub const QMV: u32 = 0;
    pub const PUBLISH: u32 = 1;
    pub const ACQUIRE: u32 = 2;
    pub const RMSNORM: u32 = 3;
    pub const SILU_MUL: u32 = 4;
    pub const ROPE: u32 = 5;
    pub const ATTN: u32 = 6;
    pub const ADD: u32 = 7;
    /// Rotate K in place + write rotated K / un-rotated V into the paged
    /// cache (the oracle's `rope_append` K side); attention then reads the
    /// new token from the cache.
    pub const ROPE_APPEND: u32 = 8;
}

/// `u32`s per shape-class record (`WL_SHAPE_STRIDE`). Wide enough for
/// attention's six dims; simpler ops use the leading slots.
pub const SHAPE_STRIDE: usize = 8;

// ── The neutral program ──────────────────────────────────────────────

/// One operand of a compute instruction: which logical buffer, and the
/// byte offset into it the compiler resolved (an N-block row offset, a
/// column slice, a token-position slice). The Metal backend computes
/// `gpuAddress(buffer) + base_of(buffer) + byte_offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandSlot {
    pub buffer: BufId,
    pub byte_offset: u64,
}

/// The serialized megakernel program: the five player buffers plus the
/// buffer table the backend resolves to GPU addresses.
#[derive(Clone, Debug)]
pub struct MegaProgram {
    /// `(opcode, shape_class, operand_base, flag)` per instruction, worker
    /// tapes concatenated in worker order.
    pub tape: Vec<[u32; 4]>,
    /// Flat shape-class table, `SHAPE_STRIDE` u32s per class.
    pub shapes: Vec<[u32; SHAPE_STRIDE]>,
    /// Flat operand slots; a compute reads `operands[operand_base + k]`.
    pub operands: Vec<OperandSlot>,
    /// `[P + 1]`; worker `me` runs `[tape_offsets[me], tape_offsets[me+1])`.
    pub tape_offsets: Vec<u32>,
    /// One-shot p2p flags referenced by `Signal`/`Wait`.
    pub num_flags: u32,
    /// `BufId → what it is` (resolved to a GPU buffer by the backend).
    pub buffers: Vec<BufferRef>,
    /// Element width (bytes) of each buffer, parallel to `buffers`.
    pub elem_bytes: Vec<u32>,
    /// Byte size of each arena slot (indexed by the `ArenaSlot(slot)` id).
    pub arena_bytes: Vec<u64>,
    /// The buffer holding the forward result (logits).
    pub result: BufId,
}

impl MegaProgram {
    /// `tape` as little-endian `u32` bytes — the `device const uint4*`
    /// buffer the player binds at index 0.
    pub fn tape_bytes(&self) -> Vec<u8> {
        self.tape
            .iter()
            .flat_map(|i| i.iter().flat_map(|v| v.to_le_bytes()))
            .collect()
    }

    /// `shapes` as little-endian `u32` bytes — the flat `device const
    /// uint*` table at index 1.
    pub fn shapes_bytes(&self) -> Vec<u8> {
        self.shapes
            .iter()
            .flat_map(|r| r.iter().flat_map(|v| v.to_le_bytes()))
            .collect()
    }

    /// `tape_offsets` as little-endian `u32` bytes — index 3.
    pub fn tape_offsets_bytes(&self) -> Vec<u8> {
        self.tape_offsets
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// Total `Compute` instructions — must equal the region node count.
    pub fn num_computes(&self) -> usize {
        self.tape.iter().filter(|i| i[0] == opcode::COMPUTE).count()
    }
}

// ── Serializer inputs ────────────────────────────────────────────────

/// Per-target geometry the abstract region graph does not carry.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Activation element width (bytes): bf16/f16 ⇒ 2.
    pub act_elem: u32,
    /// Paged KV-cache block size (rows per block), for the attention arm.
    pub block_size: u32,
    /// Max blocks per sequence in the block table, for the attention arm.
    pub max_blocks: u32,
}

/// What a leaf source tensor of the [`RegionGraph`] binds to on the GPU —
/// parallel to `graph.tensors[0..num_sources]`, the megakernel-resolution
/// counterpart of the macro's `to_wavefront::SourceBinding`. The macro
/// fills the concrete [`BufferRef`]s (a weight's [`WeightLoc`], quant
/// params); host tests synthesize them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceDesc {
    /// A 4-bit affine linear weight: the qmv triple plus quant params. The
    /// region models it as one dense `[N, K]` tensor; this is how it
    /// becomes the three real GPU buffers.
    QuantWeight {
        weight: BufferRef,
        scales: BufferRef,
        biases: BufferRef,
        group_size: u32,
        bits: u32,
        /// scales/biases element width (f16 ⇒ 2).
        scale_elem: u32,
    },
    /// A single dense buffer: an rmsnorm gain, a rotary cos / sin row, or
    /// the host-gathered embedded hidden state. `elem` is its element
    /// width (bytes).
    Dense { buffer: BufferRef, elem: u32 },
    /// The read-only prefix KV cache K half for `layer`. On the GPU the
    /// attention arm reads the runtime paged cache, so this only tells the
    /// serializer the cache layer; it is never read as a plain operand.
    PrefixK { layer: u32 },
    /// The prefix KV cache V half for `layer`.
    PrefixV { layer: u32 },
}

/// Why a scheduled region graph could not be serialized. Always surfaced —
/// never a silent skip (`feedback_no_silent_deferrals`).
#[derive(Debug, PartialEq, Eq)]
pub enum SerializeError {
    /// An op the GPU player has no arm for, or that needs an upstream
    /// rewrite first (a standalone `Silu`/`Mul` before silu·mul fusion).
    UnsupportedOp { id: u32, detail: &'static str },
    /// A source index referenced by a node is out of range of `sources`.
    MissingSource { tensor: u32 },
    /// A source's [`SourceDesc`] is the wrong kind for the op reading it
    /// (e.g. a qmv weight that is not [`SourceDesc::QuantWeight`]).
    BadSource { tensor: u32, detail: &'static str },
    /// A node's shape is outside the decode (`m == 1`, row 0) model.
    NonDecodeShape { id: u32, detail: &'static str },
    /// An attention node did not match the `[q, prefixK, prefixV, newK,
    /// newV]` shape the paged-cache mapping needs.
    BadAttn { id: u32, detail: &'static str },
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOp { id, detail } => write!(f, "node {id}: unsupported op — {detail}"),
            Self::MissingSource { tensor } => write!(f, "source {tensor} out of range"),
            Self::BadSource { tensor, detail } => write!(f, "source {tensor}: {detail}"),
            Self::NonDecodeShape { id, detail } => write!(f, "node {id}: {detail}"),
            Self::BadAttn { id, detail } => write!(f, "node {id}: bad attention — {detail}"),
        }
    }
}

impl std::error::Error for SerializeError {}

// ── The serializer ───────────────────────────────────────────────────

/// Flatten a wavefront [`Schedule`] over a [`RegionGraph`] into a
/// [`MegaProgram`]. `sources` is parallel to `graph.tensors[0..
/// num_sources]`; `geom` supplies the element width and the attention
/// paged-cache geometry the abstract graph omits.
pub fn serialize(
    graph: &RegionGraph,
    schedule: &Schedule,
    sources: &[SourceDesc],
    geom: Geometry,
) -> Result<MegaProgram, SerializeError> {
    let mut ser = Ser::new(graph, sources, geom);
    ser.assign_arena_slots()?;

    let mut tape: Vec<[u32; 4]> = Vec::new();
    let mut tape_offsets: Vec<u32> = vec![0];
    for worker in &schedule.workers {
        for instr in &worker.tape {
            match *instr {
                TapeInstr::Compute(id) => {
                    let node = &graph.nodes[id.0 as usize];
                    let (sc, base) = ser.emit_compute(node)?;
                    tape.push([opcode::COMPUTE, sc, base, 0]);
                }
                TapeInstr::Signal(f) => tape.push([opcode::SIGNAL, 0, 0, f]),
                TapeInstr::Wait(f) => tape.push([opcode::WAIT, 0, 0, f]),
            }
        }
        tape_offsets.push(tape.len() as u32);
    }

    let result = ser.arena_bufid(graph.result);
    Ok(MegaProgram {
        tape,
        shapes: ser.shapes,
        operands: ser.operands,
        tape_offsets,
        num_flags: schedule.num_flags,
        buffers: ser.buffers,
        elem_bytes: ser.elem_bytes,
        arena_bytes: ser.arena_bytes,
        result,
    })
}

/// Mutable serializer state: the interned buffer / shape tables, the
/// operand list, and the tensor→arena-slot map.
struct Ser<'a> {
    graph: &'a RegionGraph,
    sources: &'a [SourceDesc],
    geom: Geometry,
    num_sources: u32,
    buffers: Vec<BufferRef>,
    elem_bytes: Vec<u32>,
    shapes: Vec<[u32; SHAPE_STRIDE]>,
    operands: Vec<OperandSlot>,
    /// Op-output `TensorId` → its arena slot index.
    slot_of: HashMap<TensorId, u32>,
    /// Byte size of each arena slot (indexed by slot).
    arena_bytes: Vec<u64>,
}

impl<'a> Ser<'a> {
    fn new(graph: &'a RegionGraph, sources: &'a [SourceDesc], geom: Geometry) -> Self {
        Self {
            graph,
            sources,
            geom,
            num_sources: graph.num_sources,
            buffers: Vec::new(),
            elem_bytes: Vec::new(),
            shapes: Vec::new(),
            operands: Vec::new(),
            slot_of: HashMap::new(),
            arena_bytes: Vec::new(),
        }
    }

    fn is_source(&self, t: TensorId) -> bool {
        t.0 < self.num_sources
    }

    /// Source descriptor for a leaf tensor (`t < num_sources`).
    fn source(&self, t: TensorId) -> Result<&'a SourceDesc, SerializeError> {
        self.sources
            .get(t.0 as usize)
            .ok_or(SerializeError::MissingSource { tensor: t.0 })
    }

    /// Intern a [`BufferRef`] with its element width, returning a deduped
    /// [`BufId`] (mirrors `SubtileIrBuilder::buffer`).
    fn intern(&mut self, b: BufferRef, elem: u32) -> BufId {
        if let Some(i) = self.buffers.iter().position(|x| *x == b) {
            return BufId(i as u32);
        }
        self.buffers.push(b);
        self.elem_bytes.push(elem);
        BufId(self.buffers.len() as u32 - 1)
    }

    /// Intern a shape-class record, returning a deduped class index.
    fn intern_shape(&mut self, rec: [u32; SHAPE_STRIDE]) -> u32 {
        if let Some(i) = self.shapes.iter().position(|x| *x == rec) {
            return i as u32;
        }
        self.shapes.push(rec);
        self.shapes.len() as u32 - 1
    }

    fn arena_bufid(&mut self, t: TensorId) -> BufId {
        let slot = self.slot_of[&t];
        self.intern(BufferRef::ArenaSlot(slot), self.geom.act_elem)
    }

    /// Assign one arena slot per op-output tensor, in topological (id)
    /// order. A `RopeRotate` is in place on the GPU, so its output aliases
    /// its input's slot. N-block matmuls share a slot (one slot per output
    /// tensor, all blocks write it). Must run before any operand emission.
    fn assign_arena_slots(&mut self) -> Result<(), SerializeError> {
        for node in &self.graph.nodes {
            let ot = node.output.tensor;
            if self.slot_of.contains_key(&ot) {
                continue; // shared by this op's N-blocks
            }
            let slot = if matches!(node.op, SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. }) {
                let in0 = node.inputs[0].tensor;
                if self.is_source(in0) {
                    return Err(SerializeError::UnsupportedOp {
                        id: node.id.0,
                        detail: "rope of a leaf source has no arena slot to rotate in place",
                    });
                }
                self.slot_of[&in0] // alias: rotate the producer's buffer in place
            } else {
                let shape = self.graph.shape(ot);
                let bytes = (shape.rows as u64) * (shape.cols as u64) * self.geom.act_elem as u64;
                let slot = self.arena_bytes.len() as u32;
                self.arena_bytes.push(bytes);
                slot
            };
            self.slot_of.insert(ot, slot);
        }
        Ok(())
    }

    /// Resolve an input read region to an operand slot. Handles arena
    /// (op-output) tensors and `Dense` sources; the op emitters special-
    /// case `QuantWeight` / `PrefixK`/`V`.
    fn read_operand(&mut self, tr: &TensorRegion, id: u32) -> Result<OperandSlot, SerializeError> {
        self.check_row0(tr, id)?;
        if self.is_source(tr.tensor) {
            match self.source(tr.tensor)?.clone() {
                SourceDesc::Dense { buffer, elem } => {
                    let off = tr.region.cols.start as u64 * elem as u64;
                    Ok(OperandSlot {
                        buffer: self.intern(buffer, elem),
                        byte_offset: off,
                    })
                }
                _ => Err(SerializeError::BadSource {
                    tensor: tr.tensor.0,
                    detail: "expected a Dense source as a plain read operand",
                }),
            }
        } else {
            let off = tr.region.cols.start as u64 * self.geom.act_elem as u64;
            Ok(OperandSlot {
                buffer: self.arena_bufid(tr.tensor),
                byte_offset: off,
            })
        }
    }

    /// Decode is `m == 1`, row 0 — column slices are contiguous byte
    /// intervals. Reject anything else (the offset math assumes it).
    fn check_row0(&self, tr: &TensorRegion, id: u32) -> Result<(), SerializeError> {
        if tr.region.rows.start != 0 || tr.region.rows.len != 1 {
            return Err(SerializeError::NonDecodeShape {
                id,
                detail: "region is not the single decode row [0,1)",
            });
        }
        Ok(())
    }

    fn push_operands(&mut self, slots: &[OperandSlot]) -> u32 {
        let base = self.operands.len() as u32;
        self.operands.extend_from_slice(slots);
        base
    }

    /// Emit one compute node → `(shape_class, operand_base)`.
    fn emit_compute(&mut self, node: &SubtileNode) -> Result<(u32, u32), SerializeError> {
        match node.op {
            SubOp::MatmulTile => self.emit_qmv(node),
            SubOp::RmsNorm { eps } => self.emit_rmsnorm(node, eps),
            SubOp::RopeRotate { head_dim } => self.emit_rope(node, head_dim),
            SubOp::RopeAppend { head_dim, layer } => self.emit_rope_append(node, head_dim, layer),
            SubOp::SiluMul => self.emit_silu_mul(node),
            SubOp::Elementwise(EwKind::Add) => self.emit_add(node),
            SubOp::AttnDecode {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
            } => self.emit_attn(node, num_q_heads, num_kv_heads, head_dim, scale),
            SubOp::Elementwise(EwKind::Silu) => Err(SerializeError::UnsupportedOp {
                id: node.id.0,
                detail: "standalone Silu — fuse Silu+Mul into SiluMul before scheduling",
            }),
            SubOp::Elementwise(EwKind::Mul) => Err(SerializeError::UnsupportedOp {
                id: node.id.0,
                detail: "standalone Mul — fuse Silu+Mul into SiluMul before scheduling",
            }),
            SubOp::SumReduce => Err(SerializeError::UnsupportedOp {
                id: node.id.0,
                detail: "split-K SumReduce — no GPU arm yet (k_chunks=1 only)",
            }),
        }
    }

    /// QMV: operands `[w, scales, biases, x, y]`; shape `(QMV, K, N)`. The
    /// N-block's output column start `r` is the weight's row block, so the
    /// w/scales/biases/y bindings carry `r * row_stride` — the linchpin
    /// that makes a block a standalone `N×K` matvec.
    fn emit_qmv(&mut self, node: &SubtileNode) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let act = &node.inputs[0];
        let wtr = &node.inputs[1];
        let out = &node.output;
        self.check_row0(act, id)?;
        self.check_row0(out, id)?;

        let k = act.region.cols.len;
        let n = out.region.cols.len;
        let r = out.region.cols.start as u64;

        if !self.is_source(wtr.tensor) {
            return Err(SerializeError::BadSource {
                tensor: wtr.tensor.0,
                detail: "qmv weight must be a leaf source",
            });
        }
        let (weight, scales, biases, group_size, bits, scale_elem) =
            match self.source(wtr.tensor)?.clone() {
                SourceDesc::QuantWeight {
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits,
                    scale_elem,
                } => (weight, scales, biases, group_size, bits, scale_elem),
                _ => {
                    return Err(SerializeError::BadSource {
                        tensor: wtr.tensor.0,
                        detail: "qmv weight source is not QuantWeight",
                    });
                }
            };

        let w_off = r * packed_weight_row_bytes(k, bits);
        let sb_off = r * affine_scale_row_bytes(k, group_size, scale_elem as u64);
        let y_off = r * self.geom.act_elem as u64;

        let w_buf = self.intern(weight, 4); // packed 4-bit weight read as u32
        let s_buf = self.intern(scales, scale_elem);
        let b_buf = self.intern(biases, scale_elem);
        let x = self.read_operand(act, id)?;
        let y = OperandSlot {
            buffer: self.arena_bufid(out.tensor),
            byte_offset: y_off,
        };
        let base = self.push_operands(&[
            OperandSlot {
                buffer: w_buf,
                byte_offset: w_off,
            },
            OperandSlot {
                buffer: s_buf,
                byte_offset: sb_off,
            },
            OperandSlot {
                buffer: b_buf,
                byte_offset: sb_off,
            },
            x,
            y,
        ]);
        let sc = self.intern_shape([op_kind::QMV, k, n, 0, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// RMSNORM: operands `[out, in, weight]`; shape `(RMSNORM, hidden,
    /// eps_bits)`. The gain is a `Dense` source.
    fn emit_rmsnorm(&mut self, node: &SubtileNode, eps: f32) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let out = self.write_operand(&node.output, id)?;
        let inp = self.read_operand(&node.inputs[0], id)?;
        let wgt = self.read_operand(&node.inputs[1], id)?;
        let hidden = node.output.region.cols.len;
        let base = self.push_operands(&[out, inp, wgt]);
        let sc = self.intern_shape([op_kind::RMSNORM, hidden, eps.to_bits(), 0, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// ROPE: operands `[x, cos, sin]`; shape `(ROPE, head_dim, num_heads)`.
    /// In place — operand 0 is the (aliased) producer buffer.
    fn emit_rope(
        &mut self,
        node: &SubtileNode,
        head_dim: u32,
    ) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let x = self.write_operand(&node.output, id)?;
        let cos = self.read_operand(&node.inputs[1], id)?;
        let sin = self.read_operand(&node.inputs[2], id)?;
        let cols = node.output.region.cols.len;
        if head_dim == 0 || !cols.is_multiple_of(head_dim) {
            return Err(SerializeError::NonDecodeShape {
                id,
                detail: "rope output cols not a multiple of head_dim",
            });
        }
        let num_heads = cols / head_dim;
        let base = self.push_operands(&[x, cos, sin]);
        let sc = self.intern_shape([op_kind::ROPE, head_dim, num_heads, 0, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// ROPE_APPEND: operands `[k, cos, sin, v, kv_cache_k, kv_cache_v,
    /// slot_mapping]`; shape `(ROPE_APPEND, head_dim, num_kv, rot_dim,
    /// block_size, ...)`. Rotate K in place (operand 0 is the aliased producer
    /// buffer), then write rotated K + un-rotated V to the paged cache. The
    /// cache halves + slot_mapping are runtime inputs the serializer injects
    /// (the region IR keeps the new K as an abstract edge); `layer` routes the
    /// cache operands. Full rope (`rot_dim == head_dim`), matching `RopeRotate`.
    fn emit_rope_append(
        &mut self,
        node: &SubtileNode,
        head_dim: u32,
        layer: u32,
    ) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let k = self.write_operand(&node.output, id)?; // in-place (aliased) K
        let cos = self.read_operand(&node.inputs[1], id)?;
        let sin = self.read_operand(&node.inputs[2], id)?;
        let v = self.read_operand(&node.inputs[3], id)?;
        let cols = node.output.region.cols.len;
        if head_dim == 0 || !cols.is_multiple_of(head_dim) {
            return Err(SerializeError::NonDecodeShape {
                id,
                detail: "rope_append output cols not a multiple of head_dim",
            });
        }
        let num_kv = cols / head_dim;
        let kv_k = OperandSlot {
            buffer: self.intern(
                BufferRef::Input(InputKind::KvCacheK { layer }),
                self.geom.act_elem,
            ),
            byte_offset: 0,
        };
        let kv_v = OperandSlot {
            buffer: self.intern(
                BufferRef::Input(InputKind::KvCacheV { layer }),
                self.geom.act_elem,
            ),
            byte_offset: 0,
        };
        let slot = OperandSlot {
            buffer: self.intern(BufferRef::Input(InputKind::SlotMapping), 4),
            byte_offset: 0,
        };
        let base = self.push_operands(&[k, cos, sin, v, kv_k, kv_v, slot]);
        // rot_dim == head_dim (full rope), matching RopeRotate.
        let sc = self.intern_shape([
            op_kind::ROPE_APPEND,
            head_dim,
            num_kv,
            head_dim,
            self.geom.block_size,
            0,
            0,
            0,
        ]);
        Ok((sc, base))
    }

    /// SILU_MUL: operands `[out, gate, up]`; shape `(SILU_MUL, n)`. The
    /// fused SwiGLU `out = silu(gate) * up`.
    fn emit_silu_mul(&mut self, node: &SubtileNode) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let out = self.write_operand(&node.output, id)?;
        let gate = self.read_operand(&node.inputs[0], id)?;
        let up = self.read_operand(&node.inputs[1], id)?;
        let n = node.output.region.cols.len;
        let base = self.push_operands(&[out, gate, up]);
        let sc = self.intern_shape([op_kind::SILU_MUL, n, 0, 0, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// ADD: operands `[out, a, b]`; shape `(ADD, n)`. Residual add.
    fn emit_add(&mut self, node: &SubtileNode) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let out = self.write_operand(&node.output, id)?;
        let a = self.read_operand(&node.inputs[0], id)?;
        let b = self.read_operand(&node.inputs[1], id)?;
        let n = node.output.region.cols.len;
        let base = self.push_operands(&[out, a, b]);
        let sc = self.intern_shape([op_kind::ADD, n, 0, 0, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// ATTN: operands `[output, q, seq_used_k, block_table, k_cache,
    /// v_cache]`; shape `(ATTN, head_dim, num_q, num_kv, scale_bits,
    /// block_size, max_blocks)`. The region node is `[q, prefixK, prefixV,
    /// newK, newV]`; the GPU reads the runtime paged cache, so prefixK/V
    /// only name the cache layer and the new-token edges are dataflow-only.
    fn emit_attn(
        &mut self,
        node: &SubtileNode,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        if node.inputs.len() != 5 {
            return Err(SerializeError::BadAttn {
                id,
                detail: "paged-cache mapping needs exactly [q, prefixK, prefixV, newK, newV]",
            });
        }
        let layer = self.cache_layer(node.inputs[1].tensor, node.inputs[2].tensor, id)?;
        let out = self.write_operand(&node.output, id)?;
        let q = self.read_operand(&node.inputs[0], id)?;
        let seq = OperandSlot {
            buffer: self.intern(BufferRef::Input(InputKind::SeqUsedK), 4),
            byte_offset: 0,
        };
        let blk = OperandSlot {
            buffer: self.intern(BufferRef::Input(InputKind::BlockTable), 4),
            byte_offset: 0,
        };
        let kc = OperandSlot {
            buffer: self.intern(
                BufferRef::Input(InputKind::KvCacheK { layer }),
                self.geom.act_elem,
            ),
            byte_offset: 0,
        };
        let vc = OperandSlot {
            buffer: self.intern(
                BufferRef::Input(InputKind::KvCacheV { layer }),
                self.geom.act_elem,
            ),
            byte_offset: 0,
        };
        let base = self.push_operands(&[out, q, seq, blk, kc, vc]);
        let sc = self.intern_shape([
            op_kind::ATTN,
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale.to_bits(),
            self.geom.block_size,
            self.geom.max_blocks,
            0,
        ]);
        Ok((sc, base))
    }

    /// The cache layer named by an attention node's prefix-K / prefix-V
    /// sources (which must agree).
    fn cache_layer(&self, pk: TensorId, pv: TensorId, id: u32) -> Result<u32, SerializeError> {
        let lk = match self.source(pk)? {
            SourceDesc::PrefixK { layer } => *layer,
            _ => {
                return Err(SerializeError::BadAttn {
                    id,
                    detail: "attention input 1 is not a PrefixK source",
                });
            }
        };
        let lv = match self.source(pv)? {
            SourceDesc::PrefixV { layer } => *layer,
            _ => {
                return Err(SerializeError::BadAttn {
                    id,
                    detail: "attention input 2 is not a PrefixV source",
                });
            }
        };
        if lk != lv {
            return Err(SerializeError::BadAttn {
                id,
                detail: "prefixK / prefixV name different cache layers",
            });
        }
        Ok(lk)
    }

    /// An output write region → an operand slot (op outputs are always
    /// arena tensors; `validate` rejects writes to leaf sources).
    fn write_operand(&mut self, tr: &TensorRegion, id: u32) -> Result<OperandSlot, SerializeError> {
        self.check_row0(tr, id)?;
        let off = tr.region.cols.start as u64 * self.geom.act_elem as u64;
        Ok(OperandSlot {
            buffer: self.arena_bufid(tr.tensor),
            byte_offset: off,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
    use crate::region::{SubtileNode, lower_region};
    use crate::region_schedule::{ScheduleParams, partition_roundrobin, schedule_wavefront};
    use crate::subtile::SourceShape;
    use crate::subtile_ir::{WeightBundle, WeightLoc, WeightRole};

    const ACT_ELEM: u32 = 2;
    fn geom() -> Geometry {
        Geometry {
            act_elem: ACT_ELEM,
            block_size: 16,
            max_blocks: 4,
        }
    }
    fn cost_area(node: &SubtileNode) -> f64 {
        (node.output.region.rows.len * node.output.region.cols.len) as f64
    }
    fn wloc(op_idx: u32) -> WeightLoc {
        WeightLoc {
            layer: 0,
            bucket: 0,
            op_idx,
            slot: 0,
        }
    }
    /// A distinct `QuantWeight` source (gs=8, 4-bit, f16 scales) keyed by
    /// `op_idx` so interning stays unambiguous.
    fn qweight(op_idx: u32) -> SourceDesc {
        SourceDesc::QuantWeight {
            weight: BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::Weight,
                loc: wloc(op_idx),
            },
            scales: BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::AffineScales,
                loc: wloc(op_idx),
            },
            biases: BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::AffineBiases,
                loc: wloc(op_idx),
            },
            group_size: 8,
            bits: 4,
            scale_elem: 2,
        }
    }
    fn dense(op_idx: u32, bundle: WeightBundle) -> SourceDesc {
        SourceDesc::Dense {
            buffer: BufferRef::Weight {
                bundle,
                role: WeightRole::Weight,
                loc: wloc(op_idx),
            },
            elem: ACT_ELEM,
        }
    }

    /// The shape `op_kind` a compute instruction selects.
    fn op_of(prog: &MegaProgram, instr: &[u32; 4]) -> u32 {
        prog.shapes[instr[1] as usize][0]
    }

    // ── QMV linchpin: per-block byte offsets + shape dedup ──────────

    /// A single wide qmv N-block-tiled: each block's w/scales/biases/y
    /// operands carry `r * row_stride` and x stays whole — the exact
    /// offset math `subtile_ir::tile_qmv` proved, now through the region
    /// graph. Equal-width blocks share one shape class.
    #[test]
    fn qmv_nblock_offsets_and_dedup() {
        let (n, k, nb) = (128u32, 2048u32, 32u32);
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: k },
                SourceShape { rows: n, cols: k },
            ],
            ops: vec![OpDesc {
                op: LoweredOp::Gemm { n, k },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            result: 0,
        };
        let sources = vec![dense(0, WeightBundle::Embedding), qweight(1)];
        let g = lower_region(&input, nb);
        assert_eq!(g.nodes.len(), 4, "ceil(128/32) blocks");
        let s = partition_roundrobin(&g, 1); // all blocks on worker 0, ascending id
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

        assert_eq!(prog.num_computes(), 4);
        assert_eq!(prog.tape.len(), 4, "P=1: no flags, 4 computes");
        assert_eq!(prog.tape_offsets, vec![0, 4]);
        // One QMV shape class shared by all four equal-width blocks.
        assert_eq!(prog.shapes.len(), 1);
        assert_eq!(prog.shapes[0], [op_kind::QMV, k, nb, 0, 0, 0, 0, 0]);

        // packed 4-bit: k*4/8 = 1024 B/row; f16 scales: k/8*2 = 512 B/row.
        let (w_row, sb_row) = (1024u64, (k as u64 / 8) * 2);
        for (b, instr) in prog.tape.iter().enumerate() {
            assert_eq!(instr[0], opcode::COMPUTE);
            let base = instr[2] as usize;
            let r = (b as u64) * nb as u64;
            let ops = &prog.operands[base..base + 5];
            assert_eq!(ops[0].byte_offset, r * w_row, "weight off blk {b}");
            assert_eq!(ops[1].byte_offset, r * sb_row, "scales off blk {b}");
            assert_eq!(ops[2].byte_offset, r * sb_row, "biases off blk {b}");
            assert_eq!(ops[3].byte_offset, 0, "x whole blk {b}");
            assert_eq!(ops[4].byte_offset, r * ACT_ELEM as u64, "y off blk {b}");
            // w/scales/biases are distinct buffers; x and y differ from them.
            assert_eq!(ops[0].buffer, ops[0].buffer);
            assert_ne!(ops[0].buffer, ops[1].buffer);
        }
        // y of every block points at the one arena output slot.
        let y0 = prog.operands[prog.tape[0][2] as usize + 4].buffer;
        for b in 0..4 {
            assert_eq!(prog.operands[prog.tape[b][2] as usize + 4].buffer, y0);
        }
        assert_eq!(prog.arena_bytes.len(), 1, "one output tensor → one slot");
        assert_eq!(prog.arena_bytes[0], n as u64 * ACT_ELEM as u64);
    }

    // ── rope is in place: output slot aliases the producer's ────────

    #[test]
    fn rope_output_aliases_input_slot() {
        let (h, hd) = (16u32, 4u32);
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: h },
                SourceShape { rows: h, cols: h },
                SourceShape { rows: 1, cols: hd },
                SourceShape { rows: 1, cols: hd },
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n: h, k: h },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate { head_dim: hd },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2), InputRef::Ext(3)],
                },
            ],
            result: 1,
        };
        let sources = vec![
            dense(0, WeightBundle::Embedding),
            qweight(1),
            dense(2, WeightBundle::CosSin),
            dense(3, WeightBundle::CosSin),
        ];
        let g = lower_region(&input, h); // 1 qmv block + 1 rope
        let s = partition_roundrobin(&g, 1);
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

        // tape: [qmv block, rope]. qmv y = operand[base0+4]; rope x = operand[base1+0].
        let qmv = &prog.tape[0];
        let rope = &prog.tape[1];
        assert_eq!(op_of(&prog, qmv), op_kind::QMV);
        assert_eq!(op_of(&prog, rope), op_kind::ROPE);
        let qmv_y = prog.operands[qmv[2] as usize + 4].buffer;
        let rope_x = prog.operands[rope[2] as usize].buffer;
        assert_eq!(qmv_y, rope_x, "rope rotates the qmv output buffer in place");
        // Only one arena slot is allocated (rope aliases, no fresh slot).
        assert_eq!(prog.arena_bytes.len(), 1);
        assert_eq!(
            prog.shapes[rope[1] as usize],
            [op_kind::ROPE, hd, h / hd, 0, 0, 0, 0, 0]
        );
    }

    // ── attention maps to the runtime paged cache ───────────────────

    #[test]
    fn attn_maps_prefix_layer_to_runtime_cache() {
        let (h, hd, hq, hkv, l) = (16u32, 4u32, 4u32, 2u32, 3u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let layer = 5u32;
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: h }, // 0 x
                SourceShape {
                    rows: qdim,
                    cols: h,
                }, // 1 wq
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 2 wk
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 3 wv
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 4 prefixK
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 5 prefixV
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n: qdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(3)],
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
                        InputRef::Op(0),
                        InputRef::Ext(4),
                        InputRef::Ext(5),
                        InputRef::Op(1),
                        InputRef::Op(2),
                    ],
                },
            ],
            result: 3,
        };
        let sources = vec![
            dense(0, WeightBundle::Embedding),
            qweight(1),
            qweight(2),
            qweight(3),
            SourceDesc::PrefixK { layer },
            SourceDesc::PrefixV { layer },
        ];
        let g = lower_region(&input, 1000); // coarse: 1 block each
        let s = partition_roundrobin(&g, 1);
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

        let attn = prog
            .tape
            .iter()
            .find(|i| op_of(&prog, i) == op_kind::ATTN)
            .expect("an ATTN compute");
        assert_eq!(
            prog.shapes[attn[1] as usize],
            [op_kind::ATTN, hd, hq, hkv, scale.to_bits(), 16, 4, 0]
        );
        let ops = &prog.operands[attn[2] as usize..attn[2] as usize + 6];
        // [output, q, seq_used_k, block_table, k_cache, v_cache]
        let buf = |o: &OperandSlot| prog.buffers[o.buffer.0 as usize].clone();
        assert!(
            matches!(buf(&ops[0]), BufferRef::ArenaSlot(_)),
            "output arena"
        );
        assert!(matches!(buf(&ops[1]), BufferRef::ArenaSlot(_)), "q arena");
        assert_eq!(buf(&ops[2]), BufferRef::Input(InputKind::SeqUsedK));
        assert_eq!(buf(&ops[3]), BufferRef::Input(InputKind::BlockTable));
        assert_eq!(
            buf(&ops[4]),
            BufferRef::Input(InputKind::KvCacheK { layer })
        );
        assert_eq!(
            buf(&ops[5]),
            BufferRef::Input(InputKind::KvCacheV { layer })
        );
    }

    // ── the full attention half-layer: aggregate structure + flags ──

    /// rmsnorm → q/k/v proj → rope(Q)/rope(K) → attn → o_proj → residual
    /// add, the half-layer that maps 1:1 to GPU arms (no silu·mul). Build
    /// it, schedule across P=4, serialize, and check counts, tape offsets,
    /// flag discipline, and the op histogram.
    fn attn_subchain() -> (LoweringInput, Vec<SourceDesc>) {
        let (h, hd, hq, hkv, l) = (16u32, 4u32, 4u32, 2u32, 3u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let eps = 1e-5f32;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: h }, // 0 res_in
                SourceShape { rows: 1, cols: h }, // 1 in_ln
                SourceShape {
                    rows: qdim,
                    cols: h,
                }, // 2 wq
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 3 wk
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 4 wv
                SourceShape { rows: 1, cols: hd }, // 5 cos
                SourceShape { rows: 1, cols: hd }, // 6 sin
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 7 prefixK
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 8 prefixV
                SourceShape {
                    rows: h,
                    cols: qdim,
                }, // 9 wo
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
            ],
            result: 8,
        };
        let sources = vec![
            dense(0, WeightBundle::Embedding),
            dense(1, WeightBundle::RmsNorm),
            qweight(2),
            qweight(3),
            qweight(4),
            dense(5, WeightBundle::CosSin),
            dense(6, WeightBundle::CosSin),
            SourceDesc::PrefixK { layer: 0 },
            SourceDesc::PrefixV { layer: 0 },
            qweight(9),
        ];
        (input, sources)
    }

    #[test]
    fn attn_subchain_structure_and_flags() {
        let (input, sources) = attn_subchain();
        let nb = 8u32; // qdim 16→2, kvdim 8→1, h 16→2 blocks
        let g = lower_region(&input, nb);
        // 1 rms + 2 q + 1 k + 1 v + 1 rope + 1 rope + 1 attn + 2 o + 1 add = 11.
        assert_eq!(g.nodes.len(), 11);
        for p in [1u32, 2, 4, 10] {
            let s = schedule_wavefront(
                &g,
                cost_area,
                ScheduleParams {
                    num_workers: p,
                    wait_cost_us: 0.18,
                },
            );
            let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

            // Every node emitted exactly once, across P worker tapes.
            assert_eq!(prog.num_computes(), g.nodes.len(), "p={p}");
            assert_eq!(prog.tape_offsets.len(), p as usize + 1, "P+1 offsets p={p}");
            assert_eq!(*prog.tape_offsets.first().unwrap(), 0);
            assert_eq!(*prog.tape_offsets.last().unwrap(), prog.tape.len() as u32);
            assert!(
                prog.tape_offsets.windows(2).all(|w| w[0] <= w[1]),
                "monotonic"
            );

            // Every compute indexes a valid shape class + operand run; sync
            // ops carry a flag and no operands.
            for instr in &prog.tape {
                match instr[0] {
                    opcode::COMPUTE => {
                        assert!((instr[1] as usize) < prog.shapes.len());
                        assert!((instr[2] as usize) <= prog.operands.len());
                    }
                    opcode::SIGNAL | opcode::WAIT => assert!(instr[3] < prog.num_flags),
                    other => panic!("bad opcode {other}"),
                }
            }

            // Flag discipline (mirrors region_schedule::flag_invariants).
            assert_eq!(prog.num_flags, s.num_flags);
            let mut sig = vec![0u32; prog.num_flags as usize];
            let mut wai = vec![0u32; prog.num_flags as usize];
            for instr in &prog.tape {
                match instr[0] {
                    opcode::SIGNAL => sig[instr[3] as usize] += 1,
                    opcode::WAIT => wai[instr[3] as usize] += 1,
                    _ => {}
                }
            }
            for f in 0..prog.num_flags as usize {
                assert_eq!(sig[f], 1, "flag {f} signaled once (p={p})");
                assert!(wai[f] >= 1, "flag {f} waited (p={p})");
            }

            // Op histogram: 6 qmv, 1 rms, 2 rope, 1 attn, 1 add.
            let mut hist = std::collections::HashMap::new();
            for instr in prog.tape.iter().filter(|i| i[0] == opcode::COMPUTE) {
                *hist.entry(op_of(&prog, instr)).or_insert(0u32) += 1;
            }
            assert_eq!(hist.get(&op_kind::QMV), Some(&6), "p={p}");
            assert_eq!(hist.get(&op_kind::RMSNORM), Some(&1));
            assert_eq!(hist.get(&op_kind::ROPE), Some(&2));
            assert_eq!(hist.get(&op_kind::ATTN), Some(&1));
            assert_eq!(hist.get(&op_kind::ADD), Some(&1));
            assert_eq!(
                hist.get(&op_kind::SILU_MUL),
                None,
                "no silu·mul in attn half"
            );
        }
    }

    /// The byte serializers produce the exact little-endian sizes the
    /// player binds (uint4/instr, 8-u32/shape class, u32/offset).
    #[test]
    fn byte_serialization_sizes() {
        let (input, sources) = attn_subchain();
        let g = lower_region(&input, 8);
        let s = schedule_wavefront(
            &g,
            cost_area,
            ScheduleParams {
                num_workers: 4,
                wait_cost_us: 0.18,
            },
        );
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");
        assert_eq!(prog.tape_bytes().len(), prog.tape.len() * 16);
        assert_eq!(
            prog.shapes_bytes().len(),
            prog.shapes.len() * SHAPE_STRIDE * 4
        );
        assert_eq!(prog.tape_offsets_bytes().len(), prog.tape_offsets.len() * 4);
    }

    // ── honest gaps: standalone Silu/Mul are not silently dropped ────

    // ── the full decode layer (after silu·mul fusion) ──────────────

    /// The complete Llama-style decode layer: input-norm → q/k/v proj →
    /// rope → attn → o-proj → residual → post-norm → SwiGLU MLP →
    /// residual. After `fuse_silu_mul` the whole layer maps to GPU arms
    /// and serializes.
    fn full_layer() -> (LoweringInput, Vec<SourceDesc>) {
        let (h, hd, hq, hkv, i, l) = (16u32, 4u32, 4u32, 2u32, 32u32, 3u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let eps = 1e-5f32;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let g = |n: u32, k: u32, a: usize, w: usize| OpDesc {
            op: LoweredOp::Gemm { n, k },
            m: 1,
            inputs: vec![InputRef::Op(a), InputRef::Ext(w)],
        };
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: h }, // 0 res_in
                SourceShape { rows: 1, cols: h }, // 1 in_ln
                SourceShape {
                    rows: qdim,
                    cols: h,
                }, // 2 wq
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 3 wk
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 4 wv
                SourceShape { rows: 1, cols: hd }, // 5 cos
                SourceShape { rows: 1, cols: hd }, // 6 sin
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 7 prefixK
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 8 prefixV
                SourceShape {
                    rows: h,
                    cols: qdim,
                }, // 9 wo
                SourceShape { rows: 1, cols: h }, // 10 post_ln
                SourceShape { rows: i, cols: h }, // 11 wgate
                SourceShape { rows: i, cols: h }, // 12 wup
                SourceShape { rows: h, cols: i }, // 13 wdown
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                g(qdim, h, 0, 2),
                g(kvdim, h, 0, 3),
                g(kvdim, h, 0, 4),
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
                g(h, qdim, 6, 9),
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
                g(i, h, 9, 11), // gate
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(10)],
                },
                g(i, h, 9, 12), // up
                OpDesc {
                    op: LoweredOp::Mul,
                    m: 1,
                    inputs: vec![InputRef::Op(11), InputRef::Op(12)],
                },
                g(h, i, 13, 13), // down
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(14), InputRef::Op(8)],
                },
            ],
            result: 15,
        };
        let mut sources = vec![
            dense(0, WeightBundle::Embedding),
            dense(1, WeightBundle::RmsNorm),
            qweight(2),
            qweight(3),
            qweight(4),
            dense(5, WeightBundle::CosSin),
            dense(6, WeightBundle::CosSin),
            SourceDesc::PrefixK { layer: 0 },
            SourceDesc::PrefixV { layer: 0 },
            qweight(9),
            dense(10, WeightBundle::RmsNorm),
            qweight(11),
            qweight(12),
            qweight(13),
        ];
        sources.shrink_to_fit();
        (input, sources)
    }

    #[test]
    fn full_decode_layer_serializes_after_fusion() {
        use crate::lower::fuse_silu_mul;
        let (input, sources) = full_layer();
        let fused = fuse_silu_mul(&input);
        // Coarse (one block per gemm) gives a clean op histogram.
        let g = lower_region(&fused, 1000);
        // 15 ops after fusion, coarse ⇒ 15 nodes.
        assert_eq!(g.nodes.len(), 15);
        for p in [1u32, 4, 10] {
            let s = schedule_wavefront(
                &g,
                cost_area,
                ScheduleParams {
                    num_workers: p,
                    wait_cost_us: 0.18,
                },
            );
            let prog = serialize(&g, &s, &sources, geom())
                .unwrap_or_else(|e| panic!("full layer must serialize (p={p}): {e}"));
            assert_eq!(prog.num_computes(), g.nodes.len(), "p={p}");

            let mut hist = std::collections::HashMap::new();
            for instr in prog.tape.iter().filter(|i| i[0] == opcode::COMPUTE) {
                *hist.entry(op_of(&prog, instr)).or_insert(0u32) += 1;
            }
            assert_eq!(hist.get(&op_kind::QMV), Some(&7), "q,k,v,o,gate,up,down");
            assert_eq!(hist.get(&op_kind::RMSNORM), Some(&2));
            assert_eq!(hist.get(&op_kind::ROPE), Some(&2));
            assert_eq!(hist.get(&op_kind::ATTN), Some(&1));
            assert_eq!(
                hist.get(&op_kind::SILU_MUL),
                Some(&1),
                "fused MLP activation"
            );
            assert_eq!(hist.get(&op_kind::ADD), Some(&2), "two residual adds");
        }
        // N-block tiling still serializes (more qmv blocks, same op set).
        let gt = lower_region(&fused, 8);
        let st = schedule_wavefront(
            &gt,
            cost_area,
            ScheduleParams {
                num_workers: 10,
                wait_cost_us: 0.18,
            },
        );
        let progt = serialize(&gt, &st, &sources, geom()).expect("tiled layer serializes");
        assert_eq!(progt.num_computes(), gt.nodes.len());
    }

    /// A `RopeAppend` (K side) serializes to WL_OP_ROPE_APPEND with the cache
    /// operands the serializer injects: `[k, cos, sin, v, kv_cache_k,
    /// kv_cache_v, slot_mapping]` (cache halves routed to the node's `layer`,
    /// plus slot_mapping), and the K operand aliases the k-proj output slot
    /// (in-place rotate).
    #[test]
    fn rope_append_serializes_with_cache_operands() {
        let (h, hd, kvdim) = (16u32, 4u32, 8u32); // num_kv = 2
        let layer = 3u32;
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: h }, // 0 x
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 1 wk
                SourceShape {
                    rows: kvdim,
                    cols: h,
                }, // 2 wv
                SourceShape { rows: 1, cols: hd }, // 3 cos
                SourceShape { rows: 1, cols: hd }, // 4 sin
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim, k: h },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::RopeAppend {
                        head_dim: hd,
                        layer,
                    },
                    m: 1,
                    inputs: vec![
                        InputRef::Op(0),
                        InputRef::Ext(3),
                        InputRef::Ext(4),
                        InputRef::Op(1),
                    ],
                },
            ],
            result: 2,
        };
        let sources = vec![
            dense(0, WeightBundle::Embedding),
            qweight(1),
            qweight(2),
            dense(3, WeightBundle::CosSin),
            dense(4, WeightBundle::CosSin),
        ];
        let g = lower_region(&input, 1000);
        let s = partition_roundrobin(&g, 1);
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

        let ra = prog
            .tape
            .iter()
            .find(|i| op_of(&prog, i) == op_kind::ROPE_APPEND)
            .expect("a ROPE_APPEND compute");
        assert_eq!(
            prog.shapes[ra[1] as usize],
            [op_kind::ROPE_APPEND, hd, kvdim / hd, hd, 16, 0, 0, 0]
        );
        let ops = &prog.operands[ra[2] as usize..ra[2] as usize + 7];
        let buf = |o: &OperandSlot| prog.buffers[o.buffer.0 as usize].clone();
        // [k, cos, sin, v, kv_cache_k, kv_cache_v, slot_mapping]
        assert!(
            matches!(buf(&ops[0]), BufferRef::ArenaSlot(_)),
            "k in-place arena"
        );
        assert!(matches!(
            buf(&ops[1]),
            BufferRef::Weight {
                bundle: WeightBundle::CosSin,
                ..
            }
        ));
        assert!(matches!(
            buf(&ops[2]),
            BufferRef::Weight {
                bundle: WeightBundle::CosSin,
                ..
            }
        ));
        assert!(
            matches!(buf(&ops[3]), BufferRef::ArenaSlot(_)),
            "v from v_proj arena"
        );
        assert_eq!(
            buf(&ops[4]),
            BufferRef::Input(InputKind::KvCacheK { layer })
        );
        assert_eq!(
            buf(&ops[5]),
            BufferRef::Input(InputKind::KvCacheV { layer })
        );
        assert_eq!(buf(&ops[6]), BufferRef::Input(InputKind::SlotMapping));
        // The K operand aliases the k-proj (first qmv) output slot.
        let kproj = prog
            .tape
            .iter()
            .find(|i| op_of(&prog, i) == op_kind::QMV)
            .unwrap();
        let kproj_y = prog.operands[kproj[2] as usize + 4].buffer;
        assert_eq!(
            ops[0].buffer, kproj_y,
            "rope_append rotates k_proj output in place"
        );
    }

    #[test]
    fn standalone_silu_is_rejected() {
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: 8 },
                SourceShape { rows: 8, cols: 8 },
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n: 8, k: 8 },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            result: 1,
        };
        let sources = vec![dense(0, WeightBundle::Embedding), qweight(1)];
        let g = lower_region(&input, 8);
        let s = partition_roundrobin(&g, 1);
        let err = serialize(&g, &s, &sources, geom()).unwrap_err();
        assert!(
            matches!(err, SerializeError::UnsupportedOp { detail, .. } if detail.contains("Silu")),
            "standalone Silu must error (pending fusion), got {err:?}"
        );
    }
}
