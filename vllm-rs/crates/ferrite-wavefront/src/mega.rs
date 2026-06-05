// SPDX-License-Identifier: Apache-2.0
//! The **megakernel serializer** — turn a scheduled tensor-region graph
//! into the flat, backend-neutral buffers the on-GPU trivial tape player
//! (`shaders/wavefront_layer.metal`'s `wavefront_player`) consumes.
//!
//! # What this is
//!
//! [`crate::region_schedule`] bin-packs a [`crate::subtile_ir::SubtileIR`]
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
//! the per-dispatch [`crate::metal_tape`] path, then `operands[i] =
//! gpuAddress(buffers[slot.buffer]) + base + slot.byte_offset`.
//!
//! # Backend neutrality (locked)
//!
//! The encoding is plain serializable data — no Metal / objc2 types. The
//! MSL `wavefront_player` and a future CUDA `.cu` interpreter are parallel
//! consumers of the *same* tape/shape/operand encoding; only `BufId →
//! pointer` resolution and the flag primitive are per-target. This module
//! reuses [`crate::metal_tape`]'s neutral [`BufferRef`] / [`WeightLoc`] /
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
//! 1. **Dense weight tensor → quantized triple.** `subtile_ir` models a GEMM
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

use std::collections::{HashMap, HashSet};

use crate::region_schedule::{Schedule, TapeInstr};
use crate::subtile_ir::{
    EwKind, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId, TensorRegion, predecessors,
};
use crate::metal_tape::{
    BufId, BufferRef, InputKind, WeightBundle, affine_scale_row_bytes, packed_weight_row_bytes,
};

// ── Encoding constants (mirror wavefront_layer.metal) ────────────────

/// Tape `opcode` field (`tape[pc].x`).
pub mod opcode {
    pub const COMPUTE: u32 = 0;
    pub const SIGNAL: u32 = 1;
    pub const WAIT: u32 = 2;
    /// Intra-worker `threadgroup_barrier` emitted by the serializer ONLY at a
    /// real read-after-write boundary (a Compute whose region-overlap
    /// predecessor ran on the same worker since the last barrier, plus the
    /// ACQUIRE→Compute and Compute→PUBLISH plumbing RAWs). Replaces the player's
    /// old blind per-op barrier — the player is a dumb executor; the compiler,
    /// which holds the dependence edges, places the sync.
    pub const BARRIER: u32 = 3;
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
    /// A qmv whose output crosses workers: it writes its result DIRECTLY into
    /// the coherent (atomic u32-packed) handoff buffer, simdgroup-locally and
    /// sentinel-safe (PAT-4). Replaces the `QMV` + separate `PUBLISH` + the
    /// compute→publish barrier — the store itself is the readiness signal a
    /// consumer `ACQUIRE`-spins on. Operands `[w, scales, biases, x, y_coh]`.
    pub const QMV_COH: u32 = 9;
    /// The split-K all-reduce: sum every partial (one per activation K-chunk)
    /// into the output. Operands `[out, p0, p1, …]`; shape `(SUM_REDUCE, n,
    /// num_partials, …)`. Cross-worker partials are read from the worker's
    /// ACQUIREd private copy (`read_operand` redirect), so the reduce composes
    /// the elementwise-add atom over whatever copies it sees locally.
    pub const SUM_REDUCE: u32 = 10;
    /// Tall-skinny qmv (small contiguous K == head_dim, large N): the variant
    /// the shape-keyed selector picks for a split-K partial whose K-window is
    /// below `qmv_fast`'s 512-K minimum. Same `[w, scales, biases, x, y]`
    /// operands + `(_, k, n, row_vec)` shape as `QMV`; composes `qmv_quad_impl`.
    pub const QMV_QUAD: u32 = 11;
}

/// `u32`s per shape-class record (`WL_SHAPE_STRIDE`). Wide enough for
/// attention's six dims; simpler ops use the leading slots.
pub const SHAPE_STRIDE: usize = 8;

/// Simdgroups in a worker's fixed 1024-thread threadgroup. The tranche packer
/// distributes a tranche's mutually-independent ops across these (the player's
/// `simd_gid` runs `[0, NUM_SIMDGROUPS)`).
pub const NUM_SIMDGROUPS: u32 = 32;

/// Pack a Compute instruction's `[sg_start, sg_count)` simdgroup range into the
/// tape flag field. `sg_count == 0` (an unpacked / solo op) the player reads as
/// the whole TG, so the default `flag = 0` keeps the old whole-TG behaviour.
pub fn encode_sg_range(sg_start: u32, sg_count: u32) -> u32 {
    (sg_start & 0xFF) | ((sg_count & 0xFF) << 8)
}

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
    /// **Dynamic-dispatch path** — same compute ops as `tape` but laid out in
    /// ASAP-level order (all level-L subtiles before any level-(L+1)). A
    /// dynamic-dispatch player kernel grabs subtiles from this tape via an
    /// atomic claim and processes them with whatever TG is free, removing the
    /// per-worker fixed-chain ceiling. Empty when the schedule doesn't need
    /// this path (the conventional per-worker `tape` is the default emit).
    pub level_tape: Vec<[u32; 4]>,
    /// `[num_levels + 1]` boundaries into `level_tape`. Level L's subtiles
    /// occupy `[level_starts[L], level_starts[L + 1])`. Empty iff
    /// `level_tape` is empty.
    pub level_starts: Vec<u32>,
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

    /// Number of *region-node* computes — must equal the region node count.
    /// Excludes the cross-worker handoff `PUBLISH`/`ACQUIRE` computes (those
    /// are plumbing the scheduler's worker assignment forces, not graph nodes).
    pub fn num_computes(&self) -> usize {
        self.tape
            .iter()
            .filter(|i| {
                i[0] == opcode::COMPUTE && {
                    let op = self.shapes[i[1] as usize][0];
                    op != op_kind::PUBLISH && op != op_kind::ACQUIRE
                }
            })
            .count()
    }

    /// Number of cross-worker handoff computes (`PUBLISH` + `ACQUIRE`). Zero
    /// at `P = 1` (no cross-worker edges).
    pub fn num_handoff_ops(&self) -> usize {
        self.tape
            .iter()
            .filter(|i| {
                i[0] == opcode::COMPUTE && {
                    let op = self.shapes[i[1] as usize][0];
                    op == op_kind::PUBLISH || op == op_kind::ACQUIRE
                }
            })
            .count()
    }

    /// Dump each worker's tape GROUPED INTO TRANCHES, so the dump shows the
    /// parallelism structure the tape encodes. A tranche is a maximal run of
    /// consecutive `Compute` instructions (nothing — no `BARRIER`/`Wait`/
    /// `Signal` — between them); its ops run CONCURRENTLY on their `@start-end`
    /// simdgroup ranges, and a `BARRIER` ends it. Each worker prints
    /// `[(op,op,…), Wn, (op,…), Sn, …]`: a `(…)` group is one tranche (its ops
    /// are parallel), `Wn`/`Sn` are the cross-worker wait/signal on flag `n`.
    /// Ops: `qmv(K,N)` (+`@s-e` when packed onto a simdgroup sub-range) / `rms`
    /// / `rope` / `ropeA` / `silu` / `add` / `attn` / `pub` / `acq`. So
    /// `[(rms), (qmv(2048,256)@0-16, qmv(512,2048)@16-32)]` = tranche 0 is rms
    /// alone (whole TG), tranche 1 runs two qmvs in parallel on 16 simdgroups
    /// each. First `max_per_worker` instructions per worker.
    pub fn dump_tape(&self, max_per_worker: usize) -> String {
        let opname = |op: u32| match op {
            op_kind::QMV => "qmv",
            op_kind::PUBLISH => "pub",
            op_kind::ACQUIRE => "acq",
            op_kind::RMSNORM => "rms",
            op_kind::SILU_MUL => "silu",
            op_kind::ROPE => "rope",
            op_kind::ATTN => "attn",
            op_kind::ADD => "add",
            op_kind::ROPE_APPEND => "ropeA",
            op_kind::SUM_REDUCE => "sumr",
            op_kind::QMV_COH => "qmvC",
            op_kind::QMV_QUAD => "qmvQ",
            _ => "op?",
        };
        let mut out = String::new();
        for w in 0..self.tape_offsets.len().saturating_sub(1) {
            let lo = self.tape_offsets[w] as usize;
            let hi = self.tape_offsets[w + 1] as usize;
            let shown = hi.min(lo + max_per_worker);
            let mut groups: Vec<String> = Vec::new();
            let mut tranche: Vec<String> = Vec::new(); // current run of parallel computes
            for i in lo..shown {
                let ins = self.tape[i];
                if ins[0] == opcode::COMPUTE {
                    let sh = &self.shapes[ins[1] as usize];
                    let mut t = opname(sh[0]).to_string();
                    if sh[0] == op_kind::QMV {
                        t += &format!("({},{})", sh[1], sh[2]);
                    }
                    let sgc = (ins[3] >> 8) & 0xFF;
                    if sgc != 0 {
                        let sgs = ins[3] & 0xFF;
                        t += &format!("@{}-{}", sgs, sgs + sgc);
                    }
                    tranche.push(t);
                } else {
                    // Any sync/barrier ends the current tranche.
                    if !tranche.is_empty() {
                        groups.push(format!("({})", tranche.join(",")));
                        tranche.clear();
                    }
                    match ins[0] {
                        opcode::WAIT => groups.push(format!("W{}", ins[3])),
                        opcode::SIGNAL => groups.push(format!("S{}", ins[3])),
                        opcode::BARRIER => {} // boundary only — already split above
                        _ => {}
                    }
                }
            }
            if !tranche.is_empty() {
                groups.push(format!("({})", tranche.join(",")));
            }
            out += &format!("  w{w}: [{}]", groups.join(", "));
            if shown < hi {
                out += &format!(" …(+{} more)", hi - shown);
            }
            out.push('\n');
        }
        out
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

/// What a leaf source tensor of the [`SubtileIR`] binds to on the GPU —
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

/// How a worker's tape is laid out — the point-2 study's two emission orders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitMode {
    /// DEFAULT. Each producer's publish/signal is emitted immediately after its
    /// compute, so cross-worker consumers unblock ASAP and DAG levels overlap
    /// across workers (the low-latency ~9 ms schedule). No multi-op runs, so the
    /// simdgroup packer is a no-op.
    Pipelined,
    /// OPT-IN study. Group a worker's same-(ASAP-)level ops into a tranche and
    /// batch their publishes/signals, so the simdgroup packer can run them
    /// concurrently on disjoint simdgroup ranges. MEASURED net-neutral at M=1
    /// (the deferred signals turn each level into a BSP superstep whose sync
    /// cost cancels the packing gain). Kept as the reproducible disproof.
    Tranche,
    /// DIAGNOSTIC ONLY (WRONG RESULT). Emit each worker's computes back-to-back
    /// with NO cross-worker sync (no Wait/Signal), NO handoff (no Acquire/
    /// Publish), and NO barriers — every read hits the shared arena slot (racy/
    /// stale). The token stream is garbage; the point is the TIMING: it isolates
    /// the raw per-worker compute throughput from all sync/idle, so the gap to
    /// the real (pipelined) mega is exactly what cross-worker sync + idle costs.
    ComputeOnly,
}

/// Flatten a wavefront [`Schedule`] over a [`SubtileIR`] into a
/// [`MegaProgram`] with the default ([`EmitMode::Pipelined`]) layout. `sources`
/// is parallel to `graph.tensors[0..num_sources]`; `geom` supplies the element
/// width and the attention paged-cache geometry the abstract graph omits.
pub fn serialize(
    graph: &SubtileIR,
    schedule: &Schedule,
    sources: &[SourceDesc],
    geom: Geometry,
) -> Result<MegaProgram, SerializeError> {
    serialize_mode(graph, schedule, sources, geom, EmitMode::Pipelined)
}

/// As [`serialize`], choosing the worker-tape [`EmitMode`].
pub fn serialize_mode(
    graph: &SubtileIR,
    schedule: &Schedule,
    sources: &[SourceDesc],
    geom: Geometry,
    mode: EmitMode,
) -> Result<MegaProgram, SerializeError> {
    let mut ser = Ser::new(graph, sources, geom);
    ser.assign_arena_slots()?;
    // Cross-worker atomic handoff (plan CRITICAL FINDING / A2): a peer
    // worker's plain arena writes are not coherent on the GPU, so a tensor
    // produced on one worker and read on another rides a PUBLISH (per
    // producer block) → ACQUIRE (per consuming worker) atomic round-trip.
    // Region-overlap predecessors (the RAW edges). They drive the handoff plan
    // (a cross-worker operand-read edge needs a coherent staging slot), split a
    // worker's computes into tranches (a same-worker producer→consumer edge is a
    // tranche boundary), and tell us which tranche boundaries need an intra-
    // worker barrier. Cross-worker preds are gated by the schedule's Wait/Signal.
    let preds = predecessors(graph);
    let worker_of = reconstruct_worker_of(graph, schedule);
    ser.plan_handoffs(&worker_of, &preds)?;

    // ASAP level (longest path from a root) of every node. Nodes at the same
    // level are mutually independent AND have all inputs at strictly lower
    // levels — so a worker's level-L computes are a tranche the simdgroup packer
    // can run concurrently, and (since a level-L tranche only WAITs on `< L`
    // producers and SIGNALs `> L` consumers) batching its signals at the tranche
    // END stays deadlock-free. (Grouping by same-worker RAW instead would let a
    // tranche mix levels — waiting on a high level while a peer waits on this
    // tranche's deferred low-level signal — a cycle.) preds have smaller ids, so
    // one ascending-id pass suffices.
    let mut level = vec![0u32; graph.nodes.len()];
    for node in &graph.nodes {
        let id = node.id.0 as usize;
        level[id] = preds[id]
            .iter()
            .map(|p| level[p.0 as usize] + 1)
            .max()
            .unwrap_or(0);
    }
    // PERF DIAG (one-shot): per-DAG-level subtile count = how much subtile-
    // parallelism the schedule actually exposes. If max count per level is
    // small (~P), the persistent-megakernel design with P TGs already covers
    // it. If hundreds, a dynamic-dispatch kernel could fill the GPU much
    // more fully. FERRITE_WAVEFRONT_LEVEL_PROFILE=1 to enable.
    if std::env::var_os("FERRITE_WAVEFRONT_LEVEL_PROFILE").is_some() {
        let max_lvl = *level.iter().max().unwrap_or(&0);
        let mut per_level = vec![0u32; (max_lvl + 1) as usize];
        for &l in &level {
            per_level[l as usize] += 1;
        }
        let total_subtiles = level.len() as u32;
        let mut count_max = 0u32;
        let mut count_p2plus = 0u32; // levels with > 2 subtiles
        let mut count_p10plus = 0u32;
        let mut count_p32plus = 0u32;
        let mut count_p128plus = 0u32;
        for &c in &per_level {
            if c > count_max {
                count_max = c;
            }
            if c >= 2 {
                count_p2plus += c;
            }
            if c >= 10 {
                count_p10plus += c;
            }
            if c >= 32 {
                count_p32plus += c;
            }
            if c >= 128 {
                count_p128plus += c;
            }
        }
        eprintln!(
            "[wf-levels] num_levels={} total_subtiles={} max_per_level={} \
             subtiles_in_levels_with_>=2={} (={:.1}%) \
             subtiles_in_levels_with_>=10={} (={:.1}%) \
             subtiles_in_levels_with_>=32={} (={:.1}%) \
             subtiles_in_levels_with_>=128={} (={:.1}%)",
            per_level.len(),
            total_subtiles,
            count_max,
            count_p2plus,
            100.0 * count_p2plus as f64 / total_subtiles as f64,
            count_p10plus,
            100.0 * count_p10plus as f64 / total_subtiles as f64,
            count_p32plus,
            100.0 * count_p32plus as f64 / total_subtiles as f64,
            count_p128plus,
            100.0 * count_p128plus as f64 / total_subtiles as f64,
        );
    }
    // Each producer's Signal flag, read off the schedule (a `Compute(id)` trailed
    // by `Signal(f)`), so a consumer tranche can wait on its cross-worker inputs'
    // flags regardless of how the schedule ordered the original per-node waits.
    let mut producer_flag = vec![u32::MAX; graph.nodes.len()];
    for worker in &schedule.workers {
        let mut last: Option<SubtileId> = None;
        for instr in &worker.tape {
            match *instr {
                TapeInstr::Compute(id) => last = Some(id),
                TapeInstr::Signal(f) => {
                    if let Some(id) = last {
                        producer_flag[id.0 as usize] = f;
                    }
                }
                TapeInstr::Wait(_) => {}
            }
        }
    }

    let mut tape: Vec<[u32; 4]> = Vec::new();
    let mut tape_offsets: Vec<u32> = vec![0];
    for (wi, worker) in schedule.workers.iter().enumerate() {
        ser.acquired.clear(); // private acquired-copies are per worker
        match mode {
            EmitMode::Tranche => emit_worker_tranches(
                &mut ser,
                wi as u32,
                worker,
                &preds,
                &level,
                &producer_flag,
                &mut tape,
            )?,
            EmitMode::ComputeOnly => {
                // No sync, no handoff, no barriers — racy/wrong, timing only.
                for instr in &worker.tape {
                    if let TapeInstr::Compute(id) = *instr {
                        let node = &graph.nodes[id.0 as usize];
                        let (sc, base) = ser.emit_compute(node)?;
                        tape.push([opcode::COMPUTE, sc, base, 0]);
                    }
                }
            }
            EmitMode::Pipelined => {
                emit_worker_pipelined(&mut ser, wi as u32, worker, &preds, &mut tape)?
            }
        }
        tape_offsets.push(tape.len() as u32);
    }

    // Point-2 simdgroup packing only applies in the tranche path (which emits a
    // tranche's independent ops as a consecutive run for the packer to spread
    // across the 32 simdgroups). The pipelined default emits a producer's
    // publish/signal between computes, so there are no multi-op runs to pack;
    // skipping keeps the default tape byte-identical to the committed mega.
    // MEASURED (M5, 1B-4bit, clean): packing is BW-neutral — tranche+pack 9.3 ms
    // vs pipelined 9.0 ms (both ~78 GB/s) vs per-op 6.3 ms. The M=1 gap is
    // 1-TG-per-core occupancy, which intra-TG packing cannot change. See memory.
    if mode == EmitMode::Tranche {
        pack_simdgroup_ranges(&mut tape, &tape_offsets, &ser.shapes);
    }

    // Dynamic-dispatch path: emit a parallel `level_tape` of just the COMPUTE
    // ops in ASAP-level order (no PUBLISH/ACQUIRE/BARRIER — the level barrier
    // the new player kernel issues replaces those). `level_starts[L]` is the
    // first level-L COMPUTE entry's index in `level_tape`. A worker subtile
    // gets emitted twice — once in the per-worker `tape`, once here — so the
    // `operands` table grows; `shapes` is interned, no duplication there. The
    // existing player kernel ignores these fields; only the new dynamic-
    // dispatch player reads them.
    ser.acquired.clear();
    let mut sorted_node_ids: Vec<u32> = (0..graph.nodes.len() as u32).collect();
    sorted_node_ids.sort_by_key(|&id| (level[id as usize], id));
    let max_level = level.iter().copied().max().unwrap_or(0) as usize;
    let mut level_tape: Vec<[u32; 4]> = Vec::new();
    let mut level_starts: Vec<u32> = vec![0; max_level + 2];
    let mut current_level = 0u32;
    for nid in &sorted_node_ids {
        let node = &graph.nodes[*nid as usize];
        let node_level = level[*nid as usize];
        while current_level < node_level {
            current_level += 1;
            level_starts[current_level as usize] = level_tape.len() as u32;
        }
        let (sc, base) = ser.emit_compute(node)?;
        level_tape.push([opcode::COMPUTE, sc, base, 0]);
    }
    // Final boundary (one past the last level).
    while (current_level as usize + 1) < level_starts.len() {
        current_level += 1;
        level_starts[current_level as usize] = level_tape.len() as u32;
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
        level_tape,
        level_starts,
    })
}

/// Emit one worker's tape in the DEFAULT pipelined per-node order: each compute
/// is preceded by its `ACQUIRE`s + a barrier (iff it reads an acquired private
/// copy or a same-worker producer since the last sync), and a producer's
/// `PUBLISH` (fenced) + `Signal` are emitted IMMEDIATELY after its compute.
/// Signalling immediately lets cross-worker consumers unblock as soon as their
/// specific producer is done, so adjacent DAG levels overlap across workers —
/// the low-latency schedule. (The simdgroup packer rarely fires here: producers
/// hand off, so their publish barrier splits consecutive computes — that is the
/// cost the tranche path trades against.) `worker.tape` is `[Wait* Compute
/// Signal?]` in ascending id; emitted verbatim.
fn emit_worker_pipelined(
    ser: &mut Ser,
    wi: u32,
    worker: &crate::region_schedule::Worker,
    preds: &[Vec<SubtileId>],
    tape: &mut Vec<[u32; 4]>,
) -> Result<(), SerializeError> {
    let graph = ser.graph;
    // Nodes computed since this worker's last barrier/Signal/Wait (each carries
    // a device fence); cleared whenever one is emitted.
    let mut since: HashSet<SubtileId> = HashSet::new();
    for instr in &worker.tape {
        match *instr {
            TapeInstr::Compute(id) => {
                let node = &graph.nodes[id.0 as usize];
                let n_acq = ser.emit_acquires(node, wi, tape)?;
                let raw = preds[id.0 as usize].iter().any(|p| since.contains(p));
                if n_acq > 0 || raw {
                    tape.push([opcode::BARRIER, 0, 0, 0]);
                    since.clear();
                }
                let (sc, base) = ser.emit_compute(node)?;
                tape.push([opcode::COMPUTE, sc, base, 0]);
                since.insert(id);
                // An N-block qmv handoff producer wrote the coherent slot
                // DIRECTLY (QMV_COH) — no PUBLISH, no compute→publish barrier.
                // A non-qmv handoff producer (rope/silu/add) AND a split-K
                // matmul partial (which writes arena, not coherent) still need
                // the barrier + PUBLISH from arena.
                if ser.needs_publish(node) {
                    tape.push([opcode::BARRIER, 0, 0, 0]);
                    since.clear();
                    ser.emit_publish(node, tape)?;
                }
            }
            // PAT-4 (data-IS-the-flag): the cross-worker order is carried by the
            // consumer's spin-ACQUIRE on the producer's coherent store, so the
            // schedule's Signal/Wait flags are redundant — drop them (and the
            // device barrier each one carried in the player). These were the
            // dominant sync cost (3192 waits + signals, each a TG barrier).
            TapeInstr::Signal(_) | TapeInstr::Wait(_) => {}
        }
    }
    Ok(())
}

/// Emit one worker's tape, grouped into **tranches by ASAP level** so the
/// simdgroup packer can run a tranche's independent ops concurrently. The
/// worker's computes are bucketed by `level` and emitted in ascending level;
/// each level-bucket is one tranche of mutually-independent ops (all their
/// inputs are at strictly lower levels, already produced). Per tranche, in
/// order: its cross-worker `Wait`s (worker-deduped, on its inputs' producer
/// flags) → `ACQUIRE`s → one `BARRIER` (iff it acquired or reads an earlier
/// tranche's same-worker output) → all its `Compute`s (the packer splits the 32
/// simdgroups among them) → one `BARRIER` + the `PUBLISH`es of any cross-worker
/// outputs → the producers' `Signal`s. Batching publishes/signals AFTER all the
/// computes (vs `compute,barrier,publish,signal` per node) keeps the tranche's
/// computes consecutive for the packer; deferring the signals is safe precisely
/// because a level-L tranche signals only `> L` consumers and waits only `< L`
/// producers — no wait cycle.
#[allow(clippy::too_many_arguments)]
fn emit_worker_tranches(
    ser: &mut Ser,
    wi: u32,
    worker: &crate::region_schedule::Worker,
    preds: &[Vec<SubtileId>],
    level: &[u32],
    producer_flag: &[u32],
    tape: &mut Vec<[u32; 4]>,
) -> Result<(), SerializeError> {
    // The graph ref is also held by `ser`; bind it separately so a node borrow
    // (`&graph.nodes[..]`) can coexist with `&mut ser` (which only mutates ser's
    // other fields). Both are shared refs to the same graph.
    let graph = ser.graph;

    // This worker's computes (the schedule emits them in ascending id).
    let mut ids: Vec<SubtileId> = worker
        .tape
        .iter()
        .filter_map(|i| match i {
            TapeInstr::Compute(id) => Some(*id),
            _ => None,
        })
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    // A pred in `on_worker` is same-worker (tranche-boundary barrier); one not in
    // it is cross-worker (gated by a Wait on its producer flag).
    let on_worker: HashSet<SubtileId> = ids.iter().copied().collect();
    // Order by (level, id): a valid topological order that buckets same-level
    // (independent, concurrently-runnable) computes together.
    ids.sort_by_key(|id| (level[id.0 as usize], id.0));

    let mut waited: HashSet<u32> = HashSet::new(); // flags this worker has waited
    let mut i = 0usize;
    while i < ids.len() {
        let lvl = level[ids[i].0 as usize];
        let mut j = i;
        while j < ids.len() && level[ids[j].0 as usize] == lvl {
            j += 1;
        }
        let tranche = &ids[i..j];

        // 1. Wait on each cross-worker input's producer flag (deduped per worker).
        for &id in tranche {
            for p in &preds[id.0 as usize] {
                if !on_worker.contains(p) {
                    let f = producer_flag[p.0 as usize];
                    debug_assert_ne!(f, u32::MAX, "cross-worker producer must have a flag");
                    if waited.insert(f) {
                        tape.push([opcode::WAIT, 0, 0, f]);
                    }
                }
            }
        }
        // 2. ACQUIRE each cross-worker input (emit_acquires dedups per worker).
        let mut n_acq = 0usize;
        for &id in tranche {
            n_acq += ser.emit_acquires(&graph.nodes[id.0 as usize], wi, tape)?;
        }
        // 3. Barrier before the computes iff an ACQUIRE wrote a private copy, or
        //    a compute reads a same-worker producer from an earlier tranche.
        let cross_tranche_raw = tranche.iter().any(|&id| {
            preds[id.0 as usize]
                .iter()
                .any(|p| on_worker.contains(p) && !tranche.contains(p))
        });
        if n_acq > 0 || cross_tranche_raw {
            tape.push([opcode::BARRIER, 0, 0, 0]);
        }
        // 4. The tranche's computes (flag 0; pack_simdgroup_ranges fills it).
        for &id in tranche {
            let (sc, base) = ser.emit_compute(&graph.nodes[id.0 as usize])?;
            tape.push([opcode::COMPUTE, sc, base, 0]);
        }
        // 5. PUBLISH every cross-worker output, fenced from the computes by one
        //    barrier (the computes wrote disjoint stripes; one barrier covers all).
        let publishes: Vec<SubtileId> = tranche
            .iter()
            .copied()
            .filter(|&id| ser.needs_publish(&graph.nodes[id.0 as usize]))
            .collect();
        if !publishes.is_empty() {
            tape.push([opcode::BARRIER, 0, 0, 0]);
            for id in &publishes {
                ser.emit_publish(&graph.nodes[id.0 as usize], tape)?;
            }
        }
        // 6. Signals for this tranche's producers (after the publishes they fence).
        for &id in tranche {
            let f = producer_flag[id.0 as usize];
            if f != u32::MAX {
                tape.push([opcode::SIGNAL, 0, 0, f]);
            }
        }
        i = j;
    }
    Ok(())
}

// ── Tranche simdgroup packer (point-2) ───────────────────────────────

/// Whether the player's arm for `op` is **simdgroup-local** — it uses no
/// threadgroup scratch and no TG-wide `threadgroup_barrier`, so it runs
/// correctly on any disjoint sub-range of the 32 simdgroups (and several such
/// ops can therefore run concurrently). The threadgroup-wide arms (RMSNORM /
/// ATTN reduce across all 1024 threads; PUBLISH / ACQUIRE pack across them;
/// ROPE_APPEND has an internal device barrier) must keep the whole TG, so a run
/// containing any of them is left solo.
fn is_packable(op: u32) -> bool {
    matches!(
        op,
        op_kind::QMV | op_kind::QMV_QUAD | op_kind::SILU_MUL | op_kind::ROPE | op_kind::ADD
    )
}

/// The relative cost of one packable op, used to size its simdgroup share. A
/// qmv is bandwidth-bound on its weight read (`N × K` bytes); the elementwise
/// arms scale with their element count `N`.
fn op_cost(shape: &[u32; SHAPE_STRIDE]) -> f64 {
    match shape[0] {
        // qmv (fast or quad): bandwidth-bound on the K-slice × N weight read.
        op_kind::QMV | op_kind::QMV_QUAD => shape[1] as f64 * shape[2] as f64, // K × N
        _ => shape[1] as f64,                                                  // N
    }
}

/// Assign each op in a tranche a `[sg_start, sg_count)` simdgroup range over the
/// worker's [`NUM_SIMDGROUPS`] simdgroups, sized ∝ cost so the tranche's
/// makespan (`max over ops of work/simdgroups`) is minimised. `sg_count` is
/// always **even** (qmv pairs two simdgroups per 8-row group). Returns one
/// `(start, count)` per input op, in input order.
///
/// Ranges partition the 32 simdgroups into `min(k, 16)` contiguous **lanes**
/// (each lane ≥ 1 pair). With `k ≤ 16` (the decode reality — e.g. ~6 gate/up
/// blocks per worker) each op gets its own cost-weighted lane and they all run
/// concurrently. With `k > 16` (very fine N-blocking) ops share lanes by greedy
/// longest-processing-time balance and the lane's ops run sequentially while
/// the 16 lanes run concurrently — still correct, just less parallel.
fn pack_tranche(costs: &[f64]) -> Vec<(u32, u32)> {
    let k = costs.len();
    debug_assert!(k > 0);
    let total_pairs = NUM_SIMDGROUPS / 2; // 16 pairs (qmv needs simdgroup pairs)
    let n_lanes = k.min(total_pairs as usize);
    let cmp = |a: f64, b: f64| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);

    // Greedy LPT: place the costliest ops first, each onto the least-loaded lane.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| cmp(costs[b], costs[a]));
    let mut lane_of = vec![0usize; k];
    let mut lane_load = vec![0f64; n_lanes];
    for &oi in &order {
        let l = (0..n_lanes)
            .min_by(|&a, &b| cmp(lane_load[a], lane_load[b]))
            .unwrap();
        lane_of[oi] = l;
        lane_load[l] += costs[oi];
    }

    // Distribute the 16 simdgroup-pairs across lanes ∝ load (each lane ≥ 1 pair),
    // greedily handing the next pair to the lane with the worst load/pair ratio.
    let mut pairs = vec![1u32; n_lanes];
    let mut remaining = total_pairs - n_lanes as u32;
    while remaining > 0 {
        let l = (0..n_lanes)
            .max_by(|&a, &b| {
                cmp(
                    lane_load[a] / pairs[a] as f64,
                    lane_load[b] / pairs[b] as f64,
                )
            })
            .unwrap();
        pairs[l] += 1;
        remaining -= 1;
    }

    // Contiguous simdgroup range per lane; each op takes its lane's range.
    let mut lane_start = vec![0u32; n_lanes];
    let mut acc = 0u32;
    for l in 0..n_lanes {
        lane_start[l] = acc;
        acc += pairs[l] * 2;
    }
    debug_assert_eq!(acc, NUM_SIMDGROUPS, "lanes must cover all simdgroups");
    (0..k)
        .map(|oi| {
            let l = lane_of[oi];
            (lane_start[l], pairs[l] * 2)
        })
        .collect()
}

/// Stamp per-op simdgroup ranges into the Compute flag fields. For each worker
/// it finds maximal runs of consecutive Compute instructions — which the
/// emission guarantees are mutually independent (a BARRIER / Signal / Wait sits
/// at every dependence edge) — and, when every op in the run is simdgroup-local
/// and the run has more than one op, packs them with [`pack_tranche`]. Single-op
/// runs and runs containing a threadgroup-wide op are left at `flag = 0` (the
/// player then gives them the whole TG, i.e. the prior sequential behaviour).
fn pack_simdgroup_ranges(
    tape: &mut [[u32; 4]],
    tape_offsets: &[u32],
    shapes: &[[u32; SHAPE_STRIDE]],
) {
    // PERF DIAG (droppable): FERRITE_WAVEFRONT_PACK=0 leaves every Compute at the
    // whole TG (flag 0) so the tranche restructure can be measured WITHOUT the
    // simdgroup packing — isolating the two levers.
    if std::env::var("FERRITE_WAVEFRONT_PACK").as_deref() == Ok("0") {
        return;
    }
    let op_of = |instr: &[u32; 4]| shapes[instr[1] as usize][0];
    for w in 0..tape_offsets.len().saturating_sub(1) {
        let lo = tape_offsets[w] as usize;
        let hi = tape_offsets[w + 1] as usize;
        let mut i = lo;
        while i < hi {
            if tape[i][0] != opcode::COMPUTE {
                i += 1;
                continue;
            }
            let start = i;
            while i < hi && tape[i][0] == opcode::COMPUTE {
                i += 1;
            }
            let run = start..i;
            if run.len() < 2 || !run.clone().all(|j| is_packable(op_of(&tape[j]))) {
                continue; // solo op, or a threadgroup-wide run → leave flag = 0
            }
            let costs: Vec<f64> = run
                .clone()
                .map(|j| op_cost(&shapes[tape[j][1] as usize]))
                .collect();
            for (k, j) in run.zip(pack_tranche(&costs)) {
                let (sg_start, sg_count) = j;
                tape[k][3] = encode_sg_range(sg_start, sg_count);
            }
        }
    }
}

/// Recover `worker_of[node_id]` from the schedule's per-worker tapes: a
/// `Compute(id)` in worker `w`'s tape means node `id` runs on worker `w`.
/// The handoff planner needs the assignment the scheduler chose; the
/// [`Schedule`] only exposes the per-worker instruction streams, so rebuild it.
fn reconstruct_worker_of(graph: &SubtileIR, schedule: &Schedule) -> Vec<u32> {
    let mut worker_of = vec![0u32; graph.nodes.len()];
    for (wi, worker) in schedule.workers.iter().enumerate() {
        for instr in &worker.tape {
            if let TapeInstr::Compute(id) = instr {
                worker_of[id.0 as usize] = wi as u32;
            }
        }
    }
    worker_of
}

/// The cross-worker handoff plan for one op-output tensor: a tensor whose
/// blocks are produced on one worker and read on another. The GPU player
/// cannot read a peer worker's plain arena writes coherently (plan CRITICAL
/// FINDING / A2), so every producer block PUBLISHes its stripe (atomic
/// u32-pack) into a coherent staging buffer, and each consuming worker
/// ACQUIREs the whole tensor (unpack → a private copy) before reading it.
#[derive(Clone, Debug)]
struct Handoff {
    /// Coherent (atomic u32-packed) staging arena slot, holds the whole tensor.
    coherent_slot: u32,
    /// Workers that write a block of this tensor (each publishes its stripe).
    producers: HashSet<u32>,
    /// Tensor width in elements (decode row 0); `n_pairs = cols / 2`.
    cols: u32,
}

/// Mutable serializer state: the interned buffer / shape tables, the
/// operand list, the tensor→arena-slot map, and the cross-worker handoff plan.
struct Ser<'a> {
    graph: &'a SubtileIR,
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
    /// Cross-worker handoff plan, keyed by the shared op-output tensor.
    handoff: HashMap<TensorId, Handoff>,
    /// `(tensor, worker)` → its private acquired-copy arena slot (lazy).
    private_slot: HashMap<(TensorId, u32), u32>,
    /// Tensors the *current* worker has already acquired (reset per worker);
    /// a redirected read targets the private copy, not the shared arena slot.
    acquired: HashMap<TensorId, u32>,
}

impl<'a> Ser<'a> {
    fn new(graph: &'a SubtileIR, sources: &'a [SourceDesc], geom: Geometry) -> Self {
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
            handoff: HashMap::new(),
            private_slot: HashMap::new(),
            acquired: HashMap::new(),
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
    /// [`BufId`] (mirrors `MetalTapeBuilder::buffer`).
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
            // A cross-worker tensor this worker has ACQUIREd is read from its
            // private (coherent) copy, not the shared arena slot a peer wrote.
            let buffer = match self.acquired.get(&tr.tensor) {
                Some(&slot) => self.intern(BufferRef::ArenaSlot(slot), self.geom.act_elem),
                None => self.arena_bufid(tr.tensor),
            };
            Ok(OperandSlot {
                buffer,
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

    /// Plan the cross-worker handoffs: a tensor needs a coherent staging slot
    /// iff a region-overlap EDGE crosses workers — a consumer reads it AS A GPU
    /// OPERAND on a different worker than a producer block wrote it. Using the
    /// edges (not per-tensor producer/consumer worker-SETS) is the partition's
    /// linchpin: a column-preserving chain (q-rope→attn, gate/up→silu→down-part)
    /// reads the SAME columns its same-worker producer wrote, so NO edge crosses
    /// ⇒ NO handoff (the data stays in the worker's arena — no broadcast wait).
    /// Only the genuine joins cross: the split-K partials a SumReduce all-reduces
    /// (and, for `lower_region`, a whole-op consumer reading an N-blocked
    /// producer). Identical to the worker-set test where consumers read whole;
    /// strictly tighter where they read aligned slices. Attn reads its new K/V
    /// from the paged cache, NOT the arena rope/v output, so those edges are not
    /// operand reads here (the cache write ordering is a separate concern).
    fn plan_handoffs(
        &mut self,
        worker_of: &[u32],
        preds: &[Vec<SubtileId>],
    ) -> Result<(), SerializeError> {
        // Workers that write each op-output tensor (for the per-worker PUBLISH /
        // the consumer's ACQUIRE-from-peer decision).
        let mut producers: HashMap<TensorId, HashSet<u32>> = HashMap::new();
        for node in &self.graph.nodes {
            producers
                .entry(node.output.tensor)
                .or_default()
                .insert(worker_of[node.id.0 as usize]);
        }
        // Crossing tensors = those on a cross-worker operand-read edge.
        let mut crossing: HashSet<TensorId> = HashSet::new();
        for (cid, ps) in preds.iter().enumerate() {
            let consumer = &self.graph.nodes[cid];
            let cw = worker_of[cid];
            for p in ps {
                if worker_of[p.0 as usize] == cw {
                    continue; // same-worker edge — local arena, no handoff
                }
                let t = self.graph.nodes[p.0 as usize].output.tensor;
                if self.reads_as_operand(consumer, t) {
                    crossing.insert(t);
                }
            }
        }
        let mut tids: Vec<TensorId> = crossing.into_iter().collect();
        tids.sort_by_key(|t| t.0); // deterministic slot order
        for t in tids {
            let cols = self.graph.shape(t).cols;
            if !cols.is_multiple_of(2) {
                return Err(SerializeError::NonDecodeShape {
                    id: t.0,
                    detail: "cross-worker handoff tensor has odd width (can't pair-pack)",
                });
            }
            let coherent_slot = self.arena_bytes.len() as u32;
            self.arena_bytes.push((cols as u64 / 2) * 4); // one u32 per bf16 pair
            self.handoff.insert(
                t,
                Handoff {
                    coherent_slot,
                    producers: producers[&t].clone(),
                    cols,
                },
            );
        }
        Ok(())
    }

    /// Whether `node` reads tensor `t` as a GPU OPERAND (vs a dataflow-only
    /// edge). All ops read every input as an operand EXCEPT attention, whose
    /// new K/V (and the prefix K/V sources) are the paged cache, not arena
    /// reads — only its q (input 0) is an arena operand.
    fn reads_as_operand(&self, node: &SubtileNode, t: TensorId) -> bool {
        match node.op {
            SubOp::AttnDecode { .. } => node.inputs.first().map(|i| i.tensor) == Some(t),
            _ => node.inputs.iter().any(|i| i.tensor == t),
        }
    }

    /// Before a consumer `Compute`, emit an `ACQUIRE` for each cross-worker
    /// input tensor it reads whose data was produced (partly) on another
    /// worker — once per worker. Records the tensor→private-slot redirect so
    /// `emit_compute`'s reads target the coherent private copy, not the shared
    /// arena slot a peer wrote. Operands `[private_dst, coherent]`; the worker
    /// has already `Wait`ed on every cross-worker producer's flag.
    fn emit_acquires(
        &mut self,
        node: &SubtileNode,
        wi: u32,
        tape: &mut Vec<[u32; 4]>,
    ) -> Result<usize, SerializeError> {
        // Decide first (immutable view), then mutate — avoids aliasing self.
        let mut to_acquire: Vec<TensorId> = Vec::new();
        for inp in &node.inputs {
            let t = inp.tensor;
            if self.is_source(t) || self.acquired.contains_key(&t) || to_acquire.contains(&t) {
                continue;
            }
            if let Some(h) = self.handoff.get(&t)
                && h.producers.iter().any(|&p| p != wi)
            {
                to_acquire.push(t);
            }
        }
        let n_acquired = to_acquire.len();
        for t in to_acquire {
            let (coherent_slot, cols) = {
                let h = &self.handoff[&t];
                (h.coherent_slot, h.cols)
            };
            // This worker's private whole-tensor copy (lazy, one per worker).
            let priv_slot = match self.private_slot.get(&(t, wi)) {
                Some(&s) => s,
                None => {
                    let s = self.arena_bytes.len() as u32;
                    self.arena_bytes
                        .push(cols as u64 * self.geom.act_elem as u64);
                    self.private_slot.insert((t, wi), s);
                    s
                }
            };
            let dst = self.intern(BufferRef::ArenaSlot(priv_slot), self.geom.act_elem);
            let coh = self.intern(BufferRef::ArenaSlot(coherent_slot), 4);
            let base = self.push_operands(&[
                OperandSlot {
                    buffer: dst,
                    byte_offset: 0,
                },
                OperandSlot {
                    buffer: coh,
                    byte_offset: 0,
                },
            ]);
            let sc = self.intern_shape([op_kind::ACQUIRE, cols / 2, 0, 0, 0, 0, 0, 0]);
            tape.push([opcode::COMPUTE, sc, base, 0]);
            self.acquired.insert(t, priv_slot);
        }
        Ok(n_acquired)
    }

    /// Whether this node's output tensor crosses workers — i.e. `emit_publish`
    /// will emit a PUBLISH for it (so the caller knows to fence the producer
    /// Compute → PUBLISH read).
    fn produces_handoff(&self, node: &SubtileNode) -> bool {
        self.handoff.contains_key(&node.output.tensor)
    }

    /// After a producer block `Compute` whose output tensor is a handoff,
    /// emit a `PUBLISH` of that block's stripe (atomic u32-pack) into the
    /// coherent slot — before the `Signal` the schedule emits next. Operands
    /// `[coherent + stripe, arena + stripe]`; shape `(PUBLISH, pair0=0,
    /// n_pairs)` (the stripe offsets ride in the pointers).
    fn emit_publish(
        &mut self,
        node: &SubtileNode,
        tape: &mut Vec<[u32; 4]>,
    ) -> Result<(), SerializeError> {
        let t = node.output.tensor;
        let coherent_slot = match self.handoff.get(&t) {
            Some(h) => h.coherent_slot,
            None => return Ok(()),
        };
        self.check_row0(&node.output, node.id.0)?;
        let cols = node.output.region.cols;
        if !cols.start.is_multiple_of(2) || !cols.len.is_multiple_of(2) {
            return Err(SerializeError::NonDecodeShape {
                id: node.id.0,
                detail: "handoff producer block has odd col offset/width (can't pair-pack)",
            });
        }
        let arena = self.arena_bufid(t);
        let coh = self.intern(BufferRef::ArenaSlot(coherent_slot), 4);
        let base = self.push_operands(&[
            OperandSlot {
                buffer: coh,
                byte_offset: (cols.start as u64 / 2) * 4,
            },
            OperandSlot {
                buffer: arena,
                byte_offset: cols.start as u64 * self.geom.act_elem as u64,
            },
        ]);
        let sc = self.intern_shape([op_kind::PUBLISH, 0, cols.len / 2, 0, 0, 0, 0, 0]);
        tape.push([opcode::COMPUTE, sc, base, 0]);
        Ok(())
    }

    /// Emit one compute node → `(shape_class, operand_base)`.
    fn emit_compute(&mut self, node: &SubtileNode) -> Result<(u32, u32), SerializeError> {
        match node.op {
            SubOp::MatmulTile => self.emit_qmv(node),
            SubOp::RmsNorm { eps } => self.emit_rmsnorm(node, eps),
            SubOp::RopeRotate { head_dim, .. } => self.emit_rope(node, head_dim),
            SubOp::RopeAppend {
                head_dim, layer, ..
            } => self.emit_rope_append(node, head_dim, layer),
            SubOp::SiluMul => self.emit_silu_mul(node),
            SubOp::Elementwise(EwKind::Add) => self.emit_add(node),
            SubOp::AttnDecode {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                ..
            } => self.emit_attn(node, num_q_heads, num_kv_heads, head_dim, scale),
            SubOp::Elementwise(EwKind::Silu) => Err(SerializeError::UnsupportedOp {
                id: node.id.0,
                detail: "standalone Silu — fuse Silu+Mul into SiluMul before scheduling",
            }),
            SubOp::Elementwise(EwKind::Mul) => Err(SerializeError::UnsupportedOp {
                id: node.id.0,
                detail: "standalone Mul — fuse Silu+Mul into SiluMul before scheduling",
            }),
            SubOp::SumReduce => self.emit_sum_reduce(node),
        }
    }

    /// SUM_REDUCE: operands `[out, p0, p1, …]`; shape `(SUM_REDUCE, n,
    /// num_partials, …)`. The split-K all-reduce — each replicated copy sums
    /// every partial (one per activation K-chunk) into its own whole output.
    /// A cross-worker partial is read from this worker's ACQUIREd private copy
    /// (the `read_operand` redirect), so the arm just adds the operands it is
    /// handed; `eval_node`'s `SumReduce` is the bit-exact reference.
    fn emit_sum_reduce(&mut self, node: &SubtileNode) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let n = node.output.region.cols.len;
        let mut slots = Vec::with_capacity(node.inputs.len() + 1);
        slots.push(self.write_operand(&node.output, id)?);
        for inp in &node.inputs {
            slots.push(self.read_operand(inp, id)?);
        }
        let num_partials = node.inputs.len() as u32;
        let base = self.push_operands(&slots);
        let sc = self.intern_shape([op_kind::SUM_REDUCE, n, num_partials, 0, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// QMV: operands `[w, scales, biases, x, y]`; shape `(QMV, k, n,
    /// row_vec)`. Two region contracts, one primitive:
    ///   - **N-block**: weight rows `blk`, full K. The output col start `r`
    ///     is the weight's row block, so w/scales/biases/y carry `r * row_stride`
    ///     — the linchpin that makes a block a standalone `n×K` matvec.
    ///   - **split-K** (a partial of o_proj/down): weight rows full, K-slice
    ///     `kb`. The weight is read as a K-WINDOW: `row_vec = K_full` is the
    ///     full row stride (shape slot 3), `k = kb.len` the window length, and
    ///     w/scales/biases carry the within-row K-offset `kb.start` (group-
    ///     aligned). A split-K partial writes to ARENA + is PUBLISHed (the
    ///     coherent fuse only applies to single-reduction N-block outputs).
    fn emit_qmv(&mut self, node: &SubtileNode) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let act = &node.inputs[0];
        let wtr = &node.inputs[1];
        let out = &node.output;
        self.check_row0(act, id)?;
        self.check_row0(out, id)?;

        let k = act.region.cols.len; // K-window length (== weight K-slice width)
        let n = out.region.cols.len;
        let r = out.region.cols.start as u64; // N-block row offset (0 for split-K)

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

        // K-window: the weight row is `k_full` wide (the row stride); this op
        // reduces the K-slice `[k_off, k_off + k)`. Off-window (N-block qmv)
        // `k_full == k`, `k_off == 0` ⇒ `row_vec` slot stays 0, byte-identical
        // to the previous emit. A split-K slice must be group-aligned (the qmv
        // atom indexes scales per group). We read K_full from the ACTIVATION
        // tensor (always `[m, K]` by convention) rather than the weight tensor,
        // because the weight's stored layout can be transposed (`[K, N]` for
        // some quant formats) — using the activation gives the true K.
        let k_full = self.graph.shape(act.tensor).cols;
        let k_off = act.region.cols.start;
        let windowed = k_full != k;
        if windowed && (!k_off.is_multiple_of(group_size) || !k.is_multiple_of(group_size)) {
            return Err(SerializeError::NonDecodeShape {
                id,
                detail: "split-K weight K-slice is not group-aligned",
            });
        }
        let row_vec = if windowed { k_full } else { 0 };

        // Weight/scale offsets = N-block row offset (full-K stride) + within-row
        // K-window offset. Both terms vanish in the respective other mode.
        let w_off =
            r * packed_weight_row_bytes(k_full, bits) + packed_weight_row_bytes(k_off, bits);
        let sb_off = r * affine_scale_row_bytes(k_full, group_size, scale_elem as u64)
            + affine_scale_row_bytes(k_off, group_size, scale_elem as u64);
        let y_off = r * self.geom.act_elem as u64;

        let w_buf = self.intern(weight, 4); // packed 4-bit weight read as u32
        let s_buf = self.intern(scales, scale_elem);
        let b_buf = self.intern(biases, scale_elem);
        let x = self.read_operand(act, id)?;
        // Shape-keyed primitive variant (the cost seam, [[project_pd_wavefront_cost_driven]]):
        // K below qmv_fast's 512-value-per-pass minimum ⇒ the tall-skinny
        // qmv_quad (the head_dim-wide split-K partials); else qmv_fast. Seeded
        // with MLX's by-shape heuristic; a cost sweep can refine it later.
        let variant = if k.is_multiple_of(512) {
            op_kind::QMV
        } else {
            op_kind::QMV_QUAD
        };
        // PAT-4 producer fuse: an N-block (non-windowed) qmv_fast handoff writes
        // STRAIGHT into the handoff's coherent slot — no arena, no PUBLISH, no
        // compute→publish barrier (each simdgroup stores its own rows; the store
        // IS the readiness signal; `r` even ⇒ pair offset `r/2` exact). A split-K
        // partial CANNOT fuse (its output is a partial the SumReduce all-reduces)
        // and qmv_quad has no coherent variant ⇒ it writes arena + is PUBLISHed.
        let handoff_slot = self.handoff.get(&out.tensor).map(|h| h.coherent_slot);
        let use_coh = handoff_slot.is_some() && !windowed && variant == op_kind::QMV;
        let (y, op) = if use_coh {
            let coh = self.intern(BufferRef::ArenaSlot(handoff_slot.unwrap()), 4);
            (
                OperandSlot {
                    buffer: coh,
                    byte_offset: (r / 2) * 4,
                },
                op_kind::QMV_COH,
            )
        } else {
            (
                OperandSlot {
                    buffer: self.arena_bufid(out.tensor),
                    byte_offset: y_off,
                },
                variant,
            )
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
        let sc = self.intern_shape([op, k, n, row_vec, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// Whether a `MatmulTile` is a split-K partial — its weight reads a K-window
    /// narrower than the full weight row. Such a qmv carries `row_vec`, writes
    /// to arena, and is PUBLISHed (vs the coherent fuse for N-block outputs).
    fn qmv_is_split_k(&self, node: &SubtileNode) -> bool {
        matches!(node.op, SubOp::MatmulTile)
            && node.inputs.len() >= 2
            && self.graph.shape(node.inputs[1].tensor).cols != node.inputs[1].region.cols.len
    }

    /// Whether this node's cross-worker output must be PUBLISHed from arena.
    /// An N-block matmul handoff wrote the coherent slot directly (QMV_COH) →
    /// NO publish; everything else with a cross-worker output (rope/silu/add,
    /// and split-K matmul partials, which write arena) publishes from arena.
    fn needs_publish(&self, node: &SubtileNode) -> bool {
        // Publish iff it crosses workers AND it is not an N-block matmul (the
        // only kind that fused into the coherent slot via QMV_COH). A split-K
        // matmul partial and any non-matmul op write arena ⇒ they publish.
        let fused_into_coherent =
            matches!(node.op, SubOp::MatmulTile) && !self.qmv_is_split_k(node);
        self.produces_handoff(node) && !fused_into_coherent
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

    /// ROPE: operands `[x, cos_sin, positions]`; shape `(ROPE, head_dim,
    /// num_heads, rot_dim)`. In place — operand 0 is the (aliased) producer
    /// buffer. The rotary row a rope needs is `cos_sin[positions[t]]`, a
    /// **runtime** quantity (the position changes every decode step), so the
    /// operand is the WHOLE `cos_sin` table plus the `positions` input and the
    /// arm indexes `cos_sin + positions[0]*rot_dim` exactly like the oracle —
    /// never a compile-time-baked position offset. (The region IR's input 2,
    /// the host-eval `sin` slice, is not a GPU operand: `sin` is the same table
    /// row at `+ rot_dim/2`, derived in the arm.)
    fn emit_rope(
        &mut self,
        node: &SubtileNode,
        head_dim: u32,
    ) -> Result<(u32, u32), SerializeError> {
        let id = node.id.0;
        let x = self.write_operand(&node.output, id)?;
        let cos_sin = self.rotary_table_operand(&node.inputs[1])?;
        let positions = self.positions_operand();
        let cols = node.output.region.cols.len;
        if head_dim == 0 || !cols.is_multiple_of(head_dim) {
            return Err(SerializeError::NonDecodeShape {
                id,
                detail: "rope output cols not a multiple of head_dim",
            });
        }
        let num_heads = cols / head_dim;
        let base = self.push_operands(&[x, cos_sin, positions]);
        // rot_dim == head_dim (full rope), matching RopeAppend / the oracle.
        let sc = self.intern_shape([op_kind::ROPE, head_dim, num_heads, head_dim, 0, 0, 0, 0]);
        Ok((sc, base))
    }

    /// The rotary `cos_sin` TABLE operand for a rope op: input `tr` must be a
    /// `CosSin`-bundle weight leaf. Returns the table base (`byte_offset 0`);
    /// the arm applies the live position. This is the compile-time guard that
    /// closes the gap which let cos/sin be mis-bound as a baked position-0
    /// slice — a rope's rotary input MUST be the whole `CosSin` table (position
    /// applied at runtime), never a `Dense` gain or a pre-sliced row.
    fn rotary_table_operand(&mut self, tr: &TensorRegion) -> Result<OperandSlot, SerializeError> {
        if !self.is_source(tr.tensor) {
            return Err(SerializeError::BadSource {
                tensor: tr.tensor.0,
                detail: "rope rotary input must be a CosSin leaf source",
            });
        }
        match self.source(tr.tensor)?.clone() {
            SourceDesc::Dense {
                buffer:
                    buffer @ BufferRef::Weight {
                        bundle: WeightBundle::CosSin,
                        ..
                    },
                elem,
            } => Ok(OperandSlot {
                buffer: self.intern(buffer, elem),
                byte_offset: 0, // table base; the arm indexes + positions[0]*rot_dim
            }),
            _ => Err(SerializeError::BadSource {
                tensor: tr.tensor.0,
                detail: "rope rotary input is not a CosSin table",
            }),
        }
    }

    /// The `positions` runtime input operand (the live decode positions buffer);
    /// the rope arms read `positions[0]` (decode is `m == 1`).
    fn positions_operand(&mut self) -> OperandSlot {
        OperandSlot {
            buffer: self.intern(BufferRef::Input(InputKind::Positions), 4),
            byte_offset: 0,
        }
    }

    /// ROPE_APPEND: operands `[k, cos_sin, positions, v, kv_cache_k,
    /// kv_cache_v, slot_mapping]`; shape `(ROPE_APPEND, head_dim, num_kv,
    /// rot_dim, block_size, ...)`. Rotate K in place (operand 0 is the aliased
    /// producer buffer), then write rotated K + un-rotated V to the paged cache.
    /// `cos_sin` is the WHOLE rotary table + the `positions` runtime input (the
    /// arm applies the live position, like the oracle — never a baked offset);
    /// the cache halves + slot_mapping are runtime inputs the serializer injects
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
        let cos_sin = self.rotary_table_operand(&node.inputs[1])?;
        let positions = self.positions_operand();
        let v = self.read_operand(&node.inputs[3], id)?;
        let cols = node.output.region.cols.len;
        if head_dim == 0 || !cols.is_multiple_of(head_dim) {
            return Err(SerializeError::NonDecodeShape {
                id,
                detail: "rope_append output cols not a multiple of head_dim",
            });
        }
        // HEAD-RANGE (mirrors emit_attn): this block ropes + cache-writes a
        // contiguous kv-head slice of the K tensor. The cache is indexed by the
        // GLOBAL kv-head (`kvh_start + local`) with the GLOBAL kv-head count as
        // its stride; the K/V operands are block-based (local heads). Whole-op
        // rope_append (lower_region) is the special case start=0, block=global.
        let num_kv_block = cols / head_dim; // heads this block writes (loop bound)
        let num_kv_global = self.graph.shape(node.output.tensor).cols / head_dim;
        let kvh_start = node.output.region.cols.start / head_dim;
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
        let base = self.push_operands(&[k, cos_sin, positions, v, kv_k, kv_v, slot]);
        // shape (ROPE_APPEND, head_dim, num_kv_block, rot_dim, block_size,
        // num_kv_global, kvh_start). rot_dim == head_dim (full rope). The arm
        // writes cache head `kvh_start + h` with `num_kv_global` as the cache
        // stride; `num_kv_global == 0` ⇒ whole-op fallback (== num_kv_block).
        let sc = self.intern_shape([
            op_kind::ROPE_APPEND,
            head_dim,
            num_kv_block,
            head_dim,
            self.geom.block_size,
            num_kv_global,
            kvh_start,
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
    /// block_size, max_blocks, head_range)`. The region node is `[q, prefixK,
    /// prefixV, newK, newV]`; the GPU reads the runtime paged cache, so prefixK/V
    /// only name the cache layer and the new-token edges are dataflow-only.
    /// `num_q`/`num_kv` are the GLOBAL head counts (the GQA ratio + cache
    /// stride); `head_range = (qh_start << 16) | qh_count` is THIS block's
    /// q-head slice, derived from the output region — head-tiled (partitioned)
    /// attn computes one head block, reading the matching kv-head of the whole
    /// cache. q/output operands are block-based (their byte offset is the head
    /// slice), so the arm loops local heads and maps `qh_start + local` → kv.
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
        // This block's q-head slice (head-tiled in the partition; whole op in
        // lower_region). The output column slice IS the q-head range.
        if head_dim == 0 || !node.output.region.cols.len.is_multiple_of(head_dim) {
            return Err(SerializeError::NonDecodeShape {
                id,
                detail: "attn output cols not a multiple of head_dim",
            });
        }
        let qh_start = node.output.region.cols.start / head_dim;
        let qh_count = node.output.region.cols.len / head_dim;
        let head_range = (qh_start << 16) | qh_count;
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
            head_range,
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
    use crate::metal_tape::{WeightBundle, WeightLoc, WeightRole};
    use crate::region_schedule::{ScheduleParams, partition_roundrobin, schedule_wavefront};
    use crate::subtile_ir::{SourceShape, SubtileNode, lower_region};

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

    /// Any qmv VARIANT — the shape-keyed selector emits `QMV` (fast, K%512==0),
    /// `QMV_QUAD` (tall-skinny small-K, the toy fixtures + split-K), or `QMV_COH`
    /// (the N-block coherent fuse). Structural tests care that it's a qmv, not
    /// which variant the K shape picked.
    fn is_qmv(op: u32) -> bool {
        matches!(op, op_kind::QMV | op_kind::QMV_QUAD | op_kind::QMV_COH)
    }

    // ── QMV linchpin: per-block byte offsets + shape dedup ──────────

    /// A single wide qmv N-block-tiled: each block's w/scales/biases/y
    /// operands carry `r * row_stride` and x stays whole — the exact
    /// offset math `metal_tape::tile_qmv` proved, now through the region
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
                op: LoweredOp::Gemm { n },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            result: 0,
        };
        let sources = vec![dense(0, WeightBundle::Embedding), qweight(1)];
        let g = lower_region(&input, std::num::NonZeroU32::new(nb).unwrap());
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

    // ── cross-worker atomic handoff (PUBLISH/ACQUIRE) ───────────────

    /// A qmv→qmv chain whose first qmv N-block-splits across workers and whose
    /// second qmv reads the whole first output: at P>1 the serializer must
    /// route the cross-worker activation through the atomic handoff — a
    /// PUBLISH per producer block + an ACQUIRE per consuming worker, into a
    /// coherent staging slot, never a peer's plain arena write (plan CRITICAL
    /// FINDING). P=1 has no cross-worker edge, so emits none.
    #[test]
    fn cross_worker_qmv_chain_emits_publish_acquire() {
        let (k0, n0, n1) = (512u32, 128u32, 128u32); // k1 == n0
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: k0 },
                SourceShape { rows: n0, cols: k0 },
                SourceShape { rows: n1, cols: n0 },
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n: n0 },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: n1 },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
            ],
            result: 1,
        };
        let sources = vec![dense(0, WeightBundle::Embedding), qweight(1), qweight(2)];
        let g = lower_region(&input, std::num::NonZeroU32::new(64).unwrap()); // n0=128 → 2 blocks of 64

        // P=1: a single worker reads its own arena writes → no handoff.
        let p1 = serialize(&g, &partition_roundrobin(&g, 1), &sources, geom()).expect("p1");
        assert_eq!(p1.num_handoff_ops(), 0, "P=1: no cross-worker handoff");
        assert_eq!(p1.num_computes(), g.nodes.len());

        // P=2: the 2 producer blocks land on distinct workers (round-robin),
        // and every consumer block reads the whole first output → cross-worker.
        let p2 = serialize(&g, &partition_roundrobin(&g, 2), &sources, geom()).expect("p2");
        assert_eq!(
            p2.num_computes(),
            g.nodes.len(),
            "every region node computed once (handoff plumbing excluded)"
        );
        assert!(
            p2.num_handoff_ops() > 0,
            "P=2: the cross-worker handoff fires"
        );

        // PAT-4 producer fuse: the qmv producer writes its coherent slot
        // DIRECTLY (QMV_COH) — there is NO PUBLISH for a qmv. Each ACQUIRE reads
        // a coherent slot that some QMV_COH wrote (same staging buffer) — the
        // round-trip is still closed, just without the separate publish.
        let kinds: HashSet<u32> = p2
            .tape
            .iter()
            .filter(|i| i[0] == opcode::COMPUTE)
            .map(|i| p2.shapes[i[1] as usize][0])
            .collect();
        assert!(
            kinds.contains(&op_kind::QMV_COH),
            "a QMV_COH compute (qmv producer writes coherent directly)"
        );
        assert!(
            !kinds.contains(&op_kind::PUBLISH),
            "no PUBLISH for a qmv producer (fused into QMV_COH)"
        );
        assert!(kinds.contains(&op_kind::ACQUIRE), "an ACQUIRE compute");
        let coherent_of = |op: u32, slot_idx: usize| -> HashSet<BufId> {
            p2.tape
                .iter()
                .filter(|i| i[0] == opcode::COMPUTE && p2.shapes[i[1] as usize][0] == op)
                .map(|i| p2.operands[i[2] as usize + slot_idx].buffer)
                .collect()
        };
        let published_into = coherent_of(op_kind::QMV_COH, 4); // QMV_COH operand 4 = y_coh
        let acquired_from = coherent_of(op_kind::ACQUIRE, 1); // ACQUIRE operand 1 = coherent
        assert!(
            !acquired_from.is_empty() && acquired_from.is_subset(&published_into),
            "every acquired staging slot was written by a QMV_COH: pub={published_into:?} acq={acquired_from:?}"
        );
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
                    op: LoweredOp::Gemm { n: h },
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
        let g = lower_region(&input, std::num::NonZeroU32::new(h).unwrap()); // 1 qmv block + 1 rope
        let s = partition_roundrobin(&g, 1);
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

        // tape: [qmv block, BARRIER, rope]. The rope reads the qmv's output, so
        // the compiler places exactly one barrier at that RAW boundary. qmv y =
        // operand[base0+4]; rope x = operand[base1+0].
        let computes: Vec<&[u32; 4]> = prog
            .tape
            .iter()
            .filter(|i| i[0] == opcode::COMPUTE)
            .collect();
        let qmv = computes[0];
        let rope = computes[1];
        assert!(is_qmv(op_of(&prog, qmv)), "first compute is a qmv variant");
        assert_eq!(op_of(&prog, rope), op_kind::ROPE);
        assert_eq!(
            prog.tape.iter().filter(|i| i[0] == opcode::BARRIER).count(),
            1,
            "one barrier at the qmv→rope RAW boundary"
        );
        assert_eq!(prog.tape[1][0], opcode::BARRIER);
        let qmv_y = prog.operands[qmv[2] as usize + 4].buffer;
        let rope_base = rope[2] as usize;
        let rope_x = prog.operands[rope_base].buffer;
        assert_eq!(qmv_y, rope_x, "rope rotates the qmv output buffer in place");
        // operand 1 = the whole CosSin table at offset 0 (the arm indexes by
        // position); operand 2 = the runtime Positions input — never a baked
        // per-position cos/sin slice.
        let cos_sin = &prog.operands[rope_base + 1];
        assert_eq!(cos_sin.byte_offset, 0, "cos_sin operand is the table base");
        assert!(
            matches!(
                prog.buffers[cos_sin.buffer.0 as usize],
                BufferRef::Weight {
                    bundle: WeightBundle::CosSin,
                    ..
                }
            ),
            "rope operand 1 is the CosSin table"
        );
        assert_eq!(
            prog.buffers[prog.operands[rope_base + 2].buffer.0 as usize],
            BufferRef::Input(InputKind::Positions),
            "rope operand 2 is the runtime Positions input"
        );
        // Only one arena slot is allocated (rope aliases, no fresh slot).
        assert_eq!(prog.arena_bytes.len(), 1);
        assert_eq!(
            prog.shapes[rope[1] as usize],
            [op_kind::ROPE, hd, h / hd, hd, 0, 0, 0, 0]
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
                    op: LoweredOp::Gemm { n: qdim },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
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
        let g = lower_region(&input, std::num::NonZeroU32::new(1000).unwrap()); // coarse: 1 block each
        let s = partition_roundrobin(&g, 1);
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

        let attn = prog
            .tape
            .iter()
            .find(|i| op_of(&prog, i) == op_kind::ATTN)
            .expect("an ATTN compute");
        // Whole-op attn: head_range = (qh_start=0 << 16) | qh_count=hq = hq.
        assert_eq!(
            prog.shapes[attn[1] as usize],
            [op_kind::ATTN, hd, hq, hkv, scale.to_bits(), 16, 4, hq]
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
                    op: LoweredOp::Gemm { n: qdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(3)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
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
                    op: LoweredOp::Gemm { n: h },
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
        let g = lower_region(&input, std::num::NonZeroU32::new(nb).unwrap());
        // 1 rms + 2 q + 1 k + 1 v + 1 rope + 1 rope + 1 attn + 2 o + 2 add = 12
        // (the residual add is elementwise ⇒ tiled by nb like the GEMMs; rope/
        // attn stay whole in lower_region — head-tiling lives in lower_partitioned).
        assert_eq!(g.nodes.len(), 12);
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
                    opcode::BARRIER => assert_eq!(*instr, [opcode::BARRIER, 0, 0, 0]),
                    other => panic!("bad opcode {other}"),
                }
            }

            // PAT-4 (data-IS-the-flag): cross-worker order is carried by the
            // consumer's spin-ACQUIRE on the producer's coherent store, so the
            // tape emits NO Signal/Wait flags at all.
            let signals = prog.tape.iter().filter(|i| i[0] == opcode::SIGNAL).count();
            let waits = prog.tape.iter().filter(|i| i[0] == opcode::WAIT).count();
            assert_eq!(signals, 0, "no SIGNAL — data-is-flag (p={p})");
            assert_eq!(waits, 0, "no WAIT — data-is-flag (p={p})");

            // Op histogram: 6 qmv, 1 rms, 2 rope, 1 attn, 2 add. (A cross-worker
            // qmv producer is emitted as QMV_COH — coherent direct output — so
            // count QMV + QMV_COH for the total.)
            let mut hist = std::collections::HashMap::new();
            for instr in prog.tape.iter().filter(|i| i[0] == opcode::COMPUTE) {
                *hist.entry(op_of(&prog, instr)).or_insert(0u32) += 1;
            }
            let qmv_total = hist.get(&op_kind::QMV).copied().unwrap_or(0)
                + hist.get(&op_kind::QMV_COH).copied().unwrap_or(0)
                + hist.get(&op_kind::QMV_QUAD).copied().unwrap_or(0);
            assert_eq!(qmv_total, 6, "p={p}");
            assert_eq!(hist.get(&op_kind::RMSNORM), Some(&1));
            assert_eq!(hist.get(&op_kind::ROPE), Some(&2));
            assert_eq!(hist.get(&op_kind::ATTN), Some(&1));
            assert_eq!(
                hist.get(&op_kind::ADD),
                Some(&2),
                "residual add tiled (h=16,nb=8)"
            );
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
        let g = lower_region(&input, std::num::NonZeroU32::new(8).unwrap());
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
        let g = |n: u32, _k: u32, a: usize, w: usize| OpDesc {
            op: LoweredOp::Gemm { n },
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
        let g = lower_region(&fused, std::num::NonZeroU32::new(1000).unwrap());
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
            // q,k,v,o,gate,up,down — a cross-worker producer is QMV_COH.
            let qmv_total = hist.get(&op_kind::QMV).copied().unwrap_or(0)
                + hist.get(&op_kind::QMV_COH).copied().unwrap_or(0)
                + hist.get(&op_kind::QMV_QUAD).copied().unwrap_or(0);
            assert_eq!(qmv_total, 7, "q,k,v,o,gate,up,down");
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
        let gt = lower_region(&fused, std::num::NonZeroU32::new(8).unwrap());
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
                    op: LoweredOp::Gemm { n: kvdim },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
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
        let g = lower_region(&input, std::num::NonZeroU32::new(1000).unwrap());
        let s = partition_roundrobin(&g, 1);
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");

        let ra = prog
            .tape
            .iter()
            .find(|i| op_of(&prog, i) == op_kind::ROPE_APPEND)
            .expect("a ROPE_APPEND compute");
        // Whole-op rope_append: num_kv_block == num_kv_global == kvdim/hd,
        // kvh_start == 0 (slots 2, 5, 6).
        assert_eq!(
            prog.shapes[ra[1] as usize],
            [
                op_kind::ROPE_APPEND,
                hd,
                kvdim / hd,
                hd,
                16,
                kvdim / hd,
                0,
                0
            ]
        );
        let ops = &prog.operands[ra[2] as usize..ra[2] as usize + 7];
        let buf = |o: &OperandSlot| prog.buffers[o.buffer.0 as usize].clone();
        // [k, cos_sin, positions, v, kv_cache_k, kv_cache_v, slot_mapping]
        assert!(
            matches!(buf(&ops[0]), BufferRef::ArenaSlot(_)),
            "k in-place arena"
        );
        // operand 1 = the whole CosSin table (table base, offset 0); the arm
        // indexes the live position. operand 2 = the runtime Positions input —
        // NOT a second cos/sin slice (that was the position-blind bug).
        assert!(matches!(
            buf(&ops[1]),
            BufferRef::Weight {
                bundle: WeightBundle::CosSin,
                ..
            }
        ));
        assert_eq!(ops[1].byte_offset, 0, "cos_sin operand is the table base");
        assert_eq!(buf(&ops[2]), BufferRef::Input(InputKind::Positions));
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
        let kproj = prog.tape.iter().find(|i| is_qmv(op_of(&prog, i))).unwrap();
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
                    op: LoweredOp::Gemm { n: 8 },
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
        let g = lower_region(&input, std::num::NonZeroU32::new(8).unwrap());
        let s = partition_roundrobin(&g, 1);
        let err = serialize(&g, &s, &sources, geom()).unwrap_err();
        assert!(
            matches!(err, SerializeError::UnsupportedOp { detail, .. } if detail.contains("Silu")),
            "standalone Silu must error (pending fusion), got {err:?}"
        );
    }

    // ── tranche simdgroup packer (point-2) ──────────────────────────

    /// `(sg_start, sg_count)` ranges sorted by start must contiguously
    /// partition all `NUM_SIMDGROUPS`, with even counts ≥ 2.
    fn assert_partition(ranges: &[(u32, u32)]) {
        for &(_, c) in ranges {
            assert!(c >= 2 && c % 2 == 0, "sg_count even ≥ 2, got {c}");
            assert!(c <= NUM_SIMDGROUPS, "sg_count ≤ 32, got {c}");
        }
        let mut sorted: Vec<(u32, u32)> = ranges.to_vec();
        sorted.sort_by_key(|&(s, _)| s);
        sorted.dedup();
        let mut acc = 0u32;
        for &(s, c) in &sorted {
            assert_eq!(s, acc, "lanes contiguous from 0");
            acc += c;
        }
        assert_eq!(acc, NUM_SIMDGROUPS, "lanes cover every simdgroup");
    }

    /// `pack_tranche` always partitions the 32 simdgroups into even lanes, and
    /// (when each op gets its own lane, k ≤ 16) the costliest op gets no fewer
    /// simdgroups than the cheapest.
    #[test]
    fn pack_tranche_invariants() {
        for costs in [
            vec![1.0, 1.0],      // two equal blocks
            vec![4.0, 1.0, 1.0], // q : k : v
            vec![1.0; 6],        // ~6 gate/up blocks, equal
            vec![3.0, 2.0, 2.0, 1.0, 1.0],
            vec![8.0, 1.0], // very skewed
        ] {
            let ranges = pack_tranche(&costs);
            assert_eq!(ranges.len(), costs.len());
            assert_partition(&ranges);
            let cmp = |a: usize, b: usize| costs[a].partial_cmp(&costs[b]).unwrap();
            let imax = (0..costs.len()).max_by(|&a, &b| cmp(a, b)).unwrap();
            let imin = (0..costs.len()).min_by(|&a, &b| cmp(a, b)).unwrap();
            assert!(
                ranges[imax].1 >= ranges[imin].1,
                "costlier op gets ≥ simdgroups: costs={costs:?} ranges={ranges:?}"
            );
        }
    }

    /// More ops than simdgroup-pairs (> 16): ops share lanes but the result is
    /// still a valid even partition (each lane runs its ops sequentially).
    #[test]
    fn pack_tranche_more_ops_than_lanes() {
        let ranges = pack_tranche(&vec![1.0; 40]);
        assert_eq!(ranges.len(), 40);
        assert_partition(&ranges);
        let starts: std::collections::HashSet<u32> = ranges.iter().map(|&(s, _)| s).collect();
        assert!(starts.len() <= 16, "at most 16 lanes, got {}", starts.len());
    }

    /// A wide qmv N-block-tiled into 4 independent blocks on one worker is a
    /// tranche: the 4 consecutive Computes get disjoint simdgroup ranges
    /// covering all 32 — they run concurrently in the player.
    #[test]
    fn serialize_packs_independent_qmv_tranche() {
        let (n, k, nb) = (128u32, 2048u32, 32u32);
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: k },
                SourceShape { rows: n, cols: k },
            ],
            ops: vec![OpDesc {
                op: LoweredOp::Gemm { n },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            result: 0,
        };
        let sources = vec![dense(0, WeightBundle::Embedding), qweight(1)];
        let g = lower_region(&input, std::num::NonZeroU32::new(nb).unwrap());
        let s = partition_roundrobin(&g, 1); // all 4 blocks on worker 0
        // Tranche mode groups the 4 independent blocks into one packed tranche.
        let prog = serialize_mode(&g, &s, &sources, geom(), EmitMode::Tranche).expect("serialize");
        assert_eq!(prog.tape.len(), 4);
        let ranges: Vec<(u32, u32)> = prog
            .tape
            .iter()
            .map(|i| {
                assert_eq!(i[0], opcode::COMPUTE);
                assert_eq!(op_of(&prog, i), op_kind::QMV);
                (i[3] & 0xFF, (i[3] >> 8) & 0xFF)
            })
            .collect();
        assert_partition(&ranges);
    }

    /// A single block is alone in its tranche, so it keeps the whole TG
    /// (`flag == 0`) — no packing, the prior whole-threadgroup behaviour.
    #[test]
    fn serialize_leaves_solo_op_unpacked() {
        let (n, k) = (16u32, 256u32);
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: k },
                SourceShape { rows: n, cols: k },
            ],
            ops: vec![OpDesc {
                op: LoweredOp::Gemm { n },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            result: 0,
        };
        let sources = vec![dense(0, WeightBundle::Embedding), qweight(1)];
        let g = lower_region(&input, std::num::NonZeroU32::new(64).unwrap()); // nb ≥ n ⇒ one block
        let s = partition_roundrobin(&g, 1);
        let prog = serialize(&g, &s, &sources, geom()).expect("serialize");
        assert_eq!(prog.tape.len(), 1);
        assert_eq!(prog.tape[0][3], 0, "solo op keeps the whole TG (flag 0)");
    }
}
