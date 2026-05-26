// SPDX-License-Identifier: Apache-2.0
//! `SubtileIr` — the fully-resolved, compile-time-safe instruction tape
//! the GPU subtile player executes.
//!
//! # The law (see `feedback_subtile_ir_trivial_player`)
//!
//! Every structural decision — which kernel, every N-block byte offset,
//! the dispatch grid, every dependency edge — is resolved by the
//! **compiler** (the lowering that builds this IR) and **recorded** here.
//! The player ([`play`]) is trivial: it walks the tape and, per
//! instruction, either issues the pre-resolved dispatch or honors a sync
//! flag. **The player never chooses, computes, or infers anything.** If
//! code in the player is making a decision, that decision belongs in the
//! lowering instead.
//!
//! # Shape
//!
//! - [`SubtileIr::buffers`] — the logical operands ([`BufferRef`]): a
//!   model weight tensor, an arena activation slot, a scratch buffer, or
//!   a runtime input. Resolved to a concrete GPU buffer + base offset
//!   **once** at player setup, never per-instruction.
//! - [`SubtileIr::pipelines`] — kernel specializations ([`PipelineSpec`]:
//!   library + symbol + function constants). Resolved to a compute
//!   pipeline state **once** at setup.
//! - [`SubtileIr::tape`] — the [`SubtileInstr`] stream. A `Run` carries a
//!   fully-resolved [`Dispatch`]; `Wait`/`Signal` carry a [`FlagId`].
//!   **Dependencies are instructions** — the player does no hazard
//!   analysis.
//!
//! # Compile-time safety
//!
//! Buffer / pipeline / flag handles are newtypes ([`BufId`] / [`PipeId`]
//! / [`FlagId`]). [`Dispatch`]'s per-op constructors (e.g.
//! [`Dispatch::qmv_block`]) take exactly the operands that op requires,
//! so a malformed subtile op (missing scales, wrong arity) is
//! unrepresentable at the construction site. [`validate`] then checks the
//! whole tape's cross-references and flag discipline.
//!
//! # Backend independence
//!
//! This module is host-pure and names no GPU API. It is the backend
//! **ISA**: a per-backend *compiler* lowers a forward into a [`SubtileIr`]
//! (picking kernels, tiling, offsets, grids, sync) and a per-backend
//! [`Executor`] replays it. [`play`] is the one shared, trivial driver.
//! The Metal backend is the first consumer; a CUDA backend would add a
//! `CudaExecutor: Executor` + its own compiler, both targeting this same
//! IR unchanged. The field vocabulary is deliberately neutral — a
//! [`PipelineSpec`] is a (module, entry-point, specialization-constants)
//! triple (Metal library/function/function-constants ≈ CUDA
//! module/kernel/template-or-launch specialization); a [`Grid`] is
//! grid-groups × group-threads (Metal threadgroups ≈ CUDA blocks);
//! [`BufferRef::Weight`] addresses weights through the `WeightAccessors`
//! locator that *both* interpreters already share.

#![allow(dead_code)]

use crate::subtile::{Range, Region};

// ── Handles ─────────────────────────────────────────────────────────

/// Index into [`SubtileIr::buffers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufId(pub u32);

/// Index into [`SubtileIr::pipelines`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PipeId(pub u32);

/// A point-to-point sync flag. `Signal(f)` makes a producer's writes
/// visible; `Wait(f)` blocks a consumer until the matching signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FlagId(pub u32);

// ── Logical operands ────────────────────────────────────────────────

/// Which tensor of a weight bundle a [`BufferRef::Weight`] names. Mirrors
/// the interpreters' shared `WeightTensor` 1:1 (decode subset) so the
/// compiler's `LoweredCommand`→`BufferRef` map is unambiguous both ways.
/// Backend-neutral. `Weight` is the matrix — the resolver picks the
/// packed-quant vs dense tensor from the layer type, not from this tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRole {
    /// The weight matrix / norm gain (`WeightTensor::Weight`).
    Weight,
    /// Per-output dense bias (`WeightTensor::Bias`).
    Bias,
    /// Affine dequant scales (`WeightTensor::AffineScales`).
    AffineScales,
    /// Affine dequant biases (`WeightTensor::AffineBiases`).
    AffineBiases,
    /// Affine quant's per-output linear bias (`WeightTensor::AffineLinearBias`).
    AffineLinearBias,
}

/// Which weight-bundle accessor resolves a [`BufferRef::Weight`]. Mirrors
/// the interpreters' shared `WeightBundleKind` (the subset the decode
/// path needs); the Executor maps it to the concrete `WeightAccessors`
/// call. Backend-neutral — both Metal and CUDA resolve through the same
/// accessor trait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightBundle {
    RmsNorm,
    Embedding,
    LinearLayer,
    /// Rotary cos/sin table bundle (`WeightBundleKind::CosSin`).
    CosSin,
    /// Quantized (affine) token-embedding bundle.
    AffineQuantEmbedding,
}

/// The `WeightAccessors` locator both interpreters use to recover a
/// per-layer weight tensor: the macro-baked `(bucket, op_idx, slot)`
/// triple plus the unrolled `layer`. Plain data; no backend types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightLoc {
    pub layer: u32,
    pub bucket: u32,
    pub op_idx: u32,
    pub slot: u32,
}

/// A runtime per-forward input buffer. Mirrors the interpreters' shared
/// `RuntimeBindingKind` so the compiler maps it 1:1. (Cos/sin are a
/// `WeightBundle::CosSin` weight, not a runtime input.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    InputIds,
    Positions,
    SlotMapping,
    CuSeqlensQ,
    SeqUsedK,
    BlockTable,
    /// Paged KV cache K half for `layer` (resolver picks the offset).
    KvCacheK {
        layer: u32,
    },
    /// Paged KV cache V half for `layer`.
    KvCacheV {
        layer: u32,
    },
    /// `[1]` u32 — the forward's actual `num_tokens`.
    NumTokens,
}

/// What a logical buffer in [`SubtileIr::buffers`] is bound to. The metal
/// `Executor` resolves each of these to a `(MTLBuffer, base_offset)`
/// once, at setup (weights via the per-arch `WeightAccessors`, arena
/// slots from the worker arena, etc.).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferRef {
    /// A model weight tensor, resolved via the shared `WeightAccessors`
    /// trait: `bundle` picks the accessor, `loc` the per-layer tensor,
    /// `role` which tensor of the bundle (packed / scales / biases / …).
    Weight {
        bundle: WeightBundle,
        role: WeightRole,
        loc: WeightLoc,
    },
    /// A colored arena activation slot.
    ArenaSlot(u32),
    /// A shared scratch buffer (e.g. split-K partials).
    Scratch(u32),
    /// A runtime per-forward input.
    Input(InputKind),
}

// ── Dispatch payload ────────────────────────────────────────────────

/// One kernel argument-table binding: bind `buffer` (plus the byte
/// `offset` the compiler resolved — e.g. an N-block's row offset) at
/// kernel argument `index`. The offset is **carried, not computed** by
/// the player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub buffer: BufId,
    pub offset: u64,
    pub index: u32,
}

impl Binding {
    pub fn new(buffer: BufId, offset: u64, index: u32) -> Self {
        Self {
            buffer,
            offset,
            index,
        }
    }
    /// Bind the whole buffer (offset 0) at `index`.
    pub fn whole(buffer: BufId, index: u32) -> Self {
        Self::new(buffer, 0, index)
    }
}

/// A 2-D region of a buffer a dispatch reads or writes — the DATAFLOW the
/// validator checks, distinct from the kernel `Binding`s the player
/// issues. Row-major `Region{rows, cols}` (region.rs's model), so it
/// expresses a column slice (an N-block), a whole tensor (elementwise),
/// or a strided sub-block — what a flat byte interval cannot. The
/// validator derives byte extents from `region` + the buffer's element
/// width when it needs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionRef {
    pub buffer: BufId,
    pub region: Region,
}

impl RegionRef {
    pub fn new(buffer: BufId, region: Region) -> Self {
        Self { buffer, region }
    }
    /// A single `[rows]×[c0, c0+w)` row-major region (the common M-row,
    /// column-slice access — e.g. an N-block of a matvec output).
    pub fn rows_cols(buffer: BufId, rows: u32, c0: u32, w: u32) -> Self {
        Self {
            buffer,
            region: Region {
                rows: Range::new(0, rows),
                cols: Range::new(c0, w),
            },
        }
    }
}

/// A dispatch grid, resolved by the compiler: `tg` grid-groups each of
/// `tpt` threads (Metal threadgroups × threads-per-threadgroup; the same
/// shape as CUDA grid-blocks × threads-per-block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub tg: [u32; 3],
    pub tpt: [u32; 3],
}

impl Grid {
    pub fn new(tg: [u32; 3], tpt: [u32; 3]) -> Self {
        Self { tg, tpt }
    }
}

/// A function-constant value used to specialize a pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

/// One `function_constant(index)` assignment for a pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FnConst {
    pub index: u32,
    pub value: ConstValue,
}

/// A kernel specialization the `Executor` resolves to a pipeline /
/// launchable kernel once at setup: a (module, entry-point,
/// specialization-constants) triple. On Metal that is (library, function,
/// function-constants); a CUDA backend would read it as (module, kernel,
/// template/launch specialization). For an N-block qmv the `OUT_VEC_SIZE
/// = nb` (N) constant is the linchpin that makes a block a standalone
/// `nb × K` matvec with no kernel changes.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineSpec {
    /// Backend module/library that contains `symbol`.
    pub library: &'static str,
    /// Entry-point / kernel symbol name.
    pub symbol: String,
    /// Compile-time specialization constants.
    pub constants: Vec<FnConst>,
}

/// The subtile-operation tag a [`Dispatch`] carries. Purely descriptive
/// (debugging, validation, op histograms) — the player issues the same
/// bind+dispatch sequence regardless of tag, because the [`Dispatch`]
/// payload is already fully resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// One N-block of a quantized matvec (`qmv` / `qmv_fast` / `qmv_quad`).
    QmvBlock,
    RmsNorm,
    FusedAddRmsNorm,
    RopeAppend,
    AttnViaCache,
    SiluMul,
    FusedGateUpSiluMul,
    Add,
    Embed,
    AffineEmbed,
    /// Any other whole op (synth megakernels, gather/scatter, …). The tag
    /// is descriptive only — the player ignores it — so a coarse
    /// catch-all is fine; the precise kernel is in the `PipelineSpec`.
    Other,
}

/// The arena dataflow of a dispatch — what the validator checks (NOT
/// executed; the player only runs `bindings`). One **fixed-arity** variant
/// per op, so a malformed op (a qmv missing its `y` write, an `Add` with
/// the wrong inputs) is a TYPE ERROR at the construction site, not a
/// runtime `validate` miss. There is deliberately **no `Vec`** here: every
/// op's arena read/write count is statically known (even the fused synth
/// kernels — their slots are fixed; only their kernel *bindings* vary,
/// which is why `Dispatch.bindings` stays a list and this does not).
/// Weights / runtime inputs are external (always defined) and are not
/// modeled here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpDataflow {
    /// One N-block: reads the whole activation `x`, writes a column slice
    /// of the output `y`.
    QmvBlock { x: RegionRef, y: RegionRef },
    /// reads `x`, writes `out` (rmsnorm, fused-gate-up-silu-mul, …).
    Map { x: RegionRef, out: RegionRef },
    /// reads `a` + `b`, writes `out` (silu·mul).
    Zip {
        a: RegionRef,
        b: RegionRef,
        out: RegionRef,
    },
    /// writes `out`, reads only external operands (embed from input_ids).
    Produce { out: RegionRef },
    /// reads `x`, writes `out` (attention; the KV cache is external).
    Attn { x: RegionRef, out: RegionRef },
    /// reads `delta` + `residual`, writes `residual` in place (residual add).
    AddInPlace {
        delta: RegionRef,
        residual: RegionRef,
    },
    /// reads `x`, updates `residual` in place, and writes `out` — the
    /// fused (add-norm → … → out) megakernels.
    UpdateProduce {
        x: RegionRef,
        residual: RegionRef,
        out: RegionRef,
    },
}

impl OpDataflow {
    /// The arena regions this op reads (excludes external weights/inputs).
    pub fn reads(&self) -> Vec<RegionRef> {
        match *self {
            OpDataflow::QmvBlock { x, .. } => vec![x],
            OpDataflow::Map { x, .. } => vec![x],
            OpDataflow::Zip { a, b, .. } => vec![a, b],
            OpDataflow::Produce { .. } => vec![],
            OpDataflow::Attn { x, .. } => vec![x],
            OpDataflow::AddInPlace { delta, residual } => vec![delta, residual],
            OpDataflow::UpdateProduce { x, residual, .. } => vec![x, residual],
        }
    }
    /// The arena regions this op writes.
    pub fn writes(&self) -> Vec<RegionRef> {
        match *self {
            OpDataflow::QmvBlock { y, .. } => vec![y],
            OpDataflow::Map { out, .. } => vec![out],
            OpDataflow::Zip { out, .. } => vec![out],
            OpDataflow::Produce { out } => vec![out],
            OpDataflow::Attn { out, .. } => vec![out],
            OpDataflow::AddInPlace { residual, .. } => vec![residual],
            OpDataflow::UpdateProduce { residual, out, .. } => vec![residual, out],
        }
    }
}

/// A fully-resolved kernel dispatch. The compiler picked `pipeline`,
/// every binding's buffer + byte offset, the `grid`, and the typed
/// `dataflow`. The player only replays the bindings + grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch {
    pub op: OpKind,
    pub pipeline: PipeId,
    pub bindings: Vec<Binding>,
    pub grid: Grid,
    /// Typed, fixed-arity arena dataflow (validation only — never executed).
    pub dataflow: OpDataflow,
}

impl Dispatch {
    /// One N-block of a 4-bit affine matvec. Bindings match
    /// `quantized_qmv.metal`: 0=packed weight, 1=scales, 2=biases,
    /// 3=x activations, 4=y output. The weight/scales/biases/y bindings
    /// carry the block's row offset; `pipeline` is the `OUT_VEC_SIZE=nb`
    /// specialization; `grid` is the block's grid. `reads`/`write` are the
    /// dataflow (x read, the y column-slice this block writes). Exactly
    /// five operands — a qmv block cannot be built malformed.
    #[allow(clippy::too_many_arguments)]
    pub fn qmv_block(
        pipeline: PipeId,
        weight: Binding,
        scales: Binding,
        biases: Binding,
        x: Binding,
        y: Binding,
        grid: Grid,
        x_read: RegionRef,
        y_write: RegionRef,
    ) -> Self {
        Self {
            op: OpKind::QmvBlock,
            pipeline,
            bindings: vec![weight, scales, biases, x, y],
            grid,
            dataflow: OpDataflow::QmvBlock {
                x: x_read,
                y: y_write,
            },
        }
    }

    /// A generic whole-op dispatch (rmsnorm / rope / attention / silu·mul
    /// / add / embed). The compiler supplies the op tag, the resolved
    /// pipeline, the binding list, the grid, and the dataflow
    /// (`reads`/`write` regions).
    pub fn whole(
        op: OpKind,
        pipeline: PipeId,
        bindings: Vec<Binding>,
        grid: Grid,
        dataflow: OpDataflow,
    ) -> Self {
        Self {
            op,
            pipeline,
            bindings,
            grid,
            dataflow,
        }
    }
}

// ── Instructions & tape ─────────────────────────────────────────────

/// One instruction in the tape. Compute work is a fully-resolved
/// [`Dispatch`]; **dependencies are instructions** — `Wait`/`Signal` on a
/// point-to-point flag, placed by the compiler.
#[derive(Clone, Debug, PartialEq)]
pub enum SubtileInstr {
    Run(Dispatch),
    Wait(FlagId),
    Signal(FlagId),
}

/// The subtile instruction tape plus the buffer / pipeline tables it
/// references and the terminal buffer the logits land in.
#[derive(Clone, Debug)]
pub struct SubtileIr {
    pub buffers: Vec<BufferRef>,
    /// Element width (bytes) of each buffer, parallel to `buffers`. Lets
    /// the executor turn a `RegionRef`'s column extent into a byte extent
    /// for a real-buffer bounds check at resolve time.
    pub elem_bytes: Vec<u32>,
    pub pipelines: Vec<PipelineSpec>,
    pub num_flags: u32,
    pub tape: Vec<SubtileInstr>,
    /// The buffer whose contents are the forward result (logits).
    pub terminal: BufId,
}

// ── The trivial player ──────────────────────────────────────────────

/// The sink the player drives. The metal backend implements this:
/// `run` sets the resolved pipeline, binds each `(buffer, offset, index)`
/// against the pre-resolved buffer table, and dispatches `grid`;
/// `wait`/`signal` encode the p2p flag. All setup (buffer/pipeline
/// resolution, encoder creation) happens in the impl's constructor, NOT
/// per instruction — so the player below stays free of decisions.
pub trait Executor {
    fn run(&mut self, op: OpKind, pipeline: PipeId, bindings: &[Binding], grid: Grid);
    fn wait(&mut self, flag: FlagId);
    fn signal(&mut self, flag: FlagId);
}

/// The entire tape player. It cannot make anything up: one match arm per
/// instruction, each forwarding the already-resolved payload to the
/// `Executor`. Any cleverness is a bug — it belongs in the lowering that
/// produced `ir`.
pub fn play(ir: &SubtileIr, exec: &mut impl Executor) {
    for instr in &ir.tape {
        match instr {
            SubtileInstr::Run(d) => exec.run(d.op, d.pipeline, &d.bindings, d.grid),
            SubtileInstr::Wait(f) => exec.wait(*f),
            SubtileInstr::Signal(f) => exec.signal(*f),
        }
    }
}

// ── Structural + dataflow validation ────────────────────────────────

fn ranges_overlap(a: Range, b: Range) -> bool {
    a.start < b.end() && b.start < a.end()
}

/// Is `read`'s column extent fully covered by the union of `writes` whose
/// rows overlap it? (M-row, column-slice model: an N-block-tiled output is
/// covered iff its blocks tile the read's columns.) A `false` means a
/// use-before-def (no writer) or a partial-coverage gap (uninitialized
/// bytes) — both memory errors.
fn cols_covered(read: Region, writes: &[Region]) -> bool {
    let (rs, re) = (read.cols.start, read.cols.end());
    if rs >= re {
        return true; // empty read
    }
    let mut ivals: Vec<(u32, u32)> = writes
        .iter()
        .filter(|w| ranges_overlap(w.rows, read.rows))
        .map(|w| (w.cols.start, w.cols.end()))
        .collect();
    ivals.sort_unstable();
    let mut cursor = rs;
    for (s, e) in ivals {
        if s > cursor {
            break; // gap before `cursor` is reached
        }
        cursor = cursor.max(e);
        if cursor >= re {
            return true;
        }
    }
    cursor >= re
}

/// Check the IR without executing. Structural: every binding/read/write
/// buffer, every pipeline, and the terminal are in range; `elem_bytes`
/// matches `buffers`; every flag is `< num_flags` and every `Wait(f)` has
/// a preceding `Signal(f)` (deadlock-free).
///
/// **Dataflow (memory safety):** every region a dispatch READS from an
/// **arena slot** must be fully covered by the regions earlier dispatches
/// WROTE to that slot. A miss is a use-before-def (reading an unwritten /
/// wrong slot) or a partial-coverage gap (N-blocks leaving uninitialized
/// columns) — caught here at compile time instead of as GPU garbage.
/// Reads of weights / inputs / scratch are external (always defined) and
/// skip the check. Returns the instruction count.
pub fn validate(ir: &SubtileIr) -> Result<usize, String> {
    let n_buf = ir.buffers.len() as u32;
    let n_pipe = ir.pipelines.len() as u32;
    if ir.elem_bytes.len() != ir.buffers.len() {
        return Err(format!(
            "elem_bytes len {} != buffers len {}",
            ir.elem_bytes.len(),
            ir.buffers.len()
        ));
    }
    if ir.terminal.0 >= n_buf {
        return Err(format!("terminal buffer {} out of range", ir.terminal.0));
    }
    let is_arena = |b: BufId| matches!(ir.buffers.get(b.0 as usize), Some(BufferRef::ArenaSlot(_)));
    // Written regions per buffer, accumulated in tape order.
    let mut writes: Vec<Vec<Region>> = vec![Vec::new(); ir.buffers.len()];
    let mut signaled = vec![false; ir.num_flags as usize];
    for (i, instr) in ir.tape.iter().enumerate() {
        match instr {
            SubtileInstr::Run(d) => {
                if d.pipeline.0 >= n_pipe {
                    return Err(format!("instr {i}: pipeline {} out of range", d.pipeline.0));
                }
                for (b, bnd) in d.bindings.iter().enumerate() {
                    if bnd.buffer.0 >= n_buf {
                        return Err(format!(
                            "instr {i}: binding {b} buffer {} out of range",
                            bnd.buffer.0
                        ));
                    }
                }
                let df_reads = d.dataflow.reads();
                let df_writes = d.dataflow.writes();
                for r in &df_reads {
                    if r.buffer.0 >= n_buf {
                        return Err(format!(
                            "instr {i}: read buffer {} out of range",
                            r.buffer.0
                        ));
                    }
                }
                for w in &df_writes {
                    if w.buffer.0 >= n_buf {
                        return Err(format!(
                            "instr {i}: write buffer {} out of range",
                            w.buffer.0
                        ));
                    }
                }
                // Dataflow: arena-slot reads must be covered by prior writes.
                for r in &df_reads {
                    if is_arena(r.buffer) && !cols_covered(r.region, &writes[r.buffer.0 as usize]) {
                        return Err(format!(
                            "instr {i}: reads arena buffer {} region {:?} not covered by prior \
                             writes (use-before-def / partial coverage)",
                            r.buffer.0, r.region
                        ));
                    }
                }
                for w in &df_writes {
                    writes[w.buffer.0 as usize].push(w.region);
                }
            }
            SubtileInstr::Signal(f) => {
                if f.0 >= ir.num_flags {
                    return Err(format!("instr {i}: signal flag {} out of range", f.0));
                }
                signaled[f.0 as usize] = true;
            }
            SubtileInstr::Wait(f) => {
                if f.0 >= ir.num_flags {
                    return Err(format!("instr {i}: wait flag {} out of range", f.0));
                }
                if !signaled[f.0 as usize] {
                    return Err(format!(
                        "instr {i}: wait on flag {} with no preceding signal (would deadlock)",
                        f.0
                    ));
                }
            }
        }
    }
    Ok(ir.tape.len())
}

// ── The compiler's core: N-block tiling of one qmv ─────────────────
//
// This is where the linchpin lives, and it is deliberately HOST-PURE so
// the byte-offset arithmetic — the thing most likely to be wrong — is
// unit-tested without a GPU. The metal glue supplies the resolved
// operand buffers + the qmv kernel variant facts ([`QmvKernelInfo`],
// from `ferrite_metal_kernels::quantized`, the single source of truth)
// and calls [`tile_qmv`]; the result is appended to the tape verbatim.

/// `(buffer, base byte-offset)` for one qmv operand. The base offset is
/// whatever the metal weight/arena resolution produced for the WHOLE
/// tensor; [`tile_qmv`] adds the per-block row offset on top.
pub type Operand = (BufId, u64);

/// The five operands of an affine qmv, matching `quantized_qmv.metal`
/// argument order (0=weight, 1=scales, 2=biases, 3=x, 4=y).
#[derive(Clone, Copy, Debug)]
pub struct QmvOperands {
    pub weight: Operand,
    pub scales: Operand,
    pub biases: Operand,
    pub x: Operand,
    pub y: Operand,
}

/// Shape of the matvec being tiled. `m` must be 1 (decode); see
/// [`tile_qmv`] for why column-block tiling of `y` requires it.
#[derive(Clone, Copy, Debug)]
pub struct QmvShape {
    pub n: u32,
    pub k: u32,
    pub group_size: u32,
    pub bits: u32,
    pub m: u32,
}

/// The picked qmv kernel variant's dispatch facts for ONE block width,
/// sourced from `ferrite_metal_kernels::quantized` (`pick_qmv_kernel` +
/// `qmv_dispatch_shape` + `qmv_kernel_static_name`) so this crate never
/// re-derives — and can't drift from — the kernel's grid/variant rule.
/// The variant is picked **per block width** (a ragged tail of width 2
/// can't use `Fast`, which needs `N % 8 == 0`), so [`tile_qmv`] takes a
/// `Fn(width) -> QmvKernelInfo` rather than one fixed variant.
#[derive(Clone, Debug)]
pub struct QmvKernelInfo {
    pub library: &'static str,
    pub symbol: String,
    /// Output rows per threadgroup-y (`bn`): 64 for quad, 8 for
    /// fast/generic. Block `b` of width `w` dispatches `ceil(w/bn)`.
    pub bn: u32,
    pub tpt: [u32; 3],
}

/// Function-constant indices for qmv, matching the metal lowering
/// (`AffineQmvConstants`: K = fc0, N = fc1). The metal `Executor`'s
/// pipeline compile must agree.
pub const QMV_FC_K: u32 = 0;
pub const QMV_FC_N: u32 = 1;

/// Byte stride of one output row of the packed quantized weight
/// (`[out, in*bits/8]`): `k * bits / 8`. (4-bit, k=2048 ⇒ 1024 B/row.)
pub fn packed_weight_row_bytes(k: u32, bits: u32) -> u64 {
    (k as u64 * bits as u64) / 8
}

/// Byte stride of one output row of the affine scales/biases
/// (`[out, in/group_size]` at `scale_elem` bytes each).
pub fn affine_scale_row_bytes(k: u32, group_size: u32, scale_elem: u64) -> u64 {
    (k / group_size) as u64 * scale_elem
}

/// Interns [`PipelineSpec`]s into a table, deduping identical
/// `(library, symbol, constants)` so the N-blocks of equal width share
/// one [`PipeId`]. The metal side resolves each entry to a pipeline
/// state once.
#[derive(Default)]
pub struct PipelineInterner {
    pub specs: Vec<PipelineSpec>,
}

impl PipelineInterner {
    pub fn intern(&mut self, spec: PipelineSpec) -> PipeId {
        if let Some(i) = self.specs.iter().position(|s| *s == spec) {
            return PipeId(i as u32);
        }
        self.specs.push(spec);
        PipeId(self.specs.len() as u32 - 1)
    }
}

/// Tile one **M=1** quantized matvec `y[0, 0..n] = qmv(W[0..n, :], x)`
/// into `ceil(n / nb)` independent N-block `Run(QmvBlock)` instructions,
/// each computing a disjoint `block_width × k` slice of the output.
///
/// Per block at output rows `[n0, n0+w)`:
///  - weight / scales / biases bindings carry `base + n0 * row_stride`
///    (the linchpin: the kernel indexes relative to the offset base,
///    so it sees a standalone `w × k` matvec — no kernel changes);
///  - `x` is unchanged (every block reads the whole activation);
///  - `y` carries `base + n0 * act_elem` (the block's output columns);
///  - the pipeline is specialized to `N = w` (and `K = k`);
///  - the grid is `(m, ceil(w / bn), 1)`.
///
/// **M=1 only:** a column block `[n0, n0+w)` of a row-major `[m, n]`
/// output is contiguous iff `m == 1`. For `m > 1` the kernel's `N = w`
/// output stride would not match the real row stride `n`; that case
/// (small-batch spec-decode) needs output-stride handling and is
/// deferred. The blocks are independent (disjoint `y` columns) so no
/// `Wait`/`Signal` is emitted between them — a consumer of the whole
/// `y` is ordered after them by the single-stream tape (the wavefront
/// scheduler inserts cross-worker flags later).
#[allow(clippy::too_many_arguments)]
pub fn tile_qmv(
    ops: &QmvOperands,
    shape: QmvShape,
    nb: u32,
    info_for: impl Fn(u32) -> QmvKernelInfo,
    act_elem: u64,
    scale_elem: u64,
    pipelines: &mut PipelineInterner,
) -> Vec<SubtileInstr> {
    assert_eq!(shape.m, 1, "tile_qmv: column-block tiling requires m == 1");
    assert!(nb >= 1, "tile_qmv: nb must be >= 1");
    let w_stride = packed_weight_row_bytes(shape.k, shape.bits);
    let s_stride = affine_scale_row_bytes(shape.k, shape.group_size, scale_elem);
    let mut out = Vec::new();
    let mut n0 = 0u32;
    while n0 < shape.n {
        let w = nb.min(shape.n - n0);
        // Variant + grid picked for THIS block's width (the tail may
        // need a different kernel than the full blocks).
        let info = info_for(w);
        let pipe = pipelines.intern(PipelineSpec {
            library: info.library,
            symbol: info.symbol.clone(),
            constants: vec![
                FnConst {
                    index: QMV_FC_K,
                    value: ConstValue::I32(shape.k as i32),
                },
                FnConst {
                    index: QMV_FC_N,
                    value: ConstValue::I32(w as i32),
                },
            ],
        });
        let off = n0 as u64;
        let d = Dispatch::qmv_block(
            pipe,
            Binding::new(ops.weight.0, ops.weight.1 + off * w_stride, 0),
            Binding::new(ops.scales.0, ops.scales.1 + off * s_stride, 1),
            Binding::new(ops.biases.0, ops.biases.1 + off * s_stride, 2),
            Binding::new(ops.x.0, ops.x.1, 3),
            Binding::new(ops.y.0, ops.y.1 + off * act_elem, 4),
            Grid::new([shape.m, w.div_ceil(info.bn), 1], info.tpt),
            // Dataflow: reads the whole activation row; writes this
            // block's output columns [n0, n0+w). Weights/scales/biases are
            // external (always defined) so they're not dataflow reads.
            RegionRef::rows_cols(ops.x.0, shape.m, 0, shape.k),
            RegionRef::rows_cols(ops.y.0, shape.m, n0, w),
        );
        out.push(SubtileInstr::Run(d));
        n0 += w;
    }
    out
}

// ── Backend-agnostic IR construction ───────────────────────────────

/// Accumulates a [`SubtileIr`] while interning its buffer + pipeline
/// tables (so equal `BufferRef`s / `PipelineSpec`s collapse to one
/// entry). A per-backend compiler drives it: resolve operands to
/// [`BufId`]s via [`buffer`](Self::buffer), append dispatches (whole ops
/// via [`whole`](Self::whole), tiled matmuls via [`qmv`](Self::qmv)),
/// place sync via [`flag`](Self::flag)/[`signal`](Self::signal)/
/// [`wait`](Self::wait), then [`finish`](Self::finish). Backend-neutral —
/// the Metal and (future) CUDA compilers share it; only the content they
/// feed in differs.
#[derive(Default)]
pub struct SubtileIrBuilder {
    buffers: Vec<BufferRef>,
    elem_bytes: Vec<u32>,
    pipelines: PipelineInterner,
    tape: Vec<SubtileInstr>,
    num_flags: u32,
}

impl SubtileIrBuilder {
    /// Intern a logical buffer with its element width (bytes), returning
    /// its (deduped) [`BufId`].
    pub fn buffer(&mut self, b: BufferRef, elem_bytes: u32) -> BufId {
        if let Some(i) = self.buffers.iter().position(|x| *x == b) {
            return BufId(i as u32);
        }
        self.buffers.push(b);
        self.elem_bytes.push(elem_bytes);
        BufId(self.buffers.len() as u32 - 1)
    }

    /// Intern a pipeline spec, returning its (deduped) [`PipeId`].
    pub fn pipeline(&mut self, spec: PipelineSpec) -> PipeId {
        self.pipelines.intern(spec)
    }

    /// Append a fully-resolved whole-op dispatch with its typed dataflow.
    pub fn whole(
        &mut self,
        op: OpKind,
        pipeline: PipeId,
        bindings: Vec<Binding>,
        grid: Grid,
        dataflow: OpDataflow,
    ) {
        self.tape.push(SubtileInstr::Run(Dispatch::whole(
            op, pipeline, bindings, grid, dataflow,
        )));
    }

    /// Append a pre-built dispatch (e.g. a `qmv_block`).
    pub fn run(&mut self, d: Dispatch) {
        self.tape.push(SubtileInstr::Run(d));
    }

    /// N-block-tile one M=1 qmv (via [`tile_qmv`], interning each block's
    /// pipeline) and append the blocks to the tape.
    pub fn qmv(
        &mut self,
        ops: &QmvOperands,
        shape: QmvShape,
        nb: u32,
        info_for: impl Fn(u32) -> QmvKernelInfo,
        act_elem: u64,
        scale_elem: u64,
    ) {
        let instrs = tile_qmv(
            ops,
            shape,
            nb,
            info_for,
            act_elem,
            scale_elem,
            &mut self.pipelines,
        );
        self.tape.extend(instrs);
    }

    /// Allocate a fresh sync flag.
    pub fn flag(&mut self) -> FlagId {
        let f = FlagId(self.num_flags);
        self.num_flags += 1;
        f
    }
    pub fn signal(&mut self, f: FlagId) {
        self.tape.push(SubtileInstr::Signal(f));
    }
    pub fn wait(&mut self, f: FlagId) {
        self.tape.push(SubtileInstr::Wait(f));
    }

    /// Seal the IR. `terminal` is the buffer holding the forward result.
    pub fn finish(self, terminal: BufId) -> SubtileIr {
        SubtileIr {
            buffers: self.buffers,
            elem_bytes: self.elem_bytes,
            pipelines: self.pipelines.specs,
            num_flags: self.num_flags,
            tape: self.tape,
            terminal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `Executor` that records every call so a test can assert the
    /// player forwards the tape verbatim, in order, adding nothing.
    #[derive(Default)]
    struct Recorder {
        trace: Vec<String>,
    }
    impl Executor for Recorder {
        fn run(&mut self, op: OpKind, pipeline: PipeId, bindings: &[Binding], grid: Grid) {
            self.trace.push(format!(
                "run {op:?} p{} b{} g{:?}",
                pipeline.0,
                bindings.len(),
                grid.tg
            ));
        }
        fn wait(&mut self, flag: FlagId) {
            self.trace.push(format!("wait {}", flag.0));
        }
        fn signal(&mut self, flag: FlagId) {
            self.trace.push(format!("signal {}", flag.0));
        }
    }

    fn pipe(symbol: &str) -> PipelineSpec {
        PipelineSpec {
            library: "quantized_qmv",
            symbol: symbol.to_string(),
            constants: vec![],
        }
    }

    /// A small two-block qmv + sync + consumer tape: a hand-built IR with
    /// a VALID dataflow (the two blocks tile y's 8 columns; the consumer
    /// reads the whole y, covered) that validates and whose play() trace
    /// is exactly the tape in order.
    fn sample_ir() -> SubtileIr {
        let wl = WeightLoc {
            layer: 0,
            bucket: 0,
            op_idx: 0,
            slot: 0,
        };
        // 0=W 1=scales 2=biases (external weights), 3=x (external input),
        // 4=y (arena, written by the blocks), 5=z (arena, consumer output).
        let buffers = vec![
            BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::Weight,
                loc: wl,
            },
            BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::AffineScales,
                loc: wl,
            },
            BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::AffineBiases,
                loc: wl,
            },
            BufferRef::Input(InputKind::InputIds),
            BufferRef::ArenaSlot(4),
            BufferRef::ArenaSlot(5),
        ];
        let elem_bytes = vec![4, 2, 2, 2, 2, 2];
        let pipelines = vec![pipe("qmv_nb4"), pipe("add")];
        let (w, s, b, x, y, z) = (BufId(0), BufId(1), BufId(2), BufId(3), BufId(4), BufId(5));
        // y has 8 columns; block 0 writes cols [0,4), block 1 writes [4,8).
        let blk0 = Dispatch::qmv_block(
            PipeId(0),
            Binding::new(w, 0, 0),
            Binding::new(s, 0, 1),
            Binding::new(b, 0, 2),
            Binding::whole(x, 3),
            Binding::new(y, 0, 4),
            Grid::new([1, 1, 1], [32, 2, 1]),
            RegionRef::rows_cols(x, 1, 0, 8),
            RegionRef::rows_cols(y, 1, 0, 4),
        );
        let blk1 = Dispatch::qmv_block(
            PipeId(0),
            Binding::new(w, 8, 0),
            Binding::new(s, 8, 1),
            Binding::new(b, 8, 2),
            Binding::whole(x, 3),
            Binding::new(y, 8, 4),
            Grid::new([1, 1, 1], [32, 2, 1]),
            RegionRef::rows_cols(x, 1, 0, 8),
            RegionRef::rows_cols(y, 1, 4, 4),
        );
        // Consumer reads the whole y (cols [0,8), covered by both blocks)
        // and writes z.
        let add = Dispatch::whole(
            OpKind::Add,
            PipeId(1),
            vec![Binding::whole(y, 0), Binding::whole(z, 1)],
            Grid::new([1, 1, 1], [256, 1, 1]),
            OpDataflow::Map {
                x: RegionRef::rows_cols(y, 1, 0, 8),
                out: RegionRef::rows_cols(z, 1, 0, 8),
            },
        );
        SubtileIr {
            buffers,
            elem_bytes,
            pipelines,
            num_flags: 1,
            tape: vec![
                SubtileInstr::Run(blk0),
                SubtileInstr::Run(blk1),
                SubtileInstr::Signal(FlagId(0)),
                SubtileInstr::Wait(FlagId(0)),
                SubtileInstr::Run(add),
            ],
            terminal: z,
        }
    }

    #[test]
    fn sample_ir_validates() {
        assert_eq!(validate(&sample_ir()).unwrap(), 5);
    }

    #[test]
    fn player_forwards_tape_verbatim() {
        let ir = sample_ir();
        let mut rec = Recorder::default();
        play(&ir, &mut rec);
        assert_eq!(
            rec.trace,
            vec![
                "run QmvBlock p0 b5 g[1, 1, 1]",
                "run QmvBlock p0 b5 g[1, 1, 1]",
                "signal 0",
                "wait 0",
                "run Add p1 b2 g[1, 1, 1]",
            ]
        );
    }

    #[test]
    fn qmv_block_has_exactly_five_bindings() {
        let d = Dispatch::qmv_block(
            PipeId(0),
            Binding::new(BufId(0), 0, 0),
            Binding::new(BufId(1), 0, 1),
            Binding::new(BufId(2), 0, 2),
            Binding::whole(BufId(3), 3),
            Binding::new(BufId(4), 0, 4),
            Grid::new([1, 1, 1], [32, 2, 1]),
            RegionRef::rows_cols(BufId(3), 1, 0, 64),
            RegionRef::rows_cols(BufId(4), 1, 0, 8),
        );
        assert_eq!(d.bindings.len(), 5);
        assert_eq!(d.op, OpKind::QmvBlock);
    }

    #[test]
    fn validate_rejects_out_of_range_buffer() {
        let mut ir = sample_ir();
        if let SubtileInstr::Run(d) = &mut ir.tape[0] {
            d.bindings[0].buffer = BufId(99);
        }
        assert!(validate(&ir).is_err());
    }

    #[test]
    fn validate_rejects_out_of_range_pipeline() {
        let mut ir = sample_ir();
        if let SubtileInstr::Run(d) = &mut ir.tape[0] {
            d.pipeline = PipeId(99);
        }
        assert!(validate(&ir).is_err());
    }

    #[test]
    fn validate_rejects_wait_before_signal() {
        let mut ir = sample_ir();
        // Swap signal/wait order → wait now precedes its signal.
        ir.tape.swap(2, 3);
        assert!(validate(&ir).is_err());
    }

    #[test]
    fn validate_rejects_out_of_range_flag() {
        let mut ir = sample_ir();
        ir.tape.push(SubtileInstr::Signal(FlagId(7)));
        assert!(validate(&ir).is_err());
    }

    // ── tile_qmv: the linchpin offset math ──────────────────────────

    fn fast_info() -> QmvKernelInfo {
        QmvKernelInfo {
            library: "quantized_qmv",
            symbol: "affine_qmv_fast_f16_s_f16_gs_64_b_4_batch_0".into(),
            bn: 8,
            tpt: [32, 2, 1],
        }
    }
    fn quad_info() -> QmvKernelInfo {
        QmvKernelInfo {
            library: "quantized_qmv",
            symbol: "affine_qmv_quad_f16_s_f16_gs_64_b_4_d_64_batch_0".into(),
            bn: 64,
            tpt: [32, 1, 1],
        }
    }
    fn generic_info() -> QmvKernelInfo {
        QmvKernelInfo {
            library: "quantized_qmv",
            symbol: "affine_qmv_f16_s_f16_gs_64_b_4_batch_0".into(),
            bn: 8,
            tpt: [32, 2, 1],
        }
    }

    fn ops() -> QmvOperands {
        // distinct base offsets so an operand mix-up is visible.
        QmvOperands {
            weight: (BufId(0), 0),
            scales: (BufId(1), 0),
            biases: (BufId(2), 0),
            x: (BufId(3), 0),
            y: (BufId(4), 0),
        }
    }

    fn as_dispatch(instr: &SubtileInstr) -> &Dispatch {
        match instr {
            SubtileInstr::Run(d) => d,
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn stride_helpers_match_mlx_affine_b4_g64() {
        // 4-bit packed: 2048 inputs * 4 bits / 8 = 1024 bytes/row.
        assert_eq!(packed_weight_row_bytes(2048, 4), 1024);
        // f16 scales: 2048/64 = 32 groups * 2 bytes = 64 bytes/row.
        assert_eq!(affine_scale_row_bytes(2048, 64, 2), 64);
    }

    #[test]
    fn tile_qmv_even_split_offsets_and_dedup() {
        // n=128, nb=32, k=2048, g=64, 4-bit, M=1 → 4 equal blocks.
        let mut pl = PipelineInterner::default();
        let instrs = tile_qmv(
            &ops(),
            QmvShape {
                n: 128,
                k: 2048,
                group_size: 64,
                bits: 4,
                m: 1,
            },
            32,
            |_w| fast_info(),
            2, // act elem (f16)
            2, // scale elem (f16)
            &mut pl,
        );
        assert_eq!(instrs.len(), 4, "ceil(128/32)");
        // All four blocks have the same width → one shared pipeline.
        assert_eq!(pl.specs.len(), 1, "equal-width blocks dedup to 1 pipeline");
        assert_eq!(
            pl.specs[0].constants,
            vec![
                FnConst {
                    index: QMV_FC_K,
                    value: ConstValue::I32(2048)
                },
                FnConst {
                    index: QMV_FC_N,
                    value: ConstValue::I32(32)
                },
            ]
        );
        for (b, instr) in instrs.iter().enumerate() {
            let d = as_dispatch(instr);
            let n0 = (b as u64) * 32;
            assert_eq!(d.op, OpKind::QmvBlock);
            // weight: n0 * 1024,  scales/biases: n0 * 64,  y: n0 * 2,  x: 0.
            assert_eq!(d.bindings[0].offset, n0 * 1024, "weight off blk {b}");
            assert_eq!(d.bindings[1].offset, n0 * 64, "scales off blk {b}");
            assert_eq!(d.bindings[2].offset, n0 * 64, "biases off blk {b}");
            assert_eq!(d.bindings[3].offset, 0, "x not row-offset blk {b}");
            assert_eq!(d.bindings[4].offset, n0 * 2, "y off blk {b}");
            // grid: (M=1, ceil(32/8)=4, 1), tpt from variant.
            assert_eq!(d.grid.tg, [1, 4, 1]);
            assert_eq!(d.grid.tpt, [32, 2, 1]);
        }
    }

    #[test]
    fn tile_qmv_ragged_tail_picks_valid_variant_per_block() {
        // n=130, nb=64 → widths 64, 64, 2. Full blocks (N%8==0) take the
        // fast variant; the width-2 tail can't (fast needs N%8==0), so the
        // per-width closure routes it to generic — proving variant pick is
        // per block, not once for the matmul.
        let mut pl = PipelineInterner::default();
        let info_for = |w: u32| {
            if w.is_multiple_of(8) {
                fast_info()
            } else {
                generic_info()
            }
        };
        let instrs = tile_qmv(
            &ops(),
            QmvShape {
                n: 130,
                k: 2048,
                group_size: 64,
                bits: 4,
                m: 1,
            },
            64,
            info_for,
            2,
            2,
            &mut pl,
        );
        assert_eq!(instrs.len(), 3);
        // Distinct (symbol, N) → two pipelines: fast/N=64 (blocks 0,1) and
        // generic/N=2 (tail).
        assert_eq!(pl.specs.len(), 2);
        let offs: Vec<u64> = instrs
            .iter()
            .map(|i| as_dispatch(i).bindings[0].offset)
            .collect();
        assert_eq!(offs, vec![0, 64 * 1024, 128 * 1024], "weight row offsets");
        let tail = as_dispatch(&instrs[2]);
        assert_eq!(tail.bindings[4].offset, 128 * 2, "tail y offset");
        assert_eq!(tail.grid.tg, [1, 1, 1], "ceil(2/8)=1");
        let tail_pipe = &pl.specs[tail.pipeline.0 as usize];
        assert_eq!(
            tail_pipe.symbol,
            generic_info().symbol,
            "tail routed to generic"
        );
        assert_eq!(
            tail_pipe.constants[1],
            FnConst {
                index: QMV_FC_N,
                value: ConstValue::I32(2)
            }
        );
        // The full blocks share the fast pipeline.
        let blk0 = as_dispatch(&instrs[0]);
        assert_eq!(
            pl.specs[blk0.pipeline.0 as usize].symbol,
            fast_info().symbol
        );
    }

    #[test]
    fn tile_qmv_coarse_single_block_when_nb_ge_n() {
        // nb >= n → one whole-matrix block (equivalent to no tiling).
        let mut pl = PipelineInterner::default();
        let instrs = tile_qmv(
            &ops(),
            QmvShape {
                n: 512,
                k: 512,
                group_size: 64,
                bits: 4,
                m: 1,
            },
            4096,
            |_w| fast_info(),
            2,
            2,
            &mut pl,
        );
        assert_eq!(instrs.len(), 1);
        let d = as_dispatch(&instrs[0]);
        assert_eq!(d.bindings[0].offset, 0);
        assert_eq!(d.grid.tg, [1, 512u32.div_ceil(8), 1]);
    }

    /// A 1-read-1-write whole op, for building dataflow tests cheaply.
    fn map_op(x: RegionRef, out: RegionRef) -> Dispatch {
        Dispatch::whole(
            OpKind::RmsNorm,
            PipeId(0),
            vec![],
            Grid::new([1, 1, 1], [1, 1, 1]),
            OpDataflow::Map { x, out },
        )
    }

    #[test]
    fn validate_rejects_read_before_write() {
        // An op reads arena slot `a` that NO prior op wrote → use-before-def.
        let ir = SubtileIr {
            buffers: vec![BufferRef::ArenaSlot(0), BufferRef::ArenaSlot(1)],
            elem_bytes: vec![2, 2],
            pipelines: vec![pipe("rms")],
            num_flags: 0,
            tape: vec![SubtileInstr::Run(map_op(
                RegionRef::rows_cols(BufId(0), 1, 0, 8), // reads a (unwritten!)
                RegionRef::rows_cols(BufId(1), 1, 0, 8),
            ))],
            terminal: BufId(1),
        };
        let err = validate(&ir).unwrap_err();
        assert!(
            err.contains("not covered"),
            "expected use-before-def, got: {err}"
        );
    }

    #[test]
    fn validate_rejects_partial_coverage() {
        // Two writes cover y[0,4) and y[6,10) — a gap at [4,6). A consumer
        // reading the whole y[0,10) must be rejected (uninitialized bytes).
        let x = BufId(0); // external input
        let y = BufId(1);
        let z = BufId(2);
        let ir = SubtileIr {
            buffers: vec![
                BufferRef::Input(InputKind::InputIds),
                BufferRef::ArenaSlot(1),
                BufferRef::ArenaSlot(2),
            ],
            elem_bytes: vec![2, 2, 2],
            pipelines: vec![pipe("rms")],
            num_flags: 0,
            tape: vec![
                SubtileInstr::Run(map_op(
                    RegionRef::rows_cols(x, 1, 0, 1),
                    RegionRef::rows_cols(y, 1, 0, 4),
                )),
                SubtileInstr::Run(map_op(
                    RegionRef::rows_cols(x, 1, 0, 1),
                    RegionRef::rows_cols(y, 1, 6, 4), // [6,10) — leaves [4,6) unwritten
                )),
                SubtileInstr::Run(map_op(
                    RegionRef::rows_cols(y, 1, 0, 10), // reads whole y → gap!
                    RegionRef::rows_cols(z, 1, 0, 10),
                )),
            ],
            terminal: z,
        };
        let err = validate(&ir).unwrap_err();
        assert!(
            err.contains("not covered"),
            "expected partial-coverage gap, got: {err}"
        );
    }

    /// Two writes that tile y[0,4)+[4,10) fully cover a whole y[0,10) read.
    #[test]
    fn validate_accepts_full_coverage() {
        let (x, y, z) = (BufId(0), BufId(1), BufId(2));
        let ir = SubtileIr {
            buffers: vec![
                BufferRef::Input(InputKind::InputIds),
                BufferRef::ArenaSlot(1),
                BufferRef::ArenaSlot(2),
            ],
            elem_bytes: vec![2, 2, 2],
            pipelines: vec![pipe("rms")],
            num_flags: 0,
            tape: vec![
                SubtileInstr::Run(map_op(
                    RegionRef::rows_cols(x, 1, 0, 1),
                    RegionRef::rows_cols(y, 1, 0, 4),
                )),
                SubtileInstr::Run(map_op(
                    RegionRef::rows_cols(x, 1, 0, 1),
                    RegionRef::rows_cols(y, 1, 4, 6), // [4,10) — now fully tiled
                )),
                SubtileInstr::Run(map_op(
                    RegionRef::rows_cols(y, 1, 0, 10),
                    RegionRef::rows_cols(z, 1, 0, 10),
                )),
            ],
            terminal: z,
        };
        assert!(validate(&ir).is_ok());
    }

    #[test]
    fn builder_interns_dedups_and_finishes() {
        let mut b = SubtileIrBuilder::default();
        let wl = WeightLoc {
            layer: 0,
            bucket: 0,
            op_idx: 0,
            slot: 0,
        };
        let w = b.buffer(
            BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::Weight,
                loc: wl,
            },
            4,
        );
        let s = b.buffer(
            BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::AffineScales,
                loc: wl,
            },
            2,
        );
        let bi = b.buffer(
            BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role: WeightRole::AffineBiases,
                loc: wl,
            },
            2,
        );
        // x is an external input (so the qmv's read of it is always
        // defined); y is the arena slot the blocks write.
        let x = b.buffer(BufferRef::Input(InputKind::InputIds), 2);
        let y = b.buffer(BufferRef::ArenaSlot(4), 2);
        assert_eq!(b.buffer(BufferRef::ArenaSlot(4), 2), y, "buffer dedup");

        let ops = QmvOperands {
            weight: (w, 0),
            scales: (s, 0),
            biases: (bi, 0),
            x: (x, 0),
            y: (y, 0),
        };
        b.qmv(
            &ops,
            QmvShape {
                n: 64,
                k: 512,
                group_size: 64,
                bits: 4,
                m: 1,
            },
            32,
            |_w| generic_info(),
            2,
            2,
        );
        let ir = b.finish(y);
        assert_eq!(ir.tape.len(), 2, "64/32 = 2 blocks");
        assert_eq!(ir.pipelines.len(), 1, "equal width → 1 pipeline");
        assert_eq!(ir.buffers.len(), 5, "5 distinct buffers");
        assert!(validate(&ir).is_ok());
    }

    #[test]
    #[should_panic(expected = "m == 1")]
    fn tile_qmv_rejects_m_gt_1() {
        let mut pl = PipelineInterner::default();
        tile_qmv(
            &ops(),
            QmvShape {
                n: 128,
                k: 512,
                group_size: 64,
                bits: 4,
                m: 2,
            },
            32,
            |_w| fast_info(),
            2,
            2,
            &mut pl,
        );
    }
}
