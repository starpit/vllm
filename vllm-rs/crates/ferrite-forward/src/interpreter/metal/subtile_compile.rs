// SPDX-License-Identifier: Apache-2.0
//! The Metal subtile COMPILER: lowers a decode `Instruction` tape into a
//! [`SubtileIr`] that the trivial [`super::subtile_player::MetalExecutor`]
//! replays.
//!
//! It talks subtiles where subtiling happens. An `Instruction::AffineQmm`
//! matmul is emitted NATIVELY as N-block qmv subtiles — built straight
//! from the instruction's own `(n, k, group_size, bits, in/out slot,
//! layer)` plus `quantized.rs`'s variant pick + symbol and the
//! `affine_qmm_bindings` weight-locator convention, so each block matches
//! the baseline's whole qmv per-output ⇒ bit-exact. `lower_one` is NOT in
//! this path — it talks whole ops, not subtiles.
//!
//! Every genuinely whole op (rmsnorm / rope / attention / silu·mul / add
//! / embed / the synth megakernels) IS a single subtile = its whole-op
//! dispatch, so the compiler sources that one dispatch from `lower_one`
//! (the whole-op authority) and translates `LoweredCommand → Dispatch`.
//! No re-derivation of ~1500 lines of binding/grid/symbol logic; no drift.
#![cfg(feature = "metal")]

use ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile;
use ferrite_metal_kernels::quantized::{QmvKernel, pick_qmv_kernel, qmv_kernel_static_name};
use ferrite_metal_kernels::specialized_pipeline_cache::{ConstantType, ConstantValue};
use ferrite_wavefront::subtile_ir::{
    Binding as IrBinding, BufId, BufferRef, ConstValue, FnConst, Grid, InputKind, OpDataflow,
    OpKind, PipelineSpec, QmvKernelInfo, QmvOperands, QmvShape, RegionRef, SubtileIr,
    SubtileIrBuilder, WeightBundle, WeightLoc, WeightRole,
};
use std::collections::HashMap;

use super::lowered::{
    Binding, KernelId, LoweredCommand, LoweringError, RuntimeBindingKind, WeightBundleKind,
    WeightTensor,
};
use super::lowering::{dequant_dtype_for, lower_one, scale_dtype_for};
use crate::{CanonicalParams, Instruction};

/// Why a decode tape couldn't be compiled to a [`SubtileIr`]. Always
/// surfaced; never papered over.
#[derive(Debug)]
pub enum CompileError {
    /// `lower_one` failed on a whole op.
    Lowering(LoweringError),
    /// A weight bundle outside the decode set (e.g. MoE).
    UnsupportedBundle(WeightBundleKind),
    /// A weight tensor outside the decode set (e.g. a `Moe*` tensor).
    UnsupportedTensor(WeightTensor),
    /// A MoE-only binding (`Inline` setBytes immediate or `MoeScratch`
    /// sub-region) — outside the dense decode set, not modeled.
    UnsupportedBinding,
    /// Called with `bucket_m != 1`; the subtile decode path is M=1 only.
    NotDecode { bucket_m: u32 },
    /// A `Loop(count, body_len)` whose body runs past the slice end.
    MalformedLoop {
        index: usize,
        count: u32,
        body_len: u32,
    },
    /// A nested `Loop` inside a loop body (not produced by the macro).
    NestedLoop,
    /// An instruction that lowered to a dispatch but has no `OpDataflow`
    /// model yet — surfaced (never silently skipped) so the dataflow
    /// validator can't be bypassed by an unmodeled op.
    UnmodeledDataflow(&'static str),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lowering(e) => write!(f, "lower_one failed: {e:?}"),
            Self::UnsupportedBundle(b) => write!(f, "unsupported weight bundle {b:?}"),
            Self::UnsupportedTensor(t) => write!(f, "unsupported weight tensor {t:?}"),
            Self::UnsupportedBinding => {
                write!(f, "MoE-only binding (Inline/MoeScratch) not modeled")
            }
            Self::NotDecode { bucket_m } => {
                write!(f, "subtile decode requires bucket_m=1, got {bucket_m}")
            }
            Self::MalformedLoop {
                index,
                count,
                body_len,
            } => write!(
                f,
                "malformed Loop at {index}: count={count} body_len={body_len} runs past slice end"
            ),
            Self::NestedLoop => write!(f, "nested Loop in a loop body (unexpected)"),
            Self::UnmodeledDataflow(op) => {
                write!(f, "instruction `{op}` has no OpDataflow model yet")
            }
        }
    }
}
impl std::error::Error for CompileError {}

// ── neutral-IR mappers (metal type → subtile_ir type) ───────────────

fn map_bundle(k: WeightBundleKind) -> Result<WeightBundle, CompileError> {
    Ok(match k {
        WeightBundleKind::Embedding => WeightBundle::Embedding,
        WeightBundleKind::RmsNorm => WeightBundle::RmsNorm,
        WeightBundleKind::LinearLayer => WeightBundle::LinearLayer,
        WeightBundleKind::CosSin => WeightBundle::CosSin,
        WeightBundleKind::AffineQuantEmbedding => WeightBundle::AffineQuantEmbedding,
        other => return Err(CompileError::UnsupportedBundle(other)),
    })
}

fn map_role(t: WeightTensor) -> Result<WeightRole, CompileError> {
    Ok(match t {
        WeightTensor::Weight => WeightRole::Weight,
        WeightTensor::Bias => WeightRole::Bias,
        WeightTensor::AffineScales => WeightRole::AffineScales,
        WeightTensor::AffineBiases => WeightRole::AffineBiases,
        WeightTensor::AffineLinearBias => WeightRole::AffineLinearBias,
        other => return Err(CompileError::UnsupportedTensor(other)),
    })
}

fn map_input(k: RuntimeBindingKind) -> InputKind {
    match k {
        RuntimeBindingKind::InputIds => InputKind::InputIds,
        RuntimeBindingKind::Positions => InputKind::Positions,
        RuntimeBindingKind::SlotMapping => InputKind::SlotMapping,
        RuntimeBindingKind::CuSeqlensQ => InputKind::CuSeqlensQ,
        RuntimeBindingKind::SeqUsedK => InputKind::SeqUsedK,
        RuntimeBindingKind::BlockTable => InputKind::BlockTable,
        RuntimeBindingKind::KvCacheK { layer } => InputKind::KvCacheK { layer: layer.0 },
        RuntimeBindingKind::KvCacheV { layer } => InputKind::KvCacheV { layer: layer.0 },
        RuntimeBindingKind::NumTokensU32 => InputKind::NumTokens,
    }
}

fn const_from_metal(cv: &ConstantValue) -> FnConst {
    let value = match cv.ty {
        ConstantType::UInt => ConstValue::U32(cv.bits),
        ConstantType::Int => ConstValue::I32(cv.bits as i32),
        ConstantType::Float => ConstValue::F32(f32::from_bits(cv.bits)),
        ConstantType::Bool => ConstValue::Bool(cv.bits != 0),
    };
    FnConst {
        index: cv.index as u32,
        value,
    }
}

/// A column extent meaning "the whole slot, width unknown" — used for the
/// writes of whole ops (which write their entire output) and as the
/// default read width. It covers any precise read; only N-block-tiled qmv
/// outputs are recorded with a precise width (so a tiling gap is caught).
const WHOLE: u32 = u32::MAX;

/// Element width (bytes) of a logical buffer — recorded on the IR for a
/// future resolve-time byte-bounds check. Approximate today (the dataflow
/// validator works in element columns, not bytes); packed 4-bit weight is
/// u32, affine scales/biases + activations are 2-byte, index inputs u32.
fn elem_of(b: &BufferRef) -> u32 {
    match b {
        BufferRef::Weight { role, .. } => match role {
            WeightRole::Weight => 4, // packed u32 (4-bit) — the qmv matrix
            _ => 2,                  // scales / biases / dense gain (f16)
        },
        BufferRef::ArenaSlot(_) | BufferRef::Scratch(_) => 2, // f16/bf16 activations
        BufferRef::Input(InputKind::KvCacheK { .. } | InputKind::KvCacheV { .. }) => 2,
        BufferRef::Input(_) => 4, // u32 ids / positions / cu_seqlens / …
    }
}

/// Map a lowered (whole-op) `Binding` to a neutral `(BufferRef, arg index)`.
fn map_binding(b: &Binding) -> Result<(BufferRef, u32), CompileError> {
    Ok(match b {
        Binding::ArenaSlot {
            slot,
            binding_index,
        } => (BufferRef::ArenaSlot(*slot), *binding_index as u32),
        Binding::Scratch { binding_index } => (BufferRef::Scratch(0), *binding_index as u32),
        Binding::Runtime {
            kind,
            binding_index,
        } => (BufferRef::Input(map_input(*kind)), *binding_index as u32),
        Binding::Weight {
            kind,
            which,
            layer,
            locator,
            binding_index,
        } => (
            BufferRef::Weight {
                bundle: map_bundle(*kind)?,
                role: map_role(*which)?,
                loc: WeightLoc {
                    layer: layer.0,
                    bucket: locator.bucket,
                    op_idx: locator.op_idx,
                    slot: locator.slot,
                },
            },
            *binding_index as u32,
        ),
        Binding::Inline { .. } | Binding::MoeScratch { .. } => {
            return Err(CompileError::UnsupportedBinding);
        }
    })
}

/// Descriptive op tag (player ignores it). Minimal-safe mapping; anything
/// else → `Other` (the precise kernel lives in the `PipelineSpec`).
fn map_op(k: KernelId) -> OpKind {
    match k {
        KernelId::Embed => OpKind::Embed,
        KernelId::RmsNorm => OpKind::RmsNorm,
        KernelId::FusedAddRmsNorm => OpKind::FusedAddRmsNorm,
        KernelId::AffineQmv | KernelId::AffineQmvFast | KernelId::AffineQmvQuad => OpKind::QmvBlock,
        _ => OpKind::Other,
    }
}

/// Translate one whole-op `LoweredCommand` → a single whole `Dispatch`.
/// Grid is taken as-is: at `bucket_m == 1` the command's `m_scaling` is
/// the identity, so no token scaling is needed.
fn translate_whole(
    cmd: &LoweredCommand,
    builder: &mut SubtileIrBuilder,
    dataflow: OpDataflow,
) -> Result<(), CompileError> {
    let pipe = builder.pipeline(PipelineSpec {
        library: cmd.library,
        symbol: cmd.function.to_string(),
        constants: cmd.constants.iter().map(const_from_metal).collect(),
    });
    let mut bindings = Vec::with_capacity(cmd.bindings.len());
    for b in &cmd.bindings {
        let (bref, idx) = map_binding(b)?;
        let e = elem_of(&bref);
        let bid = builder.buffer(bref, e);
        bindings.push(IrBinding::new(bid, 0, idx));
    }
    let (tg, tpt) = (
        cmd.dispatch.threadgroups,
        cmd.dispatch.threads_per_threadgroup,
    );
    builder.whole(
        map_op(cmd.kernel),
        pipe,
        bindings,
        Grid::new([tg.0, tg.1, tg.2], [tpt.0, tpt.1, tpt.2]),
        dataflow,
    );
    Ok(())
}

/// Emit N-block qmv subtiles for one `Instruction::AffineQmm`, NATIVELY
/// (no `lower_one`). Variant + symbol + locator come from the same
/// sources the baseline uses, so each block is bit-exact vs the whole qmv.
#[allow(clippy::too_many_arguments)]
fn compile_affine_qmm<W: CanonicalParams>(
    builder: &mut SubtileIrBuilder,
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    n: u32,
    k: u32,
    gs: u32,
    bits: u32,
    layer_offset: u32,
    tape_index: u32,
    op_idx: u32,
    nb: u32,
    slot_width: &mut HashMap<u32, u32>,
) {
    let loc = WeightLoc {
        layer: layer + layer_offset,
        bucket: tape_index,
        op_idx,
        slot: 0,
    };
    let w = builder_weight(builder, WeightRole::Weight, loc);
    let sc = builder_weight(builder, WeightRole::AffineScales, loc);
    let bi = builder_weight(builder, WeightRole::AffineBiases, loc);
    let x = builder.buffer(BufferRef::ArenaSlot(in_slot), 2);
    let y = builder.buffer(BufferRef::ArenaSlot(out_slot), 2);
    let ops = QmvOperands {
        weight: (w, 0),
        scales: (sc, 0),
        biases: (bi, 0),
        x: (x, 0),
        y: (y, 0),
    };

    // The whole matmul's variant (matches the baseline). All blocks use
    // it; `gran` keeps each block width valid for it (Fast needs N%8==0).
    let variant = pick_qmv_kernel(n, k, bits);
    let (bn, tpt, gran) = match variant {
        QmvKernel::Quad { .. } => (64u32, [32u32, 1, 1], 1u32),
        QmvKernel::Fast => (8u32, [32u32, 2, 1], 8u32),
        QmvKernel::Generic => (8u32, [32u32, 2, 1], 1u32),
    };
    let nb_eff = (nb / gran).max(1) * gran;
    let symbol = qmv_kernel_static_name(
        variant,
        dequant_dtype_for::<W>(),
        scale_dtype_for::<W>(),
        bits,
        gs,
    )
    .to_string();
    let info_for = move |_w: u32| QmvKernelInfo {
        library: "quantized_qmv",
        symbol: symbol.clone(),
        bn,
        tpt,
    };
    // Activation + scale element bytes are 2 (f16 / bf16) on the metal path.
    builder.qmv(
        &ops,
        QmvShape {
            n,
            k,
            group_size: gs,
            bits,
            m: 1,
        },
        nb_eff,
        info_for,
        2,
        2,
    );
    // Record the qmv's output width (n). Its blocks tile [0, n); a
    // consumer reading [0, n) is covered ONLY if they fully tile it, so a
    // tiling gap surfaces as a partial-coverage error at validate time.
    slot_width.insert(out_slot, n);
}

fn builder_weight(builder: &mut SubtileIrBuilder, role: WeightRole, loc: WeightLoc) -> BufId {
    let bref = BufferRef::Weight {
        bundle: WeightBundle::LinearLayer,
        role,
        loc,
    };
    let e = elem_of(&bref);
    builder.buffer(bref, e)
}

/// Lower a decode tape (the bucket's `backbone` then `lm_head` slices,
/// each with its own tape index) into a [`SubtileIr`]: `AffineQmm` →
/// native N-block qmv subtiles, every other op → one whole-op dispatch.
/// `terminal_slot` is the arena slot holding the logits.
pub fn compile_decode<W: CanonicalParams>(
    segments: &[(&[Instruction], u32)],
    bucket_m: u32,
    profile: Option<&MetalTargetProfile>,
    nb: u32,
    terminal_slot: u32,
) -> Result<SubtileIr, CompileError> {
    if bucket_m != 1 {
        return Err(CompileError::NotDecode { bucket_m });
    }
    let mut builder = SubtileIrBuilder::default();
    let mut splitk_scratch = 0u32;
    let mut moe_scratch = 0u32;
    // Per-arena-slot producer width: a qmv records its precise output `n`;
    // a consumer's read uses it so a tiling gap is caught (else `WHOLE`).
    let mut slot_width: HashMap<u32, u32> = HashMap::new();
    for (instrs, tape_index) in segments {
        // Mirror `lower()`'s static loop unroll: `Loop(count, body_len)`
        // expands the body `count` times with `layer_offset = iter`. A
        // body instruction's `index` (= WeightLocator op_idx) is its
        // position in the slice, the SAME across iterations — per-layer
        // disambiguation rides on `layer + layer_offset`, not the index.
        let mut i = 0usize;
        while i < instrs.len() {
            match &instrs[i] {
                Instruction::Loop(count, body_len) => {
                    let body_start = i + 1;
                    let body_end = body_start
                        .checked_add(*body_len as usize)
                        .filter(|end| *end <= instrs.len())
                        .ok_or(CompileError::MalformedLoop {
                            index: i,
                            count: *count,
                            body_len: *body_len,
                        })?;
                    for iter in 0..*count as usize {
                        for (offset, inst) in instrs[body_start..body_end].iter().enumerate() {
                            process_instr::<W>(
                                &mut builder,
                                inst,
                                body_start + offset,
                                iter as u32,
                                *tape_index,
                                nb,
                                profile,
                                &mut splitk_scratch,
                                &mut moe_scratch,
                                &mut slot_width,
                            )?;
                        }
                    }
                    i = body_end;
                }
                other => {
                    process_instr::<W>(
                        &mut builder,
                        other,
                        i,
                        0,
                        *tape_index,
                        nb,
                        profile,
                        &mut splitk_scratch,
                        &mut moe_scratch,
                        &mut slot_width,
                    )?;
                    i += 1;
                }
            }
        }
    }
    let terminal = builder.buffer(BufferRef::ArenaSlot(terminal_slot), 2);
    Ok(builder.finish(terminal))
}

/// The recorded width of an arena slot's producer (defaults to `WHOLE`
/// when no precise width was recorded — i.e. a whole-op output).
fn rw(slot_width: &HashMap<u32, u32>, slot: u32) -> u32 {
    slot_width.get(&slot).copied().unwrap_or(WHOLE)
}

/// Build the typed [`OpDataflow`] for one (loop-unrolled) whole op from
/// its instruction's in/out arena slots. Read widths come from the
/// producer's recorded width (so a consumer reading a qmv output uses the
/// precise `n` → a tiling gap surfaces as partial-coverage); writes use
/// `WHOLE`. Errors (never silently skips) on an op with no model yet.
fn whole_op_dataflow(
    inst: &Instruction,
    builder: &mut SubtileIrBuilder,
    slot_width: &HashMap<u32, u32>,
) -> Result<OpDataflow, CompileError> {
    use Instruction as I;
    let rd = |bid: BufId, w: u32| RegionRef::rows_cols(bid, 1, 0, w);
    Ok(match inst {
        I::Embed(out) | I::AffineEmbed(out, ..) => {
            let o = builder.buffer(BufferRef::ArenaSlot(*out), 2);
            OpDataflow::Produce { out: rd(o, WHOLE) }
        }
        I::RmsNorm(inp, out, ..) | I::FusedGateUpSiluMul(inp, out, _) => {
            let xi = builder.buffer(BufferRef::ArenaSlot(*inp), 2);
            let oi = builder.buffer(BufferRef::ArenaSlot(*out), 2);
            OpDataflow::Map {
                x: rd(xi, rw(slot_width, *inp)),
                out: rd(oi, WHOLE),
            }
        }
        I::SiluMul(gate, up, out) => {
            let g = builder.buffer(BufferRef::ArenaSlot(*gate), 2);
            let u = builder.buffer(BufferRef::ArenaSlot(*up), 2);
            let o = builder.buffer(BufferRef::ArenaSlot(*out), 2);
            OpDataflow::Zip {
                a: rd(g, rw(slot_width, *gate)),
                b: rd(u, rw(slot_width, *up)),
                out: rd(o, WHOLE),
            }
        }
        I::Add(delta, residual) | I::FusedAddRmsNorm(delta, residual, ..) => {
            let d = builder.buffer(BufferRef::ArenaSlot(*delta), 2);
            let r = builder.buffer(BufferRef::ArenaSlot(*residual), 2);
            OpDataflow::AddInPlace {
                delta: rd(d, rw(slot_width, *delta)),
                residual: rd(r, rw(slot_width, *residual)),
            }
        }
        I::AttentionViaCache(q, out, ..) => {
            let qi = builder.buffer(BufferRef::ArenaSlot(*q), 2);
            let oi = builder.buffer(BufferRef::ArenaSlot(*out), 2);
            OpDataflow::Attn {
                x: rd(qi, rw(slot_width, *q)),
                out: rd(oi, WHOLE),
            }
        }
        // Synth megakernels: read `delta`, update `residual` in place, and
        // produce the slot the downstream qmv consumes (q_out / silu_out).
        I::SynthPreAttn(residual, delta, produced, ..)
        | I::SynthMlpPreDown(residual, delta, produced, ..) => {
            let r = builder.buffer(BufferRef::ArenaSlot(*residual), 2);
            let d = builder.buffer(BufferRef::ArenaSlot(*delta), 2);
            let p = builder.buffer(BufferRef::ArenaSlot(*produced), 2);
            OpDataflow::UpdateProduce {
                x: rd(d, rw(slot_width, *delta)),
                residual: rd(r, rw(slot_width, *residual)),
                out: rd(p, WHOLE),
            }
        }
        _ => return Err(CompileError::UnmodeledDataflow("(unmodeled whole op)")),
    })
}

/// Process one (already loop-unrolled) instruction: `AffineQmm` → native
/// N-block qmv subtiles, every other op → its whole-op dispatch (via
/// `lower_one`) carrying the typed [`OpDataflow`]. `index` is the
/// instruction's slice position; `layer_offset` its enclosing loop
/// iteration (0 outside a loop).
#[allow(clippy::too_many_arguments)]
fn process_instr<W: CanonicalParams>(
    builder: &mut SubtileIrBuilder,
    inst: &Instruction,
    index: usize,
    layer_offset: u32,
    tape_index: u32,
    nb: u32,
    profile: Option<&MetalTargetProfile>,
    splitk_scratch: &mut u32,
    moe_scratch: &mut u32,
    slot_width: &mut HashMap<u32, u32>,
) -> Result<(), CompileError> {
    match inst {
        Instruction::AffineQmm(in_slot, out_slot, layer, n, k, group_size, bits, _vl) => {
            compile_affine_qmm::<W>(
                builder,
                *in_slot,
                *out_slot,
                *layer,
                *n,
                *k,
                *group_size,
                *bits,
                layer_offset,
                tape_index,
                index as u32,
                nb,
                slot_width,
            );
            Ok(())
        }
        Instruction::Loop(..) => Err(CompileError::NestedLoop),
        other => {
            let cmds = lower_one::<W>(
                other,
                index,
                1,
                layer_offset,
                tape_index,
                splitk_scratch,
                moe_scratch,
                profile,
            )
            .map_err(CompileError::Lowering)?;
            if cmds.is_empty() {
                return Ok(()); // metadata-only (Reshape / Alias / Free) — no dispatch
            }
            let df = whole_op_dataflow(other, builder, slot_width)?;
            for cmd in &cmds {
                translate_whole(cmd, builder, df)?;
            }
            Ok(())
        }
    }
}
