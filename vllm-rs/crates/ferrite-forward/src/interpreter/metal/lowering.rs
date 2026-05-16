// SPDX-License-Identifier: Apache-2.0
//! Lowering pass: `&[Instruction]` → `LoweredMetalTape`.
//!
//! Pure CPU code. No Metal device required, no kernel dispatch — just
//! a structural translation that:
//!
//! 1. Statically unrolls `Loop(count, body_len)` over the loop body —
//!    each iteration's `iter` index is added to per-instruction
//!    `layer` literals at weight-binding time (Metal mirror of the
//!    cuda interpreter's `ctx.layer_offset = iter` pattern).
//! 2. Drops metadata-only instructions (`Reshape`, `Alias`, `Free`)
//!    that don't emit a Metal dispatch.
//! 3. For each compute instruction, picks the right `KernelId`,
//!    derives a `DispatchShape` from `bucket_m` + the kernel's
//!    convention, and produces `Binding`s pointing at arena slots,
//!    typed weight thunks, or runtime buffers.
//!
//! Coverage (post-Phase B):
//! `Embed`, `RmsNorm`, `FusedAddRmsNorm`, `Gemm`, `FusedGateUpSiluMul`,
//! `RopeAppend`, `AttentionViaCache`, `AttentionPrefillPaged`,
//! `Add`, `ScalarMul`, plus `Loop`/`Reshape`/`Alias`/`Free` as
//! structural ops. `AttentionPrefillContiguous` is matched only as an
//! `unreachable!` arm — Phase B's metal macro adapter
//! (`metal/attention.rs::fan_out`) emits `AttentionPrefillPaged` for
//! every metal model, so the contiguous variant should never reach
//! lowering. Every other variant raises
//! [`LoweringError::UnsupportedVariant`] with the variant's type name
//! so the model author knows which arm to add next.

use crate::{CanonicalParams, Instruction};
use ferrite_metal_kernels::quantized::{
    DequantDtype, QmmTKernel, QmvKernel, ScaleDtype, pick_qmm_t_kernel, pick_qmv_kernel,
    qmm_t_dispatch_shape, qmm_t_kernel_static_name, qmv_dispatch_shape, qmv_kernel_static_name,
    splitk_reduce_kernel_static_name,
};
use ferrite_metal_kernels::specialized_pipeline_cache::ConstantValue;

use super::lowered::{
    Binding, DispatchShape, GemmDims, KernelId, LoweredCommand, LoweredMetalTape, LoweringError,
    MetalDtype, RuntimeBindingKind, WeightBundleKind, WeightLocator, WeightTensor,
};

/// Lower one bucket's `(backbone ++ lm_head)` instruction stream.
///
/// Concatenation matches the cuda interpreter's effective behavior:
/// `forward()` runs backbone then lm_head in sequence for a given
/// bucket. Lowering them as one stream lets the worker bake both
/// halves into the bucket's single ICB plan, with no extra mid-bucket
/// boundary the caller needs to manage.
///
/// Avoids requiring `Instruction: Clone` — the caller hands two
/// `&[Instruction]` slices and we walk them in place, unrolling
/// `Loop` per the same rules as the single-slice [`lower`]. Loop
/// bodies that span the backbone/lm_head boundary are not supported
/// (no model emits one — the cuda lowering pass partitions loops
/// strictly inside one half), but if one ever shows up the malformed-
/// loop check fires inside the offending half and surfaces the
/// per-half index.
pub fn lower_pair<W: CanonicalParams>(
    backbone: &[Instruction],
    lm_head: &[Instruction],
    backbone_barriers: &[bool],
    lm_head_barriers: &[bool],
    bucket_m: u32,
    num_arena_slots: u32,
    backbone_tape_index: u32,
    lm_head_tape_index: u32,
    profile: Option<&ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile>,
) -> Result<LoweredMetalTape, LoweringError> {
    let bb = lower::<W>(backbone, backbone_barriers, bucket_m, num_arena_slots, backbone_tape_index, profile)?;
    let lh = lower::<W>(lm_head, lm_head_barriers, bucket_m, num_arena_slots, lm_head_tape_index, profile)?;
    let mut commands = bb.commands;
    let mut barrier_before = bb.barrier_before;

    // Sample-position slice — fast lm_head for prefill.
    //
    // The lm_head GEMM only needs the LAST token's hidden state per
    // sequence; the worker discards every other num_tokens-1 row of
    // logits host-side (`embedding_gather` keyed on
    // `last_token_indices`). At num_tokens=1024 + vocab=128256 +
    // K=hidden this is ~31× wasted GEMM work — ~325ms of the prefill
    // TTFT on M4 Llama-3.2-3B-4bit.
    //
    // To shrink it we wrap the lm_head AffineQmm with a pre-GEMM
    // gather and a post-GEMM scatter:
    //
    //   1. `GatherLastToken`     : in[0,:]  := in[num_tokens-1,:]
    //   2. lm_head AffineQmm     : m-tile dispatch overridden to 1 tile
    //                              (BM=32 rows of output computed; row
    //                              0 is the only one we care about)
    //   3. `ScatterFirstToLastRow`: out[num_tokens-1,:] := out[0,:]
    //
    // Post-step keeps the worker's downstream
    // `embedding_gather(logits, last_token_indices=[N-1])` correct
    // without it needing to know about the slice.
    //
    // Preconditions for applying the slice (else fall through to the
    // standard lm_head lowering):
    //   * `bucket_m > 1` — decode/batched-decode buckets need every
    //     row of logits intact.
    //   * `lm_head` is exactly one `Instruction::AffineQmm` (the
    //     common case across Llama / Qwen / Mistral). Tied
    //     embeddings and multi-step lm_heads fall through.
    //   * It lowered to exactly one `LoweredCommand` — i.e. Standard
    //     or Nax qmm_t, not SplitK (whose split partials and reduce
    //     would need their own row handling). At num_tokens=1024 the
    //     dispatcher always picks Standard, so this is the prefill
    //     path in practice.
    let slice_info = if std::env::var_os("FERRITE_METAL_LMHEAD_SLICE").is_some()
        && bucket_m > 1
        && lm_head.len() == 1
        && lh.commands.len() == 1
    {
        if let Instruction::AffineQmm(
            in_slot,
            out_slot,
            layer,
            n,
            k,
            group_size,
            bits,
            _vector_limit,
        ) = &lm_head[0]
        {
            if matches!(
                lh.commands[0].kernel,
                KernelId::AffineQmmT | KernelId::AffineQmmTNax
            ) {
                Some(LmHeadSliceInfo {
                    in_slot: *in_slot,
                    out_slot: *out_slot,
                    layer: *layer,
                    // The lm_head AffineQmm is the LAST instruction in
                    // the lm_head slice (lm_head.len() == 1 here), so
                    // its op_idx is 0 inside the lm_head tape.
                    locator: WeightLocator {
                        bucket: lm_head_tape_index,
                        op_idx: 0,
                        slot: 0,
                    },
                    n: *n,
                    k: *k,
                    group_size: *group_size,
                    bits: *bits,
                })
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    if let Some(info) = slice_info {
        // Pre-GEMM gather: row 0 of input := row num_tokens-1.
        commands.push(gather_last_token_command::<W>(info.in_slot, info.k));
        barrier_before.push(true);
        // lm_head at M=1 — route through qmv (matvec) instead of
        // qmm_t (matmul-with-shrunk-grid). qmm_t Standard has BM=32
        // so even with a 1-tile dispatch it does 32 m-rows of work;
        // qmv at M=1 is BW-bound on the weight read (~1.6 ms vs
        // ~12 ms for qmm_t-1-tile on Llama-3.2-3B's
        // [N=128256, K=3072] lm_head shape).
        commands.push(lm_head_qmv_command::<W>(&info, profile));
        barrier_before.push(*lh.barrier_before.first().unwrap_or(&true));
        // Post-GEMM scatter: row num_tokens-1 of output := row 0.
        // Keeps the worker's `embedding_gather(logits,
        // last_token_indices=[num_tokens-1])` correct without it
        // knowing the slice ran.
        commands.push(scatter_first_to_last_row_command::<W>(info.out_slot, info.n));
        barrier_before.push(true);
    } else {
        for (i, cmd) in lh.commands.into_iter().enumerate() {
            commands.push(cmd);
            barrier_before.push(*lh.barrier_before.get(i).unwrap_or(&true));
        }
    }

    Ok(LoweredMetalTape {
        bucket_m,
        num_arena_slots,
        commands,
        barrier_before,
        splitk_scratch_bytes: bb.splitk_scratch_bytes.max(lh.splitk_scratch_bytes),
    })
}

/// Source-Instruction params captured at `lm_head` AffineQmm so the
/// slice path can synthesize a qmv command (kernel substitute) plus
/// the matching gather/scatter framing. Lives only inside `lower_pair`.
struct LmHeadSliceInfo {
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    /// Tape locator for the lm_head LinearLayer accessor — the worker
    /// resolves through `WeightAccessors::linear_at`.
    locator: WeightLocator,
    n: u32,
    k: u32,
    group_size: u32,
    bits: u32,
}

/// Lower the lm_head AffineQmm at M=1 through a qmv matvec kernel
/// instead of the BM=32 qmm_t tile. Saves ~10 ms TTFT on Llama-3.2-3B
/// lm_head (qmm_t-1-tile = 32 wasted m-rows × 4008 n-tiles, qmv at
/// M=1 is single-pass BW-bound on the 197 MB packed weight read).
fn lm_head_qmv_command<W: CanonicalParams>(
    info: &LmHeadSliceInfo,
    profile: Option<&ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile>,
) -> LoweredCommand {
    let dtype = dequant_dtype_for::<W>();
    let scale_dtype = scale_dtype_for::<W>();
    let n_v = info.n;
    let k_v = info.k;
    let bits_v = info.bits;
    let gs = info.group_size;

    let kernel = match profile {
        Some(p) => ferrite_metal_kernels::quantized::pick_qmv_kernel_by_cost(
            |name, mm, nn, kk| p.cost_us_for(name, mm, nn, kk),
            n_v,
            k_v,
            bits_v,
            gs,
            dtype,
        ),
        None => pick_qmv_kernel(n_v, k_v, bits_v),
    };
    let (tg, tpg) = qmv_dispatch_shape(kernel, /* M = */ 1, n_v, /* B = */ 1);
    let kernel_id = match kernel {
        QmvKernel::Quad { .. } => KernelId::AffineQmvQuad,
        QmvKernel::Fast => KernelId::AffineQmvFast,
        QmvKernel::Generic => KernelId::AffineQmv,
    };

    LoweredCommand {
        kernel: kernel_id,
        library: "quantized_qmv",
        function: qmv_kernel_static_name(kernel, dtype, scale_dtype, bits_v, gs),
        constants: super::kernel_constants::AffineQmvConstants {
            k: super::ids::KDimI32(k_v as i32),
            n: super::ids::NDimI32(n_v as i32),
        }
        .into(),
        dispatch: DispatchShape {
            threadgroups: tg,
            threads_per_threadgroup: tpg,
            // Fixed M=1 — no scaling.
            m_scaling: None,
        },
        bindings: affine_qmm_bindings(
            info.in_slot,
            info.out_slot,
            super::ids::LayerId(info.layer),
            info.locator,
        ),
        gemm_dims: None,
    }
}

fn gather_last_token_command<W: CanonicalParams>(
    slot: u32,
    hidden_size: u32,
) -> LoweredCommand {
    sample_slice_command::<W>(
        slot,
        hidden_size,
        KernelId::GatherLastToken,
        "gather_last_token_f16_specialized",
        "gather_last_token_bf16_specialized",
    )
}

fn scatter_first_to_last_row_command<W: CanonicalParams>(
    slot: u32,
    vocab_size: u32,
) -> LoweredCommand {
    sample_slice_command::<W>(
        slot,
        vocab_size,
        KernelId::ScatterFirstToLastRow,
        "scatter_first_to_last_row_f16_specialized",
        "scatter_first_to_last_row_bf16_specialized",
    )
}

fn sample_slice_command<W: CanonicalParams>(
    slot: u32,
    row_stride: u32,
    kernel: KernelId,
    f16_symbol: &'static str,
    bf16_symbol: &'static str,
) -> LoweredCommand {
    const THREADS_PER_TG: u32 = 256;
    LoweredCommand {
        kernel,
        library: "gather_last_token",
        function: pick_specialized_symbol(f16_symbol, bf16_symbol, W::METAL_DTYPE),
        constants: super::kernel_constants::GatherLastTokenConstants {
            row_stride: super::ids::HiddenSize(row_stride),
        }
        .into(),
        dispatch: DispatchShape {
            threadgroups: (row_stride.div_ceil(THREADS_PER_TG), 1, 1),
            threads_per_threadgroup: (THREADS_PER_TG, 1, 1),
            m_scaling: None,
        },
        bindings: vec![
            Binding::ArenaSlot {
                slot,
                binding_index: 0,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::NumTokensU32,
                binding_index: 1,
            },
        ],
        gemm_dims: None,
    }
}

/// Lower one bucket's `Instruction` tape.
///
/// `bucket_m` is the bucket point this tape is specialized for —
/// the worker's `forward()` only routes batches with `num_tokens`
/// matching this bucket's `[m_min, m_max_excl)` range here.
/// Dispatch shapes that depend on `M` (token-parallel kernels, GEMM
/// outer dim) are computed against `bucket_m`.
///
/// `num_arena_slots` is the colored slot count from
/// `colored_slot_map()` (in `ferrite-forward-macro`). The lowered
/// tape carries it through verbatim — the worker uses it to size its
/// per-shape-class arena.
pub fn lower<W: CanonicalParams>(
    instructions: &[Instruction],
    barriers_in: &[bool],
    bucket_m: u32,
    num_arena_slots: u32,
    tape_index: u32,
    profile: Option<&ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile>,
) -> Result<LoweredMetalTape, LoweringError> {
    let mut commands = Vec::with_capacity(instructions.len());
    let mut barrier_before: Vec<bool> = Vec::with_capacity(instructions.len());
    let mut splitk_scratch_bytes: u32 = 0;
    let mut i = 0usize;
    // Macro-static-aligned barrier accessor: the macro emits one bool
    // per `Instruction` (pre-loop-unrolling). Loop expansion at
    // runtime reuses the body's flags across every iteration —
    // body byte-equivalence (the precondition for the macro's loop
    // compression) guarantees the same hazard signature per iter.
    // For metadata-only instructions (Reshape/Alias/Free), no
    // LoweredCommand is emitted so the flag is unused. For the
    // SplitK matmul pair (2 LoweredCommands from 1 Instruction),
    // the first emit takes the macro flag; the second emit gets
    // `true` (intra-Instruction scratch RAW).
    let flag_for = |idx: usize| -> bool { barriers_in.get(idx).copied().unwrap_or(true) };

    while i < instructions.len() {
        match &instructions[i] {
            Instruction::Loop(count, body_len) => {
                let count_usize = *count as usize;
                let body_len_usize = *body_len as usize;
                let body_start = i + 1;
                let body_end = body_start
                    .checked_add(body_len_usize)
                    .filter(|end| *end <= instructions.len())
                    .ok_or_else(|| LoweringError::MalformedLoop {
                        index: i,
                        count: *count,
                        body_len: *body_len,
                        remaining: instructions.len().saturating_sub(body_start),
                    })?;
                let body = &instructions[body_start..body_end];
                // Static unroll. Mirrors the cuda interpreter's
                // `ctx.layer_offset = iter` pattern (`instr.rs:689` and
                // friends): each variant's compile-time `layer`
                // literal is added to the iteration index at WtFn
                // resolution time, so `Loop(22, body)` over a body that
                // names `layer = 0` resolves layer 0..21 across the 22
                // iterations. Pass `iter as u32` into `lower_one` so
                // the weight lookups bake the right per-layer offset.
                for iter in 0..count_usize {
                    for (offset, inst) in body.iter().enumerate() {
                        let cmds = lower_one::<W>(
                            inst,
                            body_start + offset,
                            bucket_m,
                            iter as u32,
                            tape_index,
                            &mut splitk_scratch_bytes,
                            profile,
                        )?;
                        let n_cmds = cmds.len();
                        commands.extend(cmds);
                        if n_cmds >= 1 {
                            barrier_before.push(flag_for(body_start + offset));
                            for _ in 1..n_cmds {
                                barrier_before.push(true);
                            }
                        }
                    }
                }
                i = body_end;
            }
            other => {
                let cmds = lower_one::<W>(other, i, bucket_m, 0, tape_index, &mut splitk_scratch_bytes, profile)?;
                let n_cmds = cmds.len();
                commands.extend(cmds);
                if n_cmds >= 1 {
                    barrier_before.push(flag_for(i));
                    for _ in 1..n_cmds {
                        barrier_before.push(true);
                    }
                }
                i += 1;
            }
        }
    }

    debug_assert_eq!(commands.len(), barrier_before.len());
    Ok(LoweredMetalTape {
        bucket_m,
        num_arena_slots,
        commands,
        barrier_before,
        splitk_scratch_bytes,
    })
}

/// Lower one non-`Loop` instruction to zero, one, or many
/// `LoweredCommand`s. Most instructions produce exactly one; metadata-
/// only instructions (`Reshape`/`Alias`/`Free`) produce zero;
/// `Instruction::AffineQmm` in the matmul branch produces two when the
/// dispatcher picks `QmmTKernel::SplitK` (the `affine_qmm_t_splitk`
/// kernel writes a `[split_k, M, N]` partial into a shared scratch
/// buffer, then `splitk_reduce_sum` reduces it into the AffineQmm's
/// arena slot).
///
/// `layer_offset` is the enclosing `Loop`'s iteration index (0 for
/// straight-line code). Combined with each variant's compile-time
/// `layer` literal at weight-binding time, this matches the cuda
/// interpreter's `let layer = ctx.layer_offset + layer;` pattern in
/// `instr.rs`. Without this, every iteration of a `Loop(N, body)`
/// emits identical commands and every per-layer weight lookup
/// resolves to layer 0 — which wedges the model on layer 0's norms,
/// projections, RoPE caches, and KV-cache slots, producing nonsense
/// logits at the end of the chain.
///
/// `splitk_scratch_bytes` is the running max of `split_k * M * N *
/// elem_size_bytes(dtype)` across every `AffineQmm` in this tape that
/// picked `SplitK`. The worker uses it to size the shared scratch
/// buffer that `Binding::Scratch` resolves against.
fn lower_one<W: CanonicalParams>(
    inst: &Instruction,
    index: usize,
    bucket_m: u32,
    layer_offset: u32,
    tape_index: u32,
    splitk_scratch_bytes: &mut u32,
    profile: Option<&ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile>,
) -> Result<Vec<LoweredCommand>, LoweringError> {
    use Instruction as I;

    let cmd = match inst {
        // ── Token embedding ────────────────────────────────────────
        I::Embed(out_slot) => LoweredCommand {
            kernel: KernelId::Embed,
            library: "embed",
            function: pick_specialized_symbol(
                "embed_f16_specialized",
                "embed_bf16_specialized",
                W::METAL_DTYPE,
            ),
            constants: super::kernel_constants::EmbedConstants {
                bucket_m: super::ids::BucketM(bucket_m),
                q_size: super::ids::QSize(W::Q_SIZE as u32),
            }
            .into(),
            // 1D dispatch over the `bucket_m` tokens; one thread per
            // token gathers a row from `embed_tokens.weight`. Scales
            // proportionally with actual M at dispatch time.
            dispatch: {
                let mut d = DispatchShape::dispatch_1d(bucket_m, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) });
                d
            },
            bindings: vec![
                // out: arena[out_slot]
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                // weight: embed_tokens.weight (layer 0; Embed is not layered)
                Binding::Weight {
                    kind: WeightBundleKind::Embedding,
                    which: WeightTensor::Weight,
                    layer: super::ids::LayerId(0),
                    locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                    binding_index: 1,
                },
                // input_ids
                Binding::Runtime {
                    kind: RuntimeBindingKind::InputIds,
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        },

        // ── Standalone RMSNorm ─────────────────────────────────────
        // Macro/cuda emit `Instruction::RmsNorm(in_slot, out_slot, ...)`
        // (see `instr.rs:689`). An earlier `(out_slot, in_slot, ...)`
        // pattern here silently swapped the names — the kernel read
        // from a fresh slot and overwrote the upstream tile.
        I::RmsNorm(in_slot, out_slot, layer) => LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            // Symbol names are `rmsnorm_<T_act>_s_<T_scale>_specialized`
            // — the in-register T_scale cast P10c added so RMSNorm
            // gains stay F16 on device (`feedback_no_silent_deferrals`,
            // mirrors P10b for quant scales). `scale_dtype_for::<W>()`
            // returns F16 today; expand the picker when a model ships
            // bf16 norm gains on disk.
            function: rmsnorm_kernel_static_name::<W>(scale_dtype_for::<W>()),
            constants: super::kernel_constants::RmsNormConstants {
                bucket_m: super::ids::BucketM(bucket_m),
                q_size: super::ids::QSize(W::Q_SIZE as u32),
                rms_norm_eps: super::ids::RmsNormEps(W::RMS_NORM_EPS),
            }
            .into(),
            // Per-token threadgroup; threads cooperate on the
            // hidden-size reduction inside.
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm,
                    which: WeightTensor::Weight,
                    layer: super::ids::LayerId(*layer + layer_offset),
                    locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        },

        // ── Fused residual-add + RMSNorm ───────────────────────────
        I::FusedAddRmsNorm(delta_slot, residual_slot, layer) => {
            LoweredCommand {
                kernel: KernelId::FusedAddRmsNorm,
                library: "fused_add_rmsnorm",
                // Symbol `fused_add_rmsnorm_<T_act>_s_<T_scale>_specialized`
                // (P10c — see RmsNorm comment above).
                function: fused_add_rmsnorm_kernel_static_name::<W>(
                    scale_dtype_for::<W>(),
                ),
                constants: super::kernel_constants::RmsNormConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                    q_size: super::ids::QSize(W::Q_SIZE as u32),
                    rms_norm_eps: super::ids::RmsNormEps(W::RMS_NORM_EPS),
                }
                .into(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, 1, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: vec![
                    // residual: read+write (in-place add target,
                    // norm-input source)
                    Binding::ArenaSlot {
                        slot: *residual_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *delta_slot,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 2,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── Generic dense GEMM ─────────────────────────────────────
        I::Gemm(in_slot, out_slot, layer, n, k) => {
            // Dispatch shape is a no-op for MPS (the worker reads
            // `gemm_dims` and calls MPSMatrixMultiplication), but a
            // hypothetical hand-rolled GEMM tile shader could still
            // consume it. Tile values are placeholders; the actual
            // execution path picks its own.
            let tg_x = bucket_m.div_ceil(GEMM_TILE_M);
            let tg_y = (*n).div_ceil(GEMM_TILE_N);
            LoweredCommand {
                kernel: KernelId::Gemm,
                // GEMM is opaque to `pipeline_for_command` — f16 takes the
                // MPS branch, bf16 has its own dims-keyed builder
                // (`pipeline_for_gemm_bf16`). Empty library/function +
                // empty constants signal the worker to route GEMM commands
                // through the special-case path instead.
                library: "",
                function: "",
                constants: Vec::new(),
                dispatch: DispatchShape {
                    threadgroups: (tg_x, tg_y, 1),
                    threads_per_threadgroup: (GEMM_TILE_M, GEMM_TILE_N, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 2,
                    },
                ],
                gemm_dims: Some(GemmDims {
                    m: bucket_m,
                    n: *n,
                    k: *k,
                }),
            }
        }

        // ── MLX-affine int4 matmul (qmv + qmm_t) ──────────────────
        //
        // `Instruction::AffineQmm` covers every per-Linear shape on a
        // metal-quantized model. The dispatcher rule mirrors MLX
        // `quantized.cpp:1387 QuantizedMatmul::eval_gpu`:
        //   * `bucket_m < vector_limit` → matvec (qmv_quad / qmv_fast /
        //     qmv) per `dispatch_qmv` (`:1365`) — D∈{64,128} pow2-bits
        //     wins quad, then N%8==0 ∧ K%512==0 wins fast, else generic.
        //   * `bucket_m ≥ vector_limit` → matmul (qmm_t for transpose=true).
        //     SplitK is a sibling of qmm_t Standard for B==1; lands in
        //     C3 alongside its downstream sum-reduce.
        //
        // `vector_limit` rides on the Instruction so the macro can bake
        // it from `get_qmv_batch_limit(K, N, arch_gen)` — see the
        // declaration on `Instruction::AffineQmm`.
        //
        // Bindings match the kernel signatures in `quantized_qmv.metal`
        // and `quantized_qmm.metal`:
        //   buffer(0) packed weight   buffer(1) scales   buffer(2) biases
        //   buffer(3) x activations   buffer(4) y output
        // K / N (and M for qmm_t) ride as function constants 0/1(/2)
        // post the C1 refactor; the dispatcher never sets them as
        // setBytes — required for ICB recording.
        I::AffineQmm(in_slot, out_slot, layer, n, k, group_size, bits, vector_limit) => {
            let dtype = dequant_dtype_for::<W>();
            let scale_dtype = scale_dtype_for::<W>();
            let n_v = *n;
            let k_v = *k;
            let bits_v = *bits;
            let gs = *group_size;
            let vl = *vector_limit;

            if bucket_m < vl {
                // Matvec branch (decode-shape). Cost-driven pick
                // when the target profile is available — walks every
                // valid qmv variant for `(n, k, bits)` and picks min
                // cost_us from the profile's CSV. Falls back to the
                // MLX-mirrored heuristic when no profile (uncalibrated
                // chip).
                let kernel = match profile {
                    Some(p) => ferrite_metal_kernels::quantized::pick_qmv_kernel_by_cost(
                        |name, mm, nn, kk| p.cost_us_for(name, mm, nn, kk),
                        n_v,
                        k_v,
                        bits_v,
                        gs,
                        dtype,
                    ),
                    None => pick_qmv_kernel(n_v, k_v, bits_v),
                };
                let (tg, tpg) = qmv_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                let kernel_id = match kernel {
                    QmvKernel::Quad { .. } => KernelId::AffineQmvQuad,
                    QmvKernel::Fast => KernelId::AffineQmvFast,
                    QmvKernel::Generic => KernelId::AffineQmv,
                };
                LoweredCommand {
                    kernel: kernel_id,
                    library: "quantized_qmv",
                    function: qmv_kernel_static_name(kernel, dtype, scale_dtype, bits_v, gs),
                    constants: super::kernel_constants::AffineQmvConstants {
                        k: super::ids::KDimI32(k_v as i32),
                        n: super::ids::NDimI32(n_v as i32),
                    }
                    .into(),
                    dispatch: DispatchShape {
                        threadgroups: tg,
                        threads_per_threadgroup: tpg,
                        // qmv grid x-axis == M directly
                        // (`qmv_dispatch_shape` returns `(m, ceil(N/bn), B)`);
                        // shrinks linearly with actual num_tokens.
                        m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                    },
                    bindings: affine_qmm_bindings(
                        *in_slot,
                        *out_slot,
                        super::ids::LayerId(*layer + layer_offset),
                        WeightLocator {
                            bucket: tape_index,
                            op_idx: index as u32,
                            slot: 0,
                        },
                    ),
                    gemm_dims: None,
                }
            } else {
                // Matmul branch (prefill-shape). `pick_qmm_t_kernel`
                // mirrors MLX `quantized.cpp:1411-1424 + :788-805`:
                //   * `Standard` when split_k ≤ 1 (target ~512 tgs).
                //   * `SplitK` when n_tiles × m_tiles is sparse enough
                //     that splitting K into `split_k` partitions pushes
                //     the total threadgroup count up to roughly 512;
                //     fed by the `splitk_reduce_sum` kernel that
                //     collapses the `[split_k, M, N]` partial to
                //     `[M, N]` in the AffineQmm's out slot.
                // NAX (M5+/A19+ only — MLX gates on arch_gen >= 17, see
                // `mlx/backend/metal/device.cpp:828`). On M4 and earlier
                // MPP `matmul2d` emulates via standard simdgroup matmul
                // with a cooperative-tensor per-thread layout that does
                // NOT match MLX `BaseNAXFrag`'s 2-row × 4-col assumption,
                // so the kernel produces garbage. `is_nax_capable` is
                // currently `false` for every gen we model; add an M5
                // variant + run the `nax_probe_dump_layout` diagnostic
                // before flipping it true. See `memory/
                // project_metal_nax_layout_bug.md` and PROGRESS.md.
                let is_nax = profile.is_some_and(|p| {
                    ferrite_metal_kernels::ferrite_metal_targets::is_nax_capable(p.generation)
                });
                let kernel = pick_qmm_t_kernel(bucket_m, n_v, k_v, /*B=*/ 1, gs, is_nax);
                // NAX tile is 64×64 so align check uses 64; Standard/SplitK use 32.
                let aligned_n = match kernel {
                    QmmTKernel::Nax => n_v.is_multiple_of(64),
                    _ => n_v.is_multiple_of(32),
                };
                match kernel {
                    QmmTKernel::Nax => {
                        let (tg, tpg) =
                            qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                        LoweredCommand {
                            kernel: KernelId::AffineQmmTNax,
                            library: "quantized_qmm_nax",
                            function: qmm_t_kernel_static_name(
                                kernel, dtype, scale_dtype, bits_v, gs, aligned_n,
                            ),
                            constants: super::kernel_constants::AffineQmmTConstants {
                                k: super::ids::KDimI32(k_v as i32),
                                n: super::ids::NDimI32(n_v as i32),
                                m: super::ids::MDimI32(bucket_m as i32),
                            }
                            .into(),
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                                // qmm_t NAX grid = (n_tiles, m_tiles=ceil(M/64), B)
                                m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::Y, bucket_m: super::ids::BucketM(bucket_m) }),
                            },
                            bindings: affine_qmm_bindings(
                                *in_slot,
                                *out_slot,
                                super::ids::LayerId(*layer + layer_offset),
                                WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                            ),
                            gemm_dims: None,
                        }
                    }
                    QmmTKernel::Standard => {
                        let (tg, tpg) =
                            qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                        LoweredCommand {
                            kernel: KernelId::AffineQmmT,
                            library: "quantized_qmm",
                            function: qmm_t_kernel_static_name(
                                kernel, dtype, scale_dtype, bits_v, gs, aligned_n,
                            ),
                            constants: super::kernel_constants::AffineQmmTConstants {
                                k: super::ids::KDimI32(k_v as i32),
                                n: super::ids::NDimI32(n_v as i32),
                                m: super::ids::MDimI32(bucket_m as i32),
                            }
                            .into(),
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                                // qmm_t Standard grid = (n_tiles, m_tiles=ceil(M/32), B)
                                m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::Y, bucket_m: super::ids::BucketM(bucket_m) }),
                            },
                            bindings: affine_qmm_bindings(
                                *in_slot,
                                *out_slot,
                                super::ids::LayerId(*layer + layer_offset),
                                WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                            ),
                            gemm_dims: None,
                        }
                    }
                    QmmTKernel::SplitK { split_k, k_partition_size } => {
                        // Two commands:
                        //   (1) qmm_t_splitk writes the `[split_k, M, N]`
                        //       partial into `Binding::Scratch`.
                        //   (2) splitk_reduce_sum reads scratch and
                        //       reduces along axis 0 into the AffineQmm's
                        //       arena slot.
                        let elem_bytes = elem_size_bytes(dtype);
                        let scratch_bytes =
                            split_k.saturating_mul(bucket_m).saturating_mul(n_v).saturating_mul(elem_bytes);
                        *splitk_scratch_bytes = (*splitk_scratch_bytes).max(scratch_bytes);

                        let (tg, tpg) =
                            qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                        let qmm_t_cmd = LoweredCommand {
                            kernel: KernelId::AffineQmmTSplitK,
                            library: "quantized_qmm",
                            function: qmm_t_kernel_static_name(
                                kernel, dtype, scale_dtype, bits_v, gs, aligned_n,
                            ),
                            // SplitK needs FOUR function constants:
                            // (0=K, 1=N, 2=M, 3=k_partition_size) per
                            // `quantized_qmm.metal:80-83`. The
                            // standalone `MetalAffineQmmT::execute`
                            // (`quantized.rs:776-781`) emits the same
                            // four; missing `k_partition_size` (slot 3)
                            // leaves the partition stride undefined and
                            // every layer's prefill output is garbage.
                            constants: super::kernel_constants::AffineQmmTSplitKConstants {
                                k: super::ids::KDimI32(k_v as i32),
                                n: super::ids::NDimI32(n_v as i32),
                                m: super::ids::MDimI32(bucket_m as i32),
                                k_partition_size: super::ids::KPartitionSizeI32(
                                    k_partition_size as i32,
                                ),
                            }
                            .into(),
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                                // qmm_t SplitK grid = (n_tiles, m_tiles=ceil(M/32), split_k)
                                m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::Y, bucket_m: super::ids::BucketM(bucket_m) }),
                            },
                            bindings: affine_qmm_splitk_bindings(
                                *in_slot,
                                super::ids::LayerId(*layer + layer_offset),
                                WeightLocator {
                                    bucket: tape_index,
                                    op_idx: index as u32,
                                    slot: 0,
                                },
                            ),
                            gemm_dims: None,
                        };

                        // splitk_reduce_sum: bindings (0=output → out_slot,
                        // 1=intermediate → Scratch), function constants
                        // (0=M, 1=N, 2=split_k), 1D dispatch over M*N
                        // output elements.
                        let nthreads = bucket_m.saturating_mul(n_v);
                        let reduce_cmd = LoweredCommand {
                            kernel: KernelId::SplitKReduceSum,
                            library: "quantized_splitk_reduce",
                            function: splitk_reduce_kernel_static_name(dtype),
                            constants: super::kernel_constants::SplitKReduceSumConstants {
                                bucket_m: super::ids::BucketM(bucket_m),
                                n: super::ids::NDim(n_v),
                                split_k: super::ids::SplitK(split_k),
                            }
                            .into(),
                            dispatch: {
                                let mut d = DispatchShape::dispatch_1d(nthreads, THREADS_PER_GROUP);
                                // groups = ceil(bucket_m * n_v / TPG) is linear
                                // in M; proportional scaling shrinks it for
                                // actual num_tokens.
                                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) });
                                d
                            },
                            bindings: vec![
                                Binding::ArenaSlot {
                                    slot: *out_slot,
                                    binding_index: 0,
                                },
                                Binding::Scratch { binding_index: 1 },
                            ],
                            gemm_dims: None,
                        };
                        return Ok(vec![qmm_t_cmd, reduce_cmd]);
                    }
                }
            }
        }

        // ── Fused silu(gate) * up for the decomposed q-MLP path ───
        //
        // C4 will start emitting `(AffineQmm gate, AffineQmm up,
        // SiluMul)` from the macro when both gate_proj and up_proj
        // are MLX-affine quantized — `MetalFusedGateUpSiluMulImpl`
        // currently rejects non-Dense storage. SiluMul is the
        // elementwise tail of that decomposition; the gate / up
        // arena slots hold the two AffineQmm outputs and SiluMul
        // writes `silu(gate) * up` into out_slot.
        I::SiluMul(gate_slot, up_slot, out_slot) => {
            let dtype = dequant_dtype_for::<W>();
            let n = bucket_m * (W::INTERMEDIATE_SIZE as u32);
            LoweredCommand {
                kernel: KernelId::SiluMul,
                library: "silu_mul",
                function: silu_mul_static_name(dtype),
                constants: super::kernel_constants::SiluMulConstants {
                    n: super::ids::HiddenSize(n),
                }
                .into(),
                // 1D dispatch over M * intermediate_size output elements,
                // one thread per element. Threadgroup width clamped to
                // the pipeline's max at execute time would be cleaner;
                // for now match the elementwise convention used by
                // `KernelId::Add` / `KernelId::ScalarMul`. Scales
                // proportionally with M at dispatch time.
                dispatch: {
                    let mut d = DispatchShape::dispatch_1d(n, THREADS_PER_GROUP);
                    d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) });
                    d
                },
                bindings: vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *gate_slot,
                        binding_index: 1,
                    },
                    Binding::ArenaSlot {
                        slot: *up_slot,
                        binding_index: 2,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── MLX-affine int4 quantized embedding (P6) ─────────────
        //
        // `Instruction::AffineEmbed` replaces `Instruction::Embed`
        // when `model.embed_tokens` ships as a quantized triple
        // `(weight=U32, scales, biases)` — i.e. every
        // `mlx-community/*-4bit` checkpoint. The fused
        // `affine_embed_<dtype>_gs_<gs>_b_4` kernel reads
        // `vocab_idx = indices[token_row]` then dequants from row
        // `vocab_idx` of the packed weight + scales + biases in one
        // pass (faithful port of `nn.QuantizedEmbedding.__call__`).
        //
        // 2D dispatch:
        //   threadgroups = (ceil((Q_SIZE/2) / 256), bucket_m, 1)
        //   threads_per_threadgroup = (256, 1, 1)
        // The kernel bounds-checks `index.x * 2 >= hidden_size`,
        // which keeps the partial trailing threadgroup safe when
        // Q_SIZE/2 is not a multiple of 256 (Llama-3.2-3B's
        // Q_SIZE=3072 → bytes_per_row=1536 = 6 × 256, clean; Qwen2
        // 1.5B's Q_SIZE=1536 → 768 = 3 × 256, clean; but
        // e.g. Q_SIZE=2048 → 1024 = 4 × 256, no partial threads —
        // pessimistically still safe).
        //
        // Bindings (mirror `affine_qmm_bindings` ordering for
        // consistency with the standalone `MetalAffineEmbed::execute`):
        //   buffer(0) packed weight   buffer(1) scales   buffer(2) biases
        //   buffer(3) input_ids       buffer(4) out (arena[out_slot])
        // hidden_size rides as `[[function_constant(0)]]`.
        #[cfg(feature = "metal")]
        I::AffineEmbed(out_slot, group_size, bits) => {
            let dtype = dequant_dtype_for::<W>();
            let scale_dtype = scale_dtype_for::<W>();
            let bits_v = *bits;
            let gs = *group_size;
            assert_eq!(
                bits_v, 4,
                "AffineEmbed: only bits=4 is wired in P6 (every sampled \
                 mlx-community 4bit checkpoint uses bits=4; \
                 INT4_PARITY_PROBES.md §1); got bits={bits_v}"
            );
            assert!(
                matches!(gs, 32 | 64 | 128),
                "AffineEmbed: only group_size ∈ {{32, 64, 128}} is wired \
                 (mlx-community uses gs=64 for every Llama/Qwen/Gemma 4bit; \
                 INT4_PARITY_PROBES.md §3); got gs={gs}"
            );
            let hidden_size = W::Q_SIZE as u32;
            let bytes_per_row = hidden_size / 2;
            let groups_x = bytes_per_row.div_ceil(THREADS_PER_GROUP);
            LoweredCommand {
                kernel: KernelId::AffineEmbed,
                library: "quantized_dequantize",
                function: affine_embed_kernel_static_name(dtype, scale_dtype, gs),
                constants: super::kernel_constants::AffineEmbedConstants {
                    hidden_size: super::ids::HiddenSize(hidden_size),
                }
                .into(),
                dispatch: DispatchShape {
                    threadgroups: (groups_x, bucket_m, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::Y, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: vec![
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(0),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 0,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(0),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(0),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 2,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::InputIds,
                        binding_index: 3,
                    },
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 4,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── Fused gate-up SwiGLU MLP (GEMM + SwiGLU in one dispatch) ──
        //
        // Two specialized variants behind `KernelId::FusedGateUpSiluMul`:
        //
        // - **Decode (`bucket_m == 1`)**: dispatches
        //   `fused_gate_up_silu_mul_decode_*_specialized`, which uses
        //   simd_sum dot products. 256 threads/group = 8 simdgroups
        //   × 32 lanes; one simdgroup owns one output. Threadgroup
        //   grid: `(ceil(N/4), 1, 1)` — `MLP_DECODE_BLOCK_M = 4`
        //   outputs per group.
        //
        // - **Prefill (`bucket_m >= 2`)**: dispatches
        //   `fused_gate_up_silu_mul_gemm_steel_*_specialized` (MLX-
        //   steel pattern, BM=BN=32, BK=16, WM=WN=2). 128
        //   threads/group = 4 simdgroups; each simdgroup carries
        //   2×2 = 4 `simdgroup_*8x8` accumulator frags per gate / up.
        //   Threadgroup grid: `(ceil(N/32), ceil(M/32), 1)`.
        //
        // Bindings are identical for both variants: (out, in, weight)
        // at indices 0/1/2. The kernel splits the packed `[gate|up]`
        // weight `[2*N, K]` internally — gate rows [0, N), up rows
        // [N, 2N). The decode-vs-steel symbol pick happens inline
        // below against `bucket_m`.
        I::FusedGateUpSiluMul(in_slot, out_slot, layer) => {
            let inter = W::INTERMEDIATE_SIZE as u32;
            let (threadgroups, threads_per_threadgroup) = if bucket_m == 1 {
                // Decode (MLX gemv port): blockM = BM*SM*TM = 4
                // outputs per threadgroup, 256 threads/group =
                // BN*SN = 8 simdgroups × 32 lanes.
                ((inter.div_ceil(MLP_DECODE_BLOCK_M), 1, 1), (256, 1, 1))
            } else {
                // Prefill (MLX-steel matrix variant): 32×32 output
                // tile, 128 threads = 4 simdgroups × 32 lanes.
                let tg_x = inter.div_ceil(MLP_STEEL_TILE);
                let tg_y = bucket_m.div_ceil(MLP_STEEL_TILE);
                ((tg_x, tg_y, 1), (MLP_STEEL_THREADS, 1, 1))
            };
            // Two specialized variants share `fused_gate_up_silu_mul.metallib`;
            // each binds a distinct constant-slot triple to avoid clashing
            // when the library is loaded:
            //   3/4/5 → M=1 decode kernel (gemv with simd_sum)
            //   6/7/8 → MLX-steel matrix kernel (production for M >= 2)
            // (slots 0/1/2 keyed the retired legacy 8×8 kernel.)
            let function = if bucket_m == 1 {
                pick_specialized_symbol(
                    "fused_gate_up_silu_mul_decode_f16_specialized",
                    "fused_gate_up_silu_mul_decode_bf16_specialized",
                    W::METAL_DTYPE,
                )
            } else {
                pick_specialized_symbol(
                    "fused_gate_up_silu_mul_gemm_steel_f16_specialized",
                    "fused_gate_up_silu_mul_gemm_steel_bf16_specialized",
                    W::METAL_DTYPE,
                )
            };
            let constants: Vec<ConstantValue> = if bucket_m == 1 {
                super::kernel_constants::FusedGateUpSiluMulDecodeConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                    intermediate_size: super::ids::IntermediateSize(W::INTERMEDIATE_SIZE as u32),
                    q_size: super::ids::QSize(W::Q_SIZE as u32),
                }
                .into()
            } else {
                super::kernel_constants::FusedGateUpSiluMulPrefillConstants {
                    bucket_m: super::ids::BucketM(bucket_m),
                    intermediate_size: super::ids::IntermediateSize(W::INTERMEDIATE_SIZE as u32),
                    q_size: super::ids::QSize(W::Q_SIZE as u32),
                }
                .into()
            };
            LoweredCommand {
                kernel: KernelId::FusedGateUpSiluMul,
                library: "fused_gate_up_silu_mul",
                function,
                constants,
                dispatch: DispatchShape {
                    threadgroups,
                    threads_per_threadgroup,
                    // Decode branch is bucket_m == 1 (M never grows);
                    // prefill branch baselines on y-axis as
                    // `bucket_m.div_ceil(MLP_STEEL_TILE)` — proportional
                    // scaling shrinks it to the actual M.
                    m_scaling: if bucket_m == 1 {
                        None
                    } else {
                        Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::Y, bucket_m: super::ids::BucketM(bucket_m) })
                    },
                },
                bindings: vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 2,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── RoPE + KV cache append ─────────────────────────────────
        I::RopeAppend(
            _q_slot,
            _k_slot,
            _v_slot,
            q_out_slot,
            k_out_slot,
            v_out_slot,
            layer,
            _interleaved,
        ) => {
            // 2D dispatch: (M, num_heads) — one threadgroup per
            // (token, head) pair rotates the head's `head_dim` slice
            // and writes K/V to the layer's paged cache page.
            let n_q_heads = W::NUM_Q_HEADS;
            LoweredCommand {
                kernel: KernelId::RopeAppend,
                library: "rope",
                function: pick_specialized_symbol(
                    "rope_append_f16_specialized",
                    "rope_append_bf16_specialized",
                    W::METAL_DTYPE,
                ),
                constants: super::kernel_constants::RopeAppendConstants {
                    head_dim: super::ids::HeadDim(W::HEAD_DIM),
                    num_q_heads: super::ids::NumQHeads(W::NUM_Q_HEADS),
                    num_kv_heads: super::ids::NumKvHeads(W::NUM_KV_HEADS),
                    rot_dim: super::ids::RotDim(W::ROT_DIM),
                    block_size: super::ids::BlockSize(W::BLOCK_SIZE),
                }
                .into(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (W::HEAD_DIM, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: super::kernel_bindings::RopeAppendBindingSet {
                    q_out: super::ids::ArenaSlotIdx(*q_out_slot),
                    k_out: super::ids::ArenaSlotIdx(*k_out_slot),
                    v_out: super::ids::ArenaSlotIdx(*v_out_slot),
                    cos_sin_locator: super::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    layer: super::ids::LayerId(*layer + layer_offset),
                }
                .into(),
                gemm_dims: None,
            }
        }

        // ── Fused QKV matmul + RoPE + paged KV-cache write ─────────
        // Dense BF16/F16 path; Llama-style NeoX, no QKV bias. Qwen2
        // bias / Cohere interleaved variants land in follow-up
        // commits, gated at the matcher.
        I::FusedQkvRopeCache(
            in_slot,
            out_slot,
            layer,
            biased,
            interleaved,
        ) => {
            assert!(
                !*biased && !*interleaved,
                "metal lowering: FusedQkvRopeCache currently supports only \
                 NeoX-style RoPE without QKV bias (biased={biased} \
                 interleaved={interleaved}); the matcher should not have \
                 claimed this shape",
            );
            let n_q_heads = W::NUM_Q_HEADS;
            let n_kv_heads = W::NUM_KV_HEADS;
            let num_heads_total = n_q_heads + 2 * n_kv_heads;
            LoweredCommand {
                kernel: KernelId::FusedQkvRopeCache,
                library: "fused_qkv_rope_cache",
                function: pick_specialized_symbol(
                    "fused_qkv_rope_cache_f16_specialized",
                    "fused_qkv_rope_cache_bf16_specialized",
                    W::METAL_DTYPE,
                ),
                constants: super::kernel_constants::FusedQkvRopeCacheConstants {
                    q_size: super::ids::QSize(W::Q_SIZE as u32),
                    num_q_heads: super::ids::NumQHeads(W::NUM_Q_HEADS),
                    num_kv_heads: super::ids::NumKvHeads(W::NUM_KV_HEADS),
                    head_dim: super::ids::HeadDim(W::HEAD_DIM),
                    rot_dim: super::ids::RotDim(W::ROT_DIM),
                    block_size: super::ids::BlockSize(W::BLOCK_SIZE),
                    bucket_m: super::ids::BucketM(bucket_m),
                }
                .into(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_heads_total, 1),
                    threads_per_threadgroup: (W::HEAD_DIM, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: super::kernel_bindings::FusedQkvRopeCacheBindingSet {
                    q_out: super::ids::ArenaSlotIdx(*out_slot),
                    input: super::ids::ArenaSlotIdx(*in_slot),
                    qkv_locator: super::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    cos_sin_locator: super::lowered::WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 1,
                    },
                    layer: super::ids::LayerId(*layer + layer_offset),
                }
                .into(),
                gemm_dims: None,
            }
        }

        // ── Compiler-synthesized pre-attention chunk ───────────────
        // Generated by ferrite-forward-macro::fuse_pass at macro time;
        // bound at runtime via the SpecializedPipelineCache's
        // source-library registry (populated at worker-pool init from
        // the macro-emitted per-arch SYNTHESIZED_KERNEL_SOURCES const).
        //
        // Same dispatch shape as the hand-written
        // fused_add_rmsnorm_affine_qkv_rope_cache kernel:
        //   threadgroups = (M, NUM_Q + 2*NUM_KV, 1)
        //   threads_per_tg = MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP
        I::SynthPreAttn(
            residual_slot,
            delta_slot,
            q_out_slot,
            layer,
            group_size,
            bits,
            symbol,
            has_linear_bias,
        ) => {
            assert_eq!(
                *bits, 4,
                "metal lowering: SynthPreAttn only wired for bits=4"
            );
            let _ = group_size; // baked into the symbol name; reserved for future per-gs dispatch tuning
            let n_q_heads = W::NUM_Q_HEADS;
            let n_kv_heads = W::NUM_KV_HEADS;
            let num_heads_total = n_q_heads + 2 * n_kv_heads;
            let threads_per_tg = 32 * W::HEAD_DIM / 4;
            LoweredCommand {
                kernel: KernelId::SynthPreAttn,
                // Library name = the synthesized kernel's symbol; the
                // SpecializedPipelineCache stores one synthesized
                // library per symbol so `library == function` here.
                library: *symbol,
                function: *symbol,
                // Pre-attn synth kernel bakes HIDDEN / NUM_Q / NUM_KV /
                // HEAD_DIM / ROT_DIM / BLOCK_SIZE / EPS as MSL
                // `constant constexpr` literals at synth time. Only
                // `M` (active token count up to bucket capacity)
                // stays a function constant — varies per bucket.
                constants: super::kernel_constants::SynthMegakernelConstants { bucket_m: super::ids::BucketM(bucket_m) }.into(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_heads_total, 1),
                    threads_per_threadgroup: (threads_per_tg, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: {
                    let mut v: Vec<Binding> = vec![
                    // 0: q_out
                    Binding::ArenaSlot { slot: *q_out_slot, binding_index: 0 },
                    // 1: residual_io (read+write)
                    Binding::ArenaSlot { slot: *residual_slot, binding_index: 1 },
                    // 2: delta (read)
                    Binding::ArenaSlot { slot: *delta_slot, binding_index: 2 },
                    // 3: rms_weight
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 3,
                    },
                    // 4..6: Q weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 6,
                    },
                    // 7..9: K weight + scales + biases (LinearLayer sub-slot 1)
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 7,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 8,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 9,
                    },
                    // 10..12: V weight + scales + biases (LinearLayer sub-slot 2)
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 2 },
                        binding_index: 10,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 2 },
                        binding_index: 11,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 2 },
                        binding_index: 12,
                    },
                    // 13: cos_sin
                    Binding::Weight {
                        kind: WeightBundleKind::CosSin,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 13,
                    },
                    // 14: positions
                    Binding::Runtime {
                        kind: RuntimeBindingKind::Positions,
                        binding_index: 14,
                    },
                    // 15: slot_mapping
                    Binding::Runtime {
                        kind: RuntimeBindingKind::SlotMapping,
                        binding_index: 15,
                    },
                    // 16: kv_cache_k
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheK { layer: super::ids::LayerId(*layer + layer_offset) },
                        binding_index: 16,
                    },
                    // 17: kv_cache_v
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV { layer: super::ids::LayerId(*layer + layer_offset) },
                        binding_index: 17,
                    },
                    ];
                    if *has_linear_bias {
                        // 18..20: Q / K / V linear bias (Qwen2-style
                        // per-row bias on each QKV LinearLayer). The
                        // worker resolves `AffineLinearBias` against
                        // `LinearLayer::AffineQuant.linear_bias` —
                        // load_affine_quant auto-detects `<prefix>.bias`
                        // in the safetensors. Bias-free arches never
                        // reach this branch.
                        v.push(Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineLinearBias,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                            binding_index: 18,
                        });
                        v.push(Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineLinearBias,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                            binding_index: 19,
                        });
                        v.push(Binding::Weight {
                            kind: WeightBundleKind::LinearLayer,
                            which: WeightTensor::AffineLinearBias,
                            layer: super::ids::LayerId(*layer + layer_offset),
                            locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 2 },
                            binding_index: 20,
                        });
                    }
                    v
                },
                gemm_dims: None,
            }
        }

        // ── Compiler-synthesized MLP pre-down megakernel ───────────
        // Mirrors SynthPreAttn but for the (FusedAddRmsNorm + gate qmv
        // + up qmv + SiluMul) chain. Output `silu_mul_out_slot` is a
        // device buffer of shape `[M, intermediate_size]` consumed by
        // the down-projection's standalone AffineQmm.
        //
        // Dispatch shape:
        //   threadgroups = (M, INTERMEDIATE_SIZE / HEAD_DIM, 1)
        //   threads_per_tg = MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP
        // The qmv atom is reused unchanged by aliasing kernel-scope
        // `__head_dim` = TILE_N and `__head` = tile index. TILE_N is
        // set to HEAD_DIM so the same `32 * HEAD_DIM / 4` thread count
        // and per-simdgroup row layout carries over from pre-attn.
        I::SynthMlpPreDown(
            residual_slot,
            delta_slot,
            silu_mul_out_slot,
            layer,
            group_size,
            bits,
            symbol,
        ) => {
            assert_eq!(
                *bits, 4,
                "metal lowering: SynthMlpPreDown only wired for bits=4"
            );
            let _ = group_size;
            let tile_n = W::HEAD_DIM;
            let intermediate = W::INTERMEDIATE_SIZE as u32;
            let num_tiles = intermediate / tile_n;
            assert!(
                intermediate % tile_n == 0,
                "metal lowering: SynthMlpPreDown requires INTERMEDIATE_SIZE \
                 ({intermediate}) divisible by HEAD_DIM ({tile_n})"
            );
            let threads_per_tg = 32 * tile_n / 4;
            LoweredCommand {
                kernel: KernelId::SynthMlpPreDown,
                library: *symbol,
                function: *symbol,
                // MLP-pre-down synth kernel bakes HIDDEN / INTERMEDIATE /
                // TILE_N / EPS as MSL `constant constexpr` literals at
                // synth time. Only `M_FC` stays a function constant.
                constants: super::kernel_constants::SynthMegakernelConstants { bucket_m: super::ids::BucketM(bucket_m) }.into(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_tiles, 1),
                    threads_per_threadgroup: (threads_per_tg, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: vec![
                    // 0: silu_mul_out
                    Binding::ArenaSlot { slot: *silu_mul_out_slot, binding_index: 0 },
                    // 1: residual_io (read+write)
                    Binding::ArenaSlot { slot: *residual_slot, binding_index: 1 },
                    // 2: delta (read)
                    Binding::ArenaSlot { slot: *delta_slot, binding_index: 2 },
                    // 3: rms_weight
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 3,
                    },
                    // 4..6: gate weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 6,
                    },
                    // 7..9: up weight + scales + biases (LinearLayer sub-slot 1)
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 7,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 8,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 9,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── SynthGateUpSiluMul — fused gate+up GEMM + SiluMul (large-M) ─────
        I::SynthGateUpSiluMul(
            x_norm_slot,
            out_slot,
            layer,
            _group_size,
            _bits,
            symbol,
        ) => {
            let intermediate = W::INTERMEDIATE_SIZE as u32;
            let tg_n = 32u32;
            let tg_m = 32u32;
            LoweredCommand {
                kernel: KernelId::SynthGateUpSiluMul,
                library: *symbol,
                function: *symbol,
                constants: super::kernel_constants::SynthMegakernelConstants { bucket_m: super::ids::BucketM(bucket_m) }.into(),
                dispatch: DispatchShape {
                    threadgroups: (intermediate.div_ceil(tg_n), bucket_m.div_ceil(tg_m), 1),
                    threads_per_threadgroup: (128, 1, 1),
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::Y, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: vec![
                    Binding::ArenaSlot { slot: *out_slot,    binding_index: 0 },
                    Binding::ArenaSlot { slot: *x_norm_slot, binding_index: 1 },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 2,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 3,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 0 },
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::Weight,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineScales,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 6,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer,
                        which: WeightTensor::AffineBiases,
                        layer: super::ids::LayerId(*layer + layer_offset),
                        locator: WeightLocator { bucket: tape_index, op_idx: index as u32, slot: 1 },
                        binding_index: 7,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── Decode-bucket attention (single query token / seq) ─────
        I::AttentionViaCache(q_slot, out_slot, layer, _is_decode) => {
            // 2D dispatch: (batch, num_q_heads). Each threadgroup
            // computes one head's attention output for one sequence.
            // `bucket_m == batch` for decode buckets.
            //
            // v2 kernel (paged-cache port of MLX sdpa_vector) uses
            // 1024 threads/group = 32 simdgroups × 32 lanes. Each
            // simdgroup processes 1/32 of the K axis with online
            // softmax; no per-token threadgroup_barrier in the K
            // loop. Production path now wires both f16 and bf16 to v2
            // (the v1 2-pass-softmax kernel accumulated bf16 rounding
            // error per layer on Llama-3.2 decode and produced
            // degenerate output after the first decode token).
            let n_q_heads = W::NUM_Q_HEADS;
            LoweredCommand {
                kernel: KernelId::AttentionViaCache,
                library: "attention",
                function: pick_specialized_symbol(
                    "attention_via_cache_v2_f16_specialized",
                    "attention_via_cache_v2_bf16_specialized",
                    W::METAL_DTYPE,
                ),
                constants: super::kernel_constants::AttentionViaCacheConstants {
                    head_dim: super::ids::HeadDim(W::HEAD_DIM),
                    num_q_heads: super::ids::NumQHeads(W::NUM_Q_HEADS),
                    num_kv_heads: super::ids::NumKvHeads(W::NUM_KV_HEADS),
                    attn_scale: super::ids::AttnScale(W::ATTN_SCALE),
                    block_size: super::ids::BlockSize(W::BLOCK_SIZE),
                    max_blocks: super::ids::MaxBlocksPerSeq(W::MAX_BLOCKS_PER_SEQ),
                }
                .into(),
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (1024, 1, 1),
                    // AttentionViaCache (decode) — bucket_m == 1 here
                    // (decode bucket). Scaling is a no-op but kept
                    // for uniformity in case decode shares a bucket.
                    m_scaling: Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) }),
                },
                bindings: super::kernel_bindings::AttentionViaCacheBindingSet {
                    output: super::ids::ArenaSlotIdx(*out_slot),
                    q: super::ids::ArenaSlotIdx(*q_slot),
                    kv_layer: super::ids::LayerId(*layer + layer_offset),
                }
                .into(),
                gemm_dims: None,
            }
        }

        // ── Plain prefill on metal goes through `AttentionPrefillPaged` ─
        // Phase B (`crates/ferrite-forward-macro/src/metal/attention.rs`
        // `(false, true)` arm) emits `Instruction::AttentionPrefillPaged`
        // for every metal Llama-arch model — the contiguous variant
        // is CUDA-only on metal builds. Hitting this arm means a
        // future macro change started emitting the wrong variant.
        I::AttentionPrefillContiguous(_, _, _, _, _) => {
            unreachable!(
                "metal lowering: AttentionPrefillContiguous is unreachable post-Phase B \
                 — metal::attention::fan_out emits AttentionPrefillPaged. \
                 If this fires, a macro adapter for a metal model is emitting the wrong variant."
            );
        }

        // ── Prefill-bucket attention reading from the paged KV cache ─
        // The paged variant of `attention_prefill_sdpa_v2_*`. The
        // upstream `RopeAppend` wrote rotated K + raw V into the
        // per-layer paged cache; this kernel reads them through
        // `block_table` indirection. K-axis covers the FULL
        // `seqused_k[seq]` (prefix + new tokens), so the kernel
        // serves chunked-prefill / prefix-cache-hit / multi-turn
        // scenarios that the contiguous prefill kernel cannot
        // (its K-axis = `cu_seqlens_q`, new tokens only).
        //
        // Bindings match the kernel's `set_buffer(i, …)` order:
        // 0 output, 1 Q, 2 cu_seqlens_q, 3 seq_used_k,
        // 4 block_table, 5 K cache (per-layer), 6 V cache (per-layer).
        // Dispatch matches `AttentionPrefillSdpa` (1 Q per
        // threadgroup, head on grid X, Q on grid Y, 1024 threads).
        I::AttentionPrefillPaged(q_slot, out_slot, layer, _interleaved) => {
            // Steel-attention paged kernel — opt-in via env var while
            // we debug the bisected coherence regression
            // (commit df84c658d). Default path is the sdpa_vector port
            // ("attention_prefill_sdpa_v2_paged_*") which is known
            // correct.
            //
            // The four (kernel, dtype) combinations are typed ZSTs in
            // `super::kernel_identity`; routing through `for_kernel<K>`
            // means the (library, function, KERNEL_ID) trio comes from
            // one source. Bug class #8 — drift between the three
            // independent `&'static str` fields — can't recur.
            use super::kernel_identity::{
                AttentionSdpaPagedBf16, AttentionSdpaPagedF16,
                AttentionSteelPagedBf16, AttentionSteelPagedF16,
            };
            let n_q_heads = W::NUM_Q_HEADS;
            let use_steel = std::env::var_os("FERRITE_METAL_STEEL_ATTN").is_some();
            const BQ_STEEL: u32 = 32;
            let (tg_shape, threads_per_tg, m_scale_axis) = if use_steel {
                let nq_blocks = bucket_m.div_ceil(BQ_STEEL);
                (
                    (nq_blocks, n_q_heads, 1),
                    (128u32, 1u32, 1u32),
                    super::lowered::MScaleAxis::X,
                )
            } else {
                (
                    (n_q_heads, bucket_m, 1),
                    (1024u32, 1u32, 1u32),
                    super::lowered::MScaleAxis::Y,
                )
            };
            let constants = super::kernel_constants::AttentionPrefillPagedConstants {
                head_dim: super::ids::HeadDim(W::HEAD_DIM),
                num_q_heads: super::ids::NumQHeads(W::NUM_Q_HEADS),
                num_kv_heads: super::ids::NumKvHeads(W::NUM_KV_HEADS),
                attn_scale: super::ids::AttnScale(W::ATTN_SCALE),
                block_size: super::ids::BlockSize(W::BLOCK_SIZE),
                max_blocks: super::ids::MaxBlocksPerSeq(W::MAX_BLOCKS_PER_SEQ),
                // Steel kernel reads slot 99; omitting it leaves Metal
                // undefined and the kernel can hit a diagnostic path
                // (the b3ddb3b46 regression). sdpa_vector ignores it.
                debug_mode: if use_steel {
                    Some(super::ids::AttnDebugMode(0))
                } else {
                    None
                },
            };
            let bindings = super::kernel_bindings::AttentionPrefillPagedBindingSet {
                output: super::ids::ArenaSlotIdx(*out_slot),
                q: super::ids::ArenaSlotIdx(*q_slot),
                kv_layer: super::ids::LayerId(*layer + layer_offset),
            };
            let dispatch = DispatchShape {
                threadgroups: tg_shape,
                threads_per_threadgroup: threads_per_tg,
                m_scaling: Some(crate::interpreter::metal::lowered::MScaling {
                    axis: m_scale_axis,
                    bucket_m: super::ids::BucketM(bucket_m),
                }),
            };
            match (use_steel, W::METAL_DTYPE) {
                (true, super::lowered::MetalDtype::Bf16) => {
                    LoweredCommand::for_kernel::<AttentionSteelPagedBf16>(
                        constants, bindings, dispatch,
                    )
                }
                (true, _) => LoweredCommand::for_kernel::<AttentionSteelPagedF16>(
                    constants, bindings, dispatch,
                ),
                (false, super::lowered::MetalDtype::Bf16) => {
                    LoweredCommand::for_kernel::<AttentionSdpaPagedBf16>(
                        constants, bindings, dispatch,
                    )
                }
                (false, _) => LoweredCommand::for_kernel::<AttentionSdpaPagedF16>(
                    constants, bindings, dispatch,
                ),
            }
        }

        // ── Plain residual add ─────────────────────────────────────
        I::Add(delta_slot, residual_slot) => LoweredCommand {
            kernel: KernelId::Add,
            library: "elementwise",
            function: pick_specialized_symbol(
                "residual_add_f16_specialized",
                "residual_add_bf16_specialized",
                W::METAL_DTYPE,
            ),
            // Token-parallel kernel reads element count from dispatch
            // shape; no function constants.
            constants: Vec::new(),
            // Token-parallel; reduction is per-element so the
            // dispatch covers `M * hidden_size` elements.
            dispatch: {
                let mut d = DispatchShape::dispatch_1d(bucket_m * W::Q_SIZE as u32, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) });
                d
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: *residual_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *delta_slot,
                    binding_index: 1,
                },
            ],
            gemm_dims: None,
        },

        // ── Scalar-multiply broadcast ──────────────────────────────
        I::ScalarMul(in_slot, out_slot, _scale) => LoweredCommand {
            kernel: KernelId::ScalarMul,
            library: "elementwise",
            function: pick_specialized_symbol(
                "scalar_mul_f16_specialized",
                "scalar_mul_bf16_specialized",
                W::METAL_DTYPE,
            ),
            constants: Vec::new(),
            dispatch: {
                let mut d = DispatchShape::dispatch_1d(bucket_m * W::Q_SIZE as u32, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling { axis: super::lowered::MScaleAxis::X, bucket_m: super::ids::BucketM(bucket_m) });
                d
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 1,
                },
                // The scalar `scale` is baked into a function
                // constant on the specialized pipeline (Phase 5.B);
                // no runtime binding needed.
            ],
            gemm_dims: None,
        },

        // ── Per-row bias broadcast (singleton, non-synth path) ────
        //
        // Emitted by `MetalBiasAddImpl` for Qwen2-style QKV biases
        // when the synth pre-attn megakernel doesn't claim the chain
        // (today: M ≥ 2 prefill). One bias-add dispatch per BiasAdd
        // tile; the synth path will subsume these at M=1 once the
        // matcher absorbs biases (P2).
        //
        // Binding contract (matches `bias_add_<dtype>_specialized` in
        // `elementwise.metal`):
        //   buffer(0) = input  [M, N]            T_act r
        //   buffer(1) = bias   [N]               T_act r
        //   buffer(2) = output [M, N]            T_act w
        //
        // `WeightTensor::Bias` resolves dense `<prefix>.bias`;
        // `WeightTensor::AffineLinearBias` resolves MLX-affine's
        // `linear_bias` (Qwen2 4bit ships this). `is_affine` flag on
        // the Instruction picks which arm — the macro knows the
        // weight's StorageFormat at FUF construction time.
        I::MetalBiasAdd(in_slot, out_slot, layer, n, is_affine) => LoweredCommand {
            kernel: KernelId::BiasAdd,
            library: "elementwise",
            function: pick_specialized_symbol(
                "bias_add_f16_specialized",
                "bias_add_bf16_specialized",
                W::METAL_DTYPE,
            ),
            constants: vec![ConstantValue::uint(0, *n)],
            dispatch: {
                let mut d =
                    DispatchShape::dispatch_1d(bucket_m * *n, THREADS_PER_GROUP);
                d.m_scaling = Some(crate::interpreter::metal::lowered::MScaling {
                    axis: super::lowered::MScaleAxis::X,
                    bucket_m: super::ids::BucketM(bucket_m),
                });
                d
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: *in_slot,
                    binding_index: 0,
                },
                Binding::Weight {
                    kind: WeightBundleKind::LinearLayer,
                    which: if *is_affine {
                        WeightTensor::AffineLinearBias
                    } else {
                        WeightTensor::Bias
                    },
                    layer: super::ids::LayerId(*layer + layer_offset),
                    locator: WeightLocator {
                        bucket: tape_index,
                        op_idx: index as u32,
                        slot: 0,
                    },
                    binding_index: 1,
                },
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        },

        // ── Metadata-only: no Metal dispatch ───────────────────────
        I::Reshape(_, _, _, _, _, _) | I::Alias(_, _) | I::Free(_) => {
            // These rebind / drop slots in the dispatcher's logical
            // view but don't touch device memory. Subsequent commands
            // in the lowered tape see the new logical shape via the
            // worker's slot tracker (resolved at worker init).
            return Ok(Vec::new());
        }

        // ── Loop already handled by the caller ─────────────────────
        I::Loop(_, _) => unreachable!("Loop is handled in `lower()` directly"),

        // ── Everything else: not yet on the TinyLlama critical path
        other => {
            return Err(LoweringError::UnsupportedVariant {
                index,
                variant_type: std::any::type_name_of_val(other),
            });
        }
    };

    Ok(vec![cmd])
}

/// Default 1D threads-per-threadgroup. Matches Metal's preferred
/// width on Apple Silicon (32-wide simdgroups × 8 = 256). The Phase
/// 5.B specialized-pipeline cache may pick a different value per
/// kernel + bucket once measured.
const THREADS_PER_GROUP: u32 = 256;

/// 2D GEMM tile dims (M × N axes). Placeholder — kept narrow so the
/// dispatch math stays sensible at small buckets; will be replaced
/// per (model, bucket) by the SpecializedPipelineCache in Phase 5.B.
const GEMM_TILE_M: u32 = 16;
const GEMM_TILE_N: u32 = 16;

/// Output tile dim for the prefill matrix variant
/// (`fused_gate_up_silu_mul_gemm_steel_*_specialized`). 4 simdgroups
/// per threadgroup × WM*WN = 2*2 placement → 32×32 output tile.
/// Matches `STEEL_BM`/`STEEL_BN` in `shaders/fused_gate_up_silu_mul.metal`.
const MLP_STEEL_TILE: u32 = 32;

/// Threads per threadgroup for the steel matrix variant.
/// `WM * WN * 32 = 2 * 2 * 32 = 128`.
const MLP_STEEL_THREADS: u32 = 128;

/// Output rows per threadgroup for the M=1 decode variant
/// (`fused_gate_up_silu_mul_decode_*_specialized`). Matches the
/// MLX gemv port's `blockM = BM*SM*TM = 4`.
const MLP_DECODE_BLOCK_M: u32 = 4;

/// Pick the f16-or-bf16 specialization for a kernel that follows the
/// `<base>_<dtype>_specialized` naming convention. Centralizes the
/// `Int4`-not-yet-wired panic so each lowering arm spells out only
/// the two symbol names it owns.
fn pick_specialized_symbol(
    f16_symbol: &'static str,
    bf16_symbol: &'static str,
    dtype: MetalDtype,
) -> &'static str {
    match dtype {
        MetalDtype::F16 => f16_symbol,
        MetalDtype::Bf16 => bf16_symbol,
        // AWQ/GPTQ dequant kernels have a different binding contract
        // (packed u32 weights + group scales) so a single dtype
        // substitution can't model them — surfaced as a panic for
        // clarity since `W::METAL_DTYPE = Int4` is unreachable today.
        MetalDtype::Int4 => {
            panic!("metal lowering: Int4 dtype not yet wired through kernel symbol picker")
        }
    }
}

/// Static `&'static str` for the `silu_mul_<dtype>` symbol exported
/// by `silu_mul.metal`. Same `&'static str` constraint as the qmv /
/// qmm name helpers — `LoweredCommand::function` can't allocate.
fn silu_mul_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "silu_mul_f16",
        DequantDtype::Bf16 => "silu_mul_bf16",
    }
}

/// Convert the lowering-side dtype enum to the kernel-dispatcher one.
/// `MetalDtype` is the lowering vocabulary; `DequantDtype` is what
/// `quantized.rs` speaks (and what `qmv_kernel_static_name` /
/// `qmm_t_kernel_static_name` consume). Same two cases either way —
/// the duplicate enum exists because the kernel-dispatcher crate
/// can't depend on lowering types.
fn dequant_dtype_for<W: CanonicalParams>() -> DequantDtype {
    match W::METAL_DTYPE {
        MetalDtype::F16 => DequantDtype::F16,
        MetalDtype::Bf16 => DequantDtype::Bf16,
        MetalDtype::Int4 => panic!(
            "metal lowering: Instruction::AffineQmm requires W::METAL_DTYPE \
             ∈ {{F16, Bf16}} (the activation dtype); got Int4"
        ),
    }
}

/// Scale-storage dtype the kernel reads `*.scales` / `*.biases` device
/// pointers as — i.e. the safetensors on-disk dtype for the affine
/// quant per-group params. Every sampled mlx-community 4bit checkpoint
/// ships F16 scales (`INT4_PARITY_PROBES.md:73,287`), so this is a
/// constant today; P11 (mixed-quant / NAX / FP-quant) will extend it.
/// Kept as a separate fn for symmetry with [`dequant_dtype_for`] so the
/// model author has a single seam to extend.
fn scale_dtype_for<W: CanonicalParams>() -> ScaleDtype {
    let _ = std::marker::PhantomData::<W>;
    ScaleDtype::F16
}

/// Bindings shared by every `Instruction::AffineQmm` lowering's qmv
/// and qmm_t Standard kernels — both bind buffers 0..4 in the same
/// order: (packed weight, scales, biases, x in, y out). Worker
/// resolves the `Affine*` `WeightTensor` arms via
/// `LinearLayer::AffineQuant` (`worker.rs:1414`).
fn affine_qmm_bindings(
    in_slot: u32,
    out_slot: u32,
    layer: super::ids::LayerId,
    locator: super::lowered::WeightLocator,
) -> Vec<Binding> {
    vec![
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::Weight,
            layer,
            locator,
            binding_index: 0,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineScales,
            layer,
            locator,
            binding_index: 1,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineBiases,
            layer,
            locator,
            binding_index: 2,
        },
        Binding::ArenaSlot {
            slot: in_slot,
            binding_index: 3,
        },
        Binding::ArenaSlot {
            slot: out_slot,
            binding_index: 4,
        },
    ]
}

/// Bindings for `affine_qmm_t_splitk`: same first four bindings as
/// `affine_qmm_bindings` (packed weight, scales, biases, x in) but
/// the `y` output (binding 4) is the shared `Binding::Scratch`
/// buffer instead of an arena slot — the kernel writes the
/// `[split_k, M, N]` partial here, and the follow-up
/// `splitk_reduce_sum` reads it.
fn affine_qmm_splitk_bindings(
    in_slot: u32,
    layer: super::ids::LayerId,
    locator: super::lowered::WeightLocator,
) -> Vec<Binding> {
    vec![
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::Weight,
            layer,
            locator,
            binding_index: 0,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineScales,
            layer,
            locator,
            binding_index: 1,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer,
            which: WeightTensor::AffineBiases,
            layer,
            locator,
            binding_index: 2,
        },
        Binding::ArenaSlot {
            slot: in_slot,
            binding_index: 3,
        },
        Binding::Scratch { binding_index: 4 },
    ]
}

/// Byte width of one element in the activation dtype the worker
/// allocates the SplitK scratch buffer against. F16 / Bf16 = 2 bytes.
fn elem_size_bytes(dtype: DequantDtype) -> u32 {
    match dtype {
        DequantDtype::F16 | DequantDtype::Bf16 => 2,
    }
}

/// Format the kernel symbol name for an `Instruction::RmsNorm`
/// lowering. Matches the `INST_RMSNORM` instantiations in
/// `shaders/rmsnorm.metal` — `rmsnorm_<T_act>_s_<T_scale>_specialized`.
/// Mirrors P10b's in-register cast pattern (`feedback_no_silent_deferrals`
/// / `INT4_PARITY_PROBES.md` §7): RMSNorm gains stay in their on-disk
/// dtype on the device, the kernel reads them through a `T_scale`
/// pointer and casts to `T_act` in registers.
fn rmsnorm_kernel_static_name<W: CanonicalParams>(
    scale_dtype: ScaleDtype,
) -> &'static str {
    use ScaleDtype as S;
    match (W::METAL_DTYPE, scale_dtype) {
        (MetalDtype::F16, S::F16) => "rmsnorm_f16_s_f16_specialized",
        (MetalDtype::Bf16, S::F16) => "rmsnorm_bf16_s_f16_specialized",
        (dt, sdt) => unreachable!(
            "rmsnorm_kernel_static_name: (dtype={dt:?}, scale_dtype={sdt:?}) \
             not instantiated — only (f16|bf16, f16) ship today; \
             lower_one's pre-checks should have caught this"
        ),
    }
}

/// As [`rmsnorm_kernel_static_name`] for `Instruction::FusedAddRmsNorm`.
/// Symbol naming: `fused_add_rmsnorm_<T_act>_s_<T_scale>_specialized`,
/// matching the `INST_FUSED_ARN` instantiations in
/// `shaders/fused_add_rmsnorm.metal`.
fn fused_add_rmsnorm_kernel_static_name<W: CanonicalParams>(
    scale_dtype: ScaleDtype,
) -> &'static str {
    use ScaleDtype as S;
    match (W::METAL_DTYPE, scale_dtype) {
        (MetalDtype::F16, S::F16) => "fused_add_rmsnorm_f16_s_f16_specialized",
        (MetalDtype::Bf16, S::F16) => "fused_add_rmsnorm_bf16_s_f16_specialized",
        (dt, sdt) => unreachable!(
            "fused_add_rmsnorm_kernel_static_name: (dtype={dt:?}, \
             scale_dtype={sdt:?}) not instantiated — only (f16|bf16, f16) \
             ship today; lower_one's pre-checks should have caught this"
        ),
    }
}

/// Format the kernel symbol name for an `AffineEmbed` lowering.
/// Matches the `DEFINE_AFFINE_EMBED_B4` macro invocations in
/// `shaders/quantized_dequantize.metal`. Same enumeration as
/// `affine_dequantize_<dtype>_s_<scale_dtype>_gs_<gs>_b_4`.
fn affine_embed_kernel_static_name(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
) -> &'static str {
    use ScaleDtype as S;
    match (dtype, scale_dtype, group_size) {
        (DequantDtype::F16, S::F16, 32)  => "affine_embed_f16_s_f16_gs_32_b_4",
        (DequantDtype::F16, S::F16, 64)  => "affine_embed_f16_s_f16_gs_64_b_4",
        (DequantDtype::F16, S::F16, 128) => "affine_embed_f16_s_f16_gs_128_b_4",
        (DequantDtype::Bf16, S::F16, 32)  => "affine_embed_bf16_s_f16_gs_32_b_4",
        (DequantDtype::Bf16, S::F16, 64)  => "affine_embed_bf16_s_f16_gs_64_b_4",
        (DequantDtype::Bf16, S::F16, 128) => "affine_embed_bf16_s_f16_gs_128_b_4",
        (dt, sdt, gs) => unreachable!(
            "affine_embed_kernel_static_name: (dtype={dt:?}, scale_dtype={sdt:?}, gs={gs}) \
             not instantiated — only (f16|bf16, f16, 32|64|128) ship; \
             lower_one's assert should have caught this"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    /// Minimal `CanonicalParams` impl for lowering-shape tests. No
    /// kernel actually runs — `lower_one` just inspects the variant
    /// fields and `W::METAL_DTYPE`. Pinned to bf16 to match the
    /// canonical Llama-3.x configuration the macro emits.
    struct TestParams;
    impl CanonicalParams for TestParams {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 32;
        const NUM_KV_HEADS: u32 = 4;
        const Q_SIZE: usize = 2048;
        const KV_SIZE: usize = 256;
        const INTERMEDIATE_SIZE: usize = 8192;
        const ATTN_SCALE: f32 = 0.125;
        const ATTN_SOFTCAP: f32 = 0.0;
        const SLIDING_WINDOW: i32 = -1;
        const KV_LORA_RANK: usize = 0;
        const QK_NOPE_HEAD_DIM: usize = 0;
        const QK_ROPE_HEAD_DIM: usize = 0;
        const V_HEAD_DIM: usize = 0;
        const FINAL_LOGIT_SOFTCAPPING: f32 = 0.0;
        const QK_HEAD_DIM: usize = 0;
        const MLA_ATTN_SCALE: f32 = 0.0;
        const METAL_DTYPE: MetalDtype = MetalDtype::Bf16;
    }

    /// `WtFn` stub. The lowering pass stores this pointer in the
    /// `Binding::Weight` payload but never calls it (resolution
    /// happens at worker bake time); a panic body keeps the type
    /// signature honest without forcing the test to construct a
    /// real `LinearLayer::AffineQuant` (which would need a Metal
    /// device for the `Buffer` allocations).
    fn affine_quant_stub(_w: &TestParams, _layer: u32) -> &ferrite_kernels::layers::LinearLayer {
        panic!("affine_quant_stub: lowering tests must not invoke wt_fn");
    }

    /// At `bucket_m < vector_limit` we should land in the qmv branch
    /// and pick `qmv_fast` for the Llama-1B q_proj shape (N=2048,
    /// K=2048, gs=64, bits=4). N % 8 == 0 && K % 512 == 0 → fast,
    /// not generic; K ∉ {64,128} → not quad.
    #[test]
    fn affine_qmm_lowers_to_qmv_fast_at_decode_bucket() {
        let inst: Instruction<TestParams> = Instruction::AffineQmm(
            /*in_slot=*/ 7,
            /*out_slot=*/ 11,
            /*layer=*/ 3,
            affine_quant_stub,
            /*n=*/ 2048,
            /*k=*/ 2048,
            /*group_size=*/ 64,
            /*bits=*/ 4,
            /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let cmds = lower_one(&inst, 0, /*bucket_m=*/ 1, /*layer_offset=*/ 5, &mut scratch)
            .expect("lower");
        assert_eq!(cmds.len(), 1, "qmv branch emits exactly one command");
        assert_eq!(scratch, 0, "qmv branch never allocates splitk scratch");
        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::AffineQmvFast);
        assert_eq!(cmd.library, "quantized_qmv");
        assert_eq!(cmd.function, "affine_qmv_fast_bf16_s_f16_gs_64_b_4_batch_0");
        assert_eq!(
            cmd.constants,
            vec![ConstantValue::int(0, 2048), ConstantValue::int(1, 2048)],
        );
        // qmv_fast grid: (M, ceil(N/8), B); group: (32, 2, 1).
        assert_eq!(cmd.dispatch.threadgroups, (1, 2048 / 8, 1));
        assert_eq!(cmd.dispatch.threads_per_threadgroup, (32, 2, 1));
        // 5 bindings: weight (idx 0), scales (1), biases (2), in (3), out (4).
        assert_eq!(cmd.bindings.len(), 5);
        // The `layer` baked into the bindings = inst.layer + layer_offset.
        match &cmd.bindings[0] {
            Binding::Weight {
                which,
                layer,
                binding_index,
                ..
            } => {
                assert_eq!(*which, WeightTensor::Weight);
                assert_eq!(layer.0, 8);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected Weight"),
        }
        match &cmd.bindings[1] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineScales),
            _ => panic!("bindings[1]: expected AffineScales Weight"),
        }
        match &cmd.bindings[2] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineBiases),
            _ => panic!("bindings[2]: expected AffineBiases Weight"),
        }
        match &cmd.bindings[3] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 7);
                assert_eq!(*binding_index, 3);
            }
            _ => panic!("bindings[3]: expected in_slot ArenaSlot"),
        }
        match &cmd.bindings[4] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 11);
                assert_eq!(*binding_index, 4);
            }
            _ => panic!("bindings[4]: expected out_slot ArenaSlot"),
        }
        assert!(cmd.gemm_dims.is_none());
    }

    /// At a bucket_m where `pick_qmm_t_kernel` returns `Standard`
    /// (current_tgs ≥ 512 → split_k=1), AffineQmm lowers to a single
    /// `KernelId::AffineQmmT` command. bucket_m=512 with N=K=2048,
    /// gs=64 gives n_tiles=64, m_tiles=16 → current_tgs=1024 ≥ 512.
    /// aligned_N=true since N=2048 % 32 == 0.
    #[test]
    fn affine_qmm_lowers_to_qmm_t_standard_when_grid_full() {
        let inst: Instruction<TestParams> = Instruction::AffineQmm(
            /*in_slot=*/ 7,
            /*out_slot=*/ 11,
            /*layer=*/ 3,
            affine_quant_stub,
            /*n=*/ 2048,
            /*k=*/ 2048,
            /*group_size=*/ 64,
            /*bits=*/ 4,
            /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let cmds = lower_one(&inst, 0, /*bucket_m=*/ 512, /*layer_offset=*/ 0, &mut scratch)
            .expect("lower");
        assert_eq!(cmds.len(), 1, "Standard path emits exactly one command");
        assert_eq!(scratch, 0, "Standard path never allocates splitk scratch");
        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::AffineQmmT);
        assert_eq!(cmd.library, "quantized_qmm");
        assert_eq!(cmd.function, "affine_qmm_t_bf16_s_f16_gs_64_b_4_alN_true_batch_0");
        assert_eq!(
            cmd.constants,
            vec![
                ConstantValue::int(0, 2048),
                ConstantValue::int(1, 2048),
                ConstantValue::int(2, 512),
            ],
        );
        // qmm_t grid: (ceil(N/32), ceil(M/32), B); group: (32, 2, 2).
        assert_eq!(cmd.dispatch.threadgroups, (2048 / 32, 512 / 32, 1));
        assert_eq!(cmd.dispatch.threads_per_threadgroup, (32, 2, 2));
        assert_eq!(cmd.bindings.len(), 5);
        assert!(cmd.gemm_dims.is_none());
    }

    /// At a bucket_m where `pick_qmm_t_kernel` returns SplitK
    /// (current_tgs < 512), AffineQmm lowers to TWO commands:
    /// `AffineQmmTSplitK` writing to the shared scratch buffer +
    /// `SplitKReduceSum` reducing `[split_k, M, N]` → `[M, N]`.
    /// bucket_m=64 with N=K=2048, gs=64 → n_tiles=64, m_tiles=2,
    /// current_tgs=128, split_k=4 (gated to 2048 % (4*64) == 0).
    #[test]
    fn affine_qmm_lowers_to_qmm_t_splitk_pair_at_sparse_prefill() {
        let inst: Instruction<TestParams> = Instruction::AffineQmm(
            /*in_slot=*/ 7,
            /*out_slot=*/ 11,
            /*layer=*/ 3,
            affine_quant_stub,
            /*n=*/ 2048,
            /*k=*/ 2048,
            /*group_size=*/ 64,
            /*bits=*/ 4,
            /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let cmds = lower_one(&inst, 0, /*bucket_m=*/ 64, /*layer_offset=*/ 0, &mut scratch)
            .expect("lower");
        assert_eq!(cmds.len(), 2, "SplitK pair emits two commands");
        // Scratch sized to split_k * M * N * 2 bytes (bf16 = 2):
        // 4 * 64 * 2048 * 2 = 1_048_576.
        assert_eq!(scratch, 4 * 64 * 2048 * 2);

        let qmm_t = &cmds[0];
        assert_eq!(qmm_t.kernel, KernelId::AffineQmmTSplitK);
        assert_eq!(qmm_t.library, "quantized_qmm");
        assert_eq!(
            qmm_t.function,
            "affine_qmm_t_splitk_bf16_s_f16_gs_64_b_4_alN_true",
        );
        // qmm_t_splitk grid: (n_tiles, m_tiles, split_k); group same as qmm_t.
        assert_eq!(qmm_t.dispatch.threadgroups, (64, 2, 4));
        assert_eq!(qmm_t.dispatch.threads_per_threadgroup, (32, 2, 2));
        // Bindings: w/scales/biases as Weight (0/1/2), in as ArenaSlot (3),
        // y output as Scratch (4).
        assert_eq!(qmm_t.bindings.len(), 5);
        match &qmm_t.bindings[4] {
            Binding::Scratch { binding_index } => assert_eq!(*binding_index, 4),
            _ => panic!("qmm_t bindings[4]: expected Scratch"),
        }

        let reduce = &cmds[1];
        assert_eq!(reduce.kernel, KernelId::SplitKReduceSum);
        assert_eq!(reduce.library, "quantized_splitk_reduce");
        assert_eq!(reduce.function, "splitk_reduce_sum_bf16");
        // reduce constants: 0=M, 1=N, 2=split_k.
        assert_eq!(
            reduce.constants,
            vec![
                ConstantValue::uint(0, 64),
                ConstantValue::uint(1, 2048),
                ConstantValue::uint(2, 4),
            ],
        );
        // Bindings: 0 = output ArenaSlot, 1 = Scratch input.
        assert_eq!(reduce.bindings.len(), 2);
        match &reduce.bindings[0] {
            Binding::ArenaSlot { slot, binding_index } => {
                assert_eq!(*slot, 11);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("reduce bindings[0]: expected out ArenaSlot"),
        }
        match &reduce.bindings[1] {
            Binding::Scratch { binding_index } => assert_eq!(*binding_index, 1),
            _ => panic!("reduce bindings[1]: expected Scratch"),
        }
    }

    /// SiluMul lowers to a 1D dispatch over `bucket_m * intermediate_size`
    /// elements with three ArenaSlot bindings (out, gate, up). Function
    /// constant 0 holds the total element count.
    #[test]
    fn silu_mul_lowers_with_three_arena_bindings_and_n_constant() {
        let inst: Instruction<TestParams> =
            Instruction::SiluMul(/*gate=*/ 5, /*up=*/ 6, /*out=*/ 7);
        let mut scratch = 0u32;
        let cmds = lower_one(&inst, 0, /*bucket_m=*/ 64, /*layer_offset=*/ 0, &mut scratch)
            .expect("lower");
        assert_eq!(cmds.len(), 1, "SiluMul emits exactly one command");
        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::SiluMul);
        assert_eq!(cmd.library, "silu_mul");
        assert_eq!(cmd.function, "silu_mul_bf16");
        // n = bucket_m * INTERMEDIATE_SIZE = 64 * 8192.
        let n_expected = 64 * (TestParams::INTERMEDIATE_SIZE as u32);
        assert_eq!(cmd.constants, vec![ConstantValue::uint(0, n_expected)]);
        assert_eq!(cmd.bindings.len(), 3);
        match &cmd.bindings[0] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 7);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected out_slot ArenaSlot"),
        }
        match &cmd.bindings[1] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 5);
                assert_eq!(*binding_index, 1);
            }
            _ => panic!("bindings[1]: expected gate_slot ArenaSlot"),
        }
        match &cmd.bindings[2] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 6);
                assert_eq!(*binding_index, 2);
            }
            _ => panic!("bindings[2]: expected up_slot ArenaSlot"),
        }
        assert!(cmd.gemm_dims.is_none());
    }

    /// Unaligned-N Llama-1B-style lm_head (N=128256 — 128256 % 32 = 0
    /// so this is actually aligned). Use a synthetic shape for the
    /// unaligned branch: N=2050 → N % 32 = 2 → alN=false. Use
    /// bucket_m=512 so we stay on the Standard path
    /// (`pick_qmm_t_kernel` returns SplitK at small bucket_m).
    #[test]
    fn affine_qmm_qmm_t_unaligned_n_picks_unaligned_kernel() {
        let inst: Instruction<TestParams> = Instruction::AffineQmm(
            7,
            11,
            0,
            affine_quant_stub,
            /*n=*/ 2050,
            /*k=*/ 2048,
            /*group_size=*/ 64,
            /*bits=*/ 4,
            /*vector_limit=*/ 18,
        );
        let mut scratch = 0u32;
        let cmds = lower_one(&inst, 0, /*bucket_m=*/ 512, /*layer_offset=*/ 0, &mut scratch)
            .expect("lower");
        assert_eq!(cmds.len(), 1);
        let cmd = &cmds[0];
        assert_eq!(
            cmd.function,
            "affine_qmm_t_bf16_s_f16_gs_64_b_4_alN_false_batch_0"
        );
        // Ceil-div on N: 2050.div_ceil(32) = 65; M-tiles: 512/32 = 16.
        assert_eq!(cmd.dispatch.threadgroups, (65, 512 / 32, 1));
    }

    /// `WtFn` stub for AffineQuantEmbedding (P6). Same shape as
    /// `affine_quant_stub` but for the embedding bundle. Lowering
    /// stores the pointer in `Binding::Weight` and the worker
    /// resolver (untested here) calls it at ICB-record time.
    fn affine_quant_embed_stub(
        _: &TestParams,
        _: u32,
    ) -> &'static ferrite_kernels::layers::AffineQuantEmbedding {
        panic!("affine_quant_embed_stub: lowering tests must not invoke wt_fn");
    }

    /// AffineEmbed lowers to a single command targeting
    /// `affine_embed_<dtype>_gs_<gs>_b_4` in `quantized_dequantize`,
    /// with hidden_size in function_constant(0) and a 2D dispatch
    /// (bytes_per_row in X, num_tokens in Y).
    #[test]
    fn affine_embed_lowers_to_single_command_with_2d_dispatch() {
        let inst: Instruction<TestParams> = Instruction::AffineEmbed(
            /*out_slot=*/ 0,
            affine_quant_embed_stub,
            /*group_size=*/ 64,
            /*bits=*/ 4,
        );
        let bucket_m = 32u32;
        let mut scratch = 0u32;
        let cmds = lower_one(&inst, 0, bucket_m, /*layer_offset=*/ 5, &mut scratch)
            .expect("lower");
        assert_eq!(cmds.len(), 1, "AffineEmbed always emits a single command");
        assert_eq!(scratch, 0, "AffineEmbed never allocates splitk scratch");

        let cmd = &cmds[0];
        assert_eq!(cmd.kernel, KernelId::AffineEmbed);
        assert_eq!(cmd.library, "quantized_dequantize");
        // TestParams pins Bf16 + Q_SIZE=2048 → gs=64 → bf16/gs=64 symbol.
        assert_eq!(cmd.function, "affine_embed_bf16_s_f16_gs_64_b_4");

        // function_constant(0) = hidden_size = W::Q_SIZE = 2048.
        assert_eq!(cmd.constants, vec![ConstantValue::uint(0, 2048)]);

        // 2D dispatch: (Q_SIZE/2 / THREADS_PER_GROUP, bucket_m, 1).
        // Q_SIZE=2048 → bytes_per_row=1024; THREADS_PER_GROUP=256.
        // groups_x = 1024.div_ceil(256) = 4.
        assert_eq!(cmd.dispatch.threadgroups, (4, bucket_m, 1));
        assert_eq!(cmd.dispatch.threads_per_threadgroup, (THREADS_PER_GROUP, 1, 1));

        // 5 bindings: weight (0), scales (1), biases (2), input_ids (3), out (4).
        assert_eq!(cmd.bindings.len(), 5);
        match &cmd.bindings[0] {
            Binding::Weight {
                kind: WeightBundleKind::AffineQuantEmbedding,
                which,
                layer,
                locator: _,
                binding_index,
            } => {
                assert_eq!(*which, WeightTensor::Weight);
                // Embed is unlayered — `layer = 0` regardless of layer_offset.
                assert_eq!(layer.0, 0);
                assert_eq!(*binding_index, 0);
            }
            _ => panic!("bindings[0]: expected AffineQuantEmbedding Weight"),
        }
        match &cmd.bindings[1] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineScales),
            _ => panic!("bindings[1]: expected AffineScales"),
        }
        match &cmd.bindings[2] {
            Binding::Weight { which, .. } => assert_eq!(*which, WeightTensor::AffineBiases),
            _ => panic!("bindings[2]: expected AffineBiases"),
        }
        match &cmd.bindings[3] {
            Binding::Runtime {
                kind,
                binding_index,
            } => {
                assert_eq!(*kind, RuntimeBindingKind::InputIds);
                assert_eq!(*binding_index, 3);
            }
            _ => panic!("bindings[3]: expected InputIds runtime"),
        }
        match &cmd.bindings[4] {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                assert_eq!(*slot, 0);
                assert_eq!(*binding_index, 4);
            }
            _ => panic!("bindings[4]: expected out_slot ArenaSlot"),
        }
        assert!(cmd.gemm_dims.is_none());
    }
}
