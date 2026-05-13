// SPDX-License-Identifier: Apache-2.0
//! Lowering pass: `&[Instruction<W>]` → `LoweredMetalTape<W>`.
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
    MetalDtype, RuntimeBindingKind, WeightBundleKind, WeightTensor,
};

/// Lower one bucket's `(backbone ++ lm_head)` instruction stream.
///
/// Concatenation matches the cuda interpreter's effective behavior:
/// `forward()` runs backbone then lm_head in sequence for a given
/// bucket. Lowering them as one stream lets the worker bake both
/// halves into the bucket's single ICB plan, with no extra mid-bucket
/// boundary the caller needs to manage.
///
/// Avoids requiring `Instruction<W>: Clone` — the caller hands two
/// `&[Instruction<W>]` slices and we walk them in place, unrolling
/// `Loop` per the same rules as the single-slice [`lower`]. Loop
/// bodies that span the backbone/lm_head boundary are not supported
/// (no model emits one — the cuda lowering pass partitions loops
/// strictly inside one half), but if one ever shows up the malformed-
/// loop check fires inside the offending half and surfaces the
/// per-half index.
pub fn lower_pair<W: CanonicalParams>(
    backbone: &[Instruction<W>],
    lm_head: &[Instruction<W>],
    backbone_barriers: &[bool],
    lm_head_barriers: &[bool],
    bucket_m: u32,
    num_arena_slots: u32,
    profile: Option<&ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile>,
) -> Result<LoweredMetalTape<W>, LoweringError> {
    let bb = lower(backbone, backbone_barriers, bucket_m, num_arena_slots, profile)?;
    let lh = lower(lm_head, lm_head_barriers, bucket_m, num_arena_slots, profile)?;
    let mut commands = bb.commands;
    commands.extend(lh.commands);
    let mut barrier_before = bb.barrier_before;
    barrier_before.extend(lh.barrier_before);
    Ok(LoweredMetalTape {
        bucket_m,
        num_arena_slots,
        commands,
        barrier_before,
        splitk_scratch_bytes: bb.splitk_scratch_bytes.max(lh.splitk_scratch_bytes),
    })
}

/// Lower one bucket's `Instruction<W>` tape.
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
    instructions: &[Instruction<W>],
    barriers_in: &[bool],
    bucket_m: u32,
    num_arena_slots: u32,
    profile: Option<&ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile>,
) -> Result<LoweredMetalTape<W>, LoweringError> {
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
                        let cmds = lower_one(
                            inst,
                            body_start + offset,
                            bucket_m,
                            iter as u32,
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
                let cmds = lower_one(other, i, bucket_m, 0, &mut splitk_scratch_bytes, profile)?;
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
    inst: &Instruction<W>,
    index: usize,
    bucket_m: u32,
    layer_offset: u32,
    splitk_scratch_bytes: &mut u32,
    profile: Option<&ferrite_metal_kernels::ferrite_metal_targets::MetalTargetProfile>,
) -> Result<Vec<LoweredCommand<W>>, LoweringError> {
    use Instruction as I;

    let cmd = match inst {
        // ── Token embedding ────────────────────────────────────────
        I::Embed(out_slot, wt_fn) => LoweredCommand {
            kernel: KernelId::Embed,
            library: "embed",
            function: pick_specialized_symbol(
                "embed_f16_specialized",
                "embed_bf16_specialized",
                W::METAL_DTYPE,
            ),
            constants: vec![
                ConstantValue::uint(0, bucket_m),
                ConstantValue::uint(1, W::Q_SIZE as u32),
            ],
            // 1D dispatch over the `bucket_m` tokens; one thread per
            // token gathers a row from `embed_tokens.weight`.
            dispatch: DispatchShape::dispatch_1d(bucket_m, THREADS_PER_GROUP),
            bindings: vec![
                // out: arena[out_slot]
                Binding::ArenaSlot {
                    slot: *out_slot,
                    binding_index: 0,
                },
                // weight: embed_tokens.weight (layer 0; Embed is not layered)
                Binding::Weight {
                    kind: WeightBundleKind::Embedding(*wt_fn),
                    which: WeightTensor::Weight,
                    layer: 0,
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
        I::RmsNorm(in_slot, out_slot, layer, wt_fn) => LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            // Symbol names are `rmsnorm_<T_act>_s_<T_scale>_specialized`
            // — the in-register T_scale cast P10c added so RMSNorm
            // gains stay F16 on device (`feedback_no_silent_deferrals`,
            // mirrors P10b for quant scales). `scale_dtype_for::<W>()`
            // returns F16 today; expand the picker when a model ships
            // bf16 norm gains on disk.
            function: rmsnorm_kernel_static_name::<W>(scale_dtype_for::<W>()),
            constants: vec![
                ConstantValue::uint(0, bucket_m),
                ConstantValue::uint(1, W::Q_SIZE as u32),
                ConstantValue::float(2, W::RMS_NORM_EPS),
            ],
            // Per-token threadgroup; threads cooperate on the
            // hidden-size reduction inside.
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
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
                    kind: WeightBundleKind::RmsNorm(*wt_fn),
                    which: WeightTensor::Weight,
                    layer: *layer + layer_offset,
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        },

        // ── Fused residual-add + RMSNorm ───────────────────────────
        I::FusedAddRmsNorm(delta_slot, residual_slot, layer, wt_fn) => {
            LoweredCommand {
                kernel: KernelId::FusedAddRmsNorm,
                library: "fused_add_rmsnorm",
                // Symbol `fused_add_rmsnorm_<T_act>_s_<T_scale>_specialized`
                // (P10c — see RmsNorm comment above).
                function: fused_add_rmsnorm_kernel_static_name::<W>(
                    scale_dtype_for::<W>(),
                ),
                constants: vec![
                    ConstantValue::uint(0, bucket_m),
                    ConstantValue::uint(1, W::Q_SIZE as u32),
                    ConstantValue::float(2, W::RMS_NORM_EPS),
                ],
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, 1, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
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
                        kind: WeightBundleKind::RmsNorm(*wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 2,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── Generic dense GEMM ─────────────────────────────────────
        I::Gemm(in_slot, out_slot, layer, wt_fn, n, k) => {
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
                        kind: WeightBundleKind::LinearLayer(*wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
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
        I::AffineQmm(in_slot, out_slot, layer, wt_fn, n, k, group_size, bits, vector_limit) => {
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
                    constants: vec![
                        ConstantValue::int(0, k_v as i32),
                        ConstantValue::int(1, n_v as i32),
                    ],
                    dispatch: DispatchShape {
                        threadgroups: tg,
                        threads_per_threadgroup: tpg,
                    },
                    bindings: affine_qmm_bindings(
                        *in_slot,
                        *out_slot,
                        *layer + layer_offset,
                        *wt_fn,
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
                let kernel = pick_qmm_t_kernel(bucket_m, n_v, k_v, /*B=*/ 1, gs);
                let aligned_n = n_v.is_multiple_of(32);
                match kernel {
                    QmmTKernel::Standard => {
                        let (tg, tpg) =
                            qmm_t_dispatch_shape(kernel, bucket_m, n_v, /*B=*/ 1);
                        LoweredCommand {
                            kernel: KernelId::AffineQmmT,
                            library: "quantized_qmm",
                            function: qmm_t_kernel_static_name(
                                kernel, dtype, scale_dtype, bits_v, gs, aligned_n,
                            ),
                            constants: vec![
                                ConstantValue::int(0, k_v as i32),
                                ConstantValue::int(1, n_v as i32),
                                ConstantValue::int(2, bucket_m as i32),
                            ],
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                            },
                            bindings: affine_qmm_bindings(
                                *in_slot,
                                *out_slot,
                                *layer + layer_offset,
                                *wt_fn,
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
                            constants: vec![
                                ConstantValue::int(0, k_v as i32),
                                ConstantValue::int(1, n_v as i32),
                                ConstantValue::int(2, bucket_m as i32),
                                ConstantValue::int(3, k_partition_size as i32),
                            ],
                            dispatch: DispatchShape {
                                threadgroups: tg,
                                threads_per_threadgroup: tpg,
                            },
                            bindings: affine_qmm_splitk_bindings(
                                *in_slot,
                                *layer + layer_offset,
                                *wt_fn,
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
                            constants: vec![
                                ConstantValue::uint(0, bucket_m),
                                ConstantValue::uint(1, n_v),
                                ConstantValue::uint(2, split_k),
                            ],
                            dispatch: DispatchShape::dispatch_1d(nthreads, THREADS_PER_GROUP),
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
                constants: vec![ConstantValue::uint(0, n)],
                // 1D dispatch over M * intermediate_size output elements,
                // one thread per element. Threadgroup width clamped to
                // the pipeline's max at execute time would be cleaner;
                // for now match the elementwise convention used by
                // `KernelId::Add` / `KernelId::ScalarMul`.
                dispatch: DispatchShape::dispatch_1d(n, THREADS_PER_GROUP),
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
        I::AffineEmbed(out_slot, wt_fn, group_size, bits) => {
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
                constants: vec![ConstantValue::uint(0, hidden_size)],
                dispatch: DispatchShape {
                    threadgroups: (groups_x, bucket_m, 1),
                    threads_per_threadgroup: (THREADS_PER_GROUP, 1, 1),
                },
                bindings: vec![
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding(*wt_fn),
                        which: WeightTensor::Weight,
                        layer: 0,
                        binding_index: 0,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding(*wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: 0,
                        binding_index: 1,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::AffineQuantEmbedding(*wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: 0,
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
        I::FusedGateUpSiluMul(in_slot, out_slot, layer, wt_fn) => {
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
            let constants = if bucket_m == 1 {
                vec![
                    ConstantValue::uint(3, bucket_m),
                    ConstantValue::uint(4, W::INTERMEDIATE_SIZE as u32),
                    ConstantValue::uint(5, W::Q_SIZE as u32),
                ]
            } else {
                vec![
                    ConstantValue::uint(6, bucket_m),
                    ConstantValue::uint(7, W::INTERMEDIATE_SIZE as u32),
                    ConstantValue::uint(8, W::Q_SIZE as u32),
                ]
            };
            LoweredCommand {
                kernel: KernelId::FusedGateUpSiluMul,
                library: "fused_gate_up_silu_mul",
                function,
                constants,
                dispatch: DispatchShape {
                    threadgroups,
                    threads_per_threadgroup,
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
                        kind: WeightBundleKind::LinearLayer(*wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
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
            cos_sin_fn,
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
                constants: vec![
                    ConstantValue::uint(0, W::HEAD_DIM),
                    ConstantValue::uint(1, W::NUM_Q_HEADS),
                    ConstantValue::uint(2, W::NUM_KV_HEADS),
                    ConstantValue::uint(3, W::ROT_DIM),
                    ConstantValue::uint(4, W::BLOCK_SIZE),
                ],
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (W::HEAD_DIM, 1, 1),
                },
                bindings: vec![
                    // Q output (rotated)
                    Binding::ArenaSlot {
                        slot: *q_out_slot,
                        binding_index: 0,
                    },
                    // K output buffer (also written to KV cache)
                    Binding::ArenaSlot {
                        slot: *k_out_slot,
                        binding_index: 1,
                    },
                    // V output buffer (also written to KV cache)
                    Binding::ArenaSlot {
                        slot: *v_out_slot,
                        binding_index: 2,
                    },
                    // RoPE cos/sin table for this layer
                    Binding::Weight {
                        kind: WeightBundleKind::CosSin(*cos_sin_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 3,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::Positions,
                        binding_index: 4,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::SlotMapping,
                        binding_index: 5,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheK {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 6,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 7,
                    },
                ],
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
            wt_fn,
            cos_sin_fn,
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
                constants: vec![
                    ConstantValue::uint(0, W::Q_SIZE as u32),
                    ConstantValue::uint(1, W::NUM_Q_HEADS),
                    ConstantValue::uint(2, W::NUM_KV_HEADS),
                    ConstantValue::uint(3, W::HEAD_DIM),
                    ConstantValue::uint(4, W::ROT_DIM),
                    ConstantValue::uint(5, W::BLOCK_SIZE),
                    ConstantValue::uint(6, bucket_m),
                ],
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_heads_total, 1),
                    threads_per_threadgroup: (W::HEAD_DIM, 1, 1),
                },
                bindings: vec![
                    // 0: q_out
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    // 1: input
                    Binding::ArenaSlot {
                        slot: *in_slot,
                        binding_index: 1,
                    },
                    // 2: packed [Q|K|V] weight
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 2,
                    },
                    // 3: cos_sin table
                    Binding::Weight {
                        kind: WeightBundleKind::CosSin(*cos_sin_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 3,
                    },
                    // 4: positions
                    Binding::Runtime {
                        kind: RuntimeBindingKind::Positions,
                        binding_index: 4,
                    },
                    // 5: slot_mapping
                    Binding::Runtime {
                        kind: RuntimeBindingKind::SlotMapping,
                        binding_index: 5,
                    },
                    // 6: kv_cache_k
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheK {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 6,
                    },
                    // 7: kv_cache_v
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 7,
                    },
                ],
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
            q_wt_fn,
            k_wt_fn,
            v_wt_fn,
            rms_wt_fn,
            cos_sin_fn,
            group_size,
            bits,
            symbol,
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
                constants: vec![ConstantValue::uint(0, bucket_m)],
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_heads_total, 1),
                    threads_per_threadgroup: (threads_per_tg, 1, 1),
                },
                bindings: vec![
                    // 0: q_out
                    Binding::ArenaSlot { slot: *q_out_slot, binding_index: 0 },
                    // 1: residual_io (read+write)
                    Binding::ArenaSlot { slot: *residual_slot, binding_index: 1 },
                    // 2: delta (read)
                    Binding::ArenaSlot { slot: *delta_slot, binding_index: 2 },
                    // 3: rms_weight
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm(*rms_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 3,
                    },
                    // 4..6: Q weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*q_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*q_wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: *layer + layer_offset,
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*q_wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: *layer + layer_offset,
                        binding_index: 6,
                    },
                    // 7..9: K weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*k_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 7,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*k_wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: *layer + layer_offset,
                        binding_index: 8,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*k_wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: *layer + layer_offset,
                        binding_index: 9,
                    },
                    // 10..12: V weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*v_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 10,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*v_wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: *layer + layer_offset,
                        binding_index: 11,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*v_wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: *layer + layer_offset,
                        binding_index: 12,
                    },
                    // 13: cos_sin
                    Binding::Weight {
                        kind: WeightBundleKind::CosSin(*cos_sin_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
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
                        kind: RuntimeBindingKind::KvCacheK { layer: *layer + layer_offset },
                        binding_index: 16,
                    },
                    // 17: kv_cache_v
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV { layer: *layer + layer_offset },
                        binding_index: 17,
                    },
                ],
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
            gate_wt_fn,
            up_wt_fn,
            rms_wt_fn,
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
                constants: vec![ConstantValue::uint(0, bucket_m)],
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, num_tiles, 1),
                    threads_per_threadgroup: (threads_per_tg, 1, 1),
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
                        kind: WeightBundleKind::RmsNorm(*rms_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 3,
                    },
                    // 4..6: gate weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*gate_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*gate_wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: *layer + layer_offset,
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*gate_wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: *layer + layer_offset,
                        binding_index: 6,
                    },
                    // 7..9: up weight + scales + biases
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*up_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 7,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*up_wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: *layer + layer_offset,
                        binding_index: 8,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*up_wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: *layer + layer_offset,
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
            gate_wt_fn,
            up_wt_fn,
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
                constants: vec![ConstantValue::uint(0, bucket_m)],
                dispatch: DispatchShape {
                    threadgroups: (intermediate.div_ceil(tg_n), bucket_m.div_ceil(tg_m), 1),
                    threads_per_threadgroup: (128, 1, 1),
                },
                bindings: vec![
                    Binding::ArenaSlot { slot: *out_slot,    binding_index: 0 },
                    Binding::ArenaSlot { slot: *x_norm_slot, binding_index: 1 },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*gate_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 2,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*gate_wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: *layer + layer_offset,
                        binding_index: 3,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*gate_wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: *layer + layer_offset,
                        binding_index: 4,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*up_wt_fn),
                        which: WeightTensor::Weight,
                        layer: *layer + layer_offset,
                        binding_index: 5,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*up_wt_fn),
                        which: WeightTensor::AffineScales,
                        layer: *layer + layer_offset,
                        binding_index: 6,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::LinearLayer(*up_wt_fn),
                        which: WeightTensor::AffineBiases,
                        layer: *layer + layer_offset,
                        binding_index: 7,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── Decode-bucket attention (single query token / seq) ─────
        I::AttentionViaCache(q_slot, out_slot, layer, _cos_sin_fn, _is_decode) => {
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
                constants: vec![
                    ConstantValue::uint(0, W::HEAD_DIM),
                    ConstantValue::uint(1, W::NUM_Q_HEADS),
                    ConstantValue::uint(2, W::NUM_KV_HEADS),
                    ConstantValue::float(3, W::ATTN_SCALE),
                    ConstantValue::uint(4, W::BLOCK_SIZE),
                    ConstantValue::uint(5, W::MAX_BLOCKS_PER_SEQ),
                ],
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (1024, 1, 1),
                },
                bindings: vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *q_slot,
                        binding_index: 1,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::SeqUsedK,
                        binding_index: 2,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::BlockTable,
                        binding_index: 3,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheK {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 4,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 5,
                    },
                ],
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
            let n_q_heads = W::NUM_Q_HEADS;
            LoweredCommand {
                kernel: KernelId::AttentionPrefillSdpaPaged,
                library: "attention",
                function: pick_specialized_symbol(
                    "attention_prefill_sdpa_v2_paged_f16_specialized",
                    "attention_prefill_sdpa_v2_paged_bf16_specialized",
                    W::METAL_DTYPE,
                ),
                // Same 0..5 layout as `AttentionViaCache` (the decode
                // kernel sibling). The prefill variant differs only in
                // dispatch and per-Q causal-mask bookkeeping; constants
                // are identical.
                constants: vec![
                    ConstantValue::uint(0, W::HEAD_DIM),
                    ConstantValue::uint(1, W::NUM_Q_HEADS),
                    ConstantValue::uint(2, W::NUM_KV_HEADS),
                    ConstantValue::float(3, W::ATTN_SCALE),
                    ConstantValue::uint(4, W::BLOCK_SIZE),
                    ConstantValue::uint(5, W::MAX_BLOCKS_PER_SEQ),
                ],
                dispatch: DispatchShape {
                    threadgroups: (n_q_heads, bucket_m, 1),
                    threads_per_threadgroup: (1024, 1, 1),
                },
                bindings: vec![
                    Binding::ArenaSlot {
                        slot: *out_slot,
                        binding_index: 0,
                    },
                    Binding::ArenaSlot {
                        slot: *q_slot,
                        binding_index: 1,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::CuSeqlensQ,
                        binding_index: 2,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::SeqUsedK,
                        binding_index: 3,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::BlockTable,
                        binding_index: 4,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheK {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 5,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV {
                            layer: *layer + layer_offset,
                        },
                        binding_index: 6,
                    },
                ],
                gemm_dims: None,
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
            dispatch: DispatchShape::dispatch_1d(bucket_m * W::Q_SIZE as u32, THREADS_PER_GROUP),
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
            dispatch: DispatchShape::dispatch_1d(bucket_m * W::Q_SIZE as u32, THREADS_PER_GROUP),
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
fn affine_qmm_bindings<W: CanonicalParams>(
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    wt_fn: crate::WtFn<W, ferrite_kernels::layers::LinearLayer>,
) -> Vec<Binding<W>> {
    vec![
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer(wt_fn),
            which: WeightTensor::Weight,
            layer,
            binding_index: 0,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer(wt_fn),
            which: WeightTensor::AffineScales,
            layer,
            binding_index: 1,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer(wt_fn),
            which: WeightTensor::AffineBiases,
            layer,
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
fn affine_qmm_splitk_bindings<W: CanonicalParams>(
    in_slot: u32,
    layer: u32,
    wt_fn: crate::WtFn<W, ferrite_kernels::layers::LinearLayer>,
) -> Vec<Binding<W>> {
    vec![
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer(wt_fn),
            which: WeightTensor::Weight,
            layer,
            binding_index: 0,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer(wt_fn),
            which: WeightTensor::AffineScales,
            layer,
            binding_index: 1,
        },
        Binding::Weight {
            kind: WeightBundleKind::LinearLayer(wt_fn),
            which: WeightTensor::AffineBiases,
            layer,
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
                assert_eq!(*layer, 8);
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
                kind: WeightBundleKind::AffineQuantEmbedding(_),
                which,
                layer,
                binding_index,
            } => {
                assert_eq!(*which, WeightTensor::Weight);
                // Embed is unlayered — `layer = 0` regardless of layer_offset.
                assert_eq!(*layer, 0);
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
