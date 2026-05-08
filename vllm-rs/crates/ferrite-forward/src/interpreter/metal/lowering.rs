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
//! Coverage (Phase 5.A — TinyLlama-1.1B critical path):
//! `Embed`, `RmsNorm`, `FusedAddRmsNorm`, `Gemm`, `FusedGateUpSiluMul`,
//! `RopeAppend`, `AttentionViaCache`, `AttentionPrefillContiguous`,
//! `Add`, `ScalarMul`, plus `Loop`/`Reshape`/`Alias`/`Free` as
//! structural ops. Every other variant raises
//! [`LoweringError::UnsupportedVariant`] with the variant's type name
//! so the model author knows which arm to add next.

use crate::{CanonicalParams, Instruction};

use super::lowered::{
    Binding, DispatchShape, GemmDims, KernelId, LoweredCommand, LoweredMetalTape, LoweringError,
    RuntimeBindingKind, WeightBundleKind, WeightTensor,
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
    bucket_m: u32,
    num_arena_slots: u32,
) -> Result<LoweredMetalTape<W>, LoweringError> {
    let bb = lower(backbone, bucket_m, num_arena_slots)?;
    let lh = lower(lm_head, bucket_m, num_arena_slots)?;
    let mut commands = bb.commands;
    commands.extend(lh.commands);
    Ok(LoweredMetalTape {
        bucket_m,
        num_arena_slots,
        commands,
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
    bucket_m: u32,
    num_arena_slots: u32,
) -> Result<LoweredMetalTape<W>, LoweringError> {
    let mut commands = Vec::with_capacity(instructions.len());
    let mut i = 0usize;

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
                        if let Some(cmd) = lower_one(
                            inst,
                            body_start + offset,
                            bucket_m,
                            iter as u32,
                        )? {
                            commands.push(cmd);
                        }
                    }
                }
                i = body_end;
            }
            other => {
                if let Some(cmd) = lower_one(other, i, bucket_m, 0)? {
                    commands.push(cmd);
                }
                i += 1;
            }
        }
    }

    Ok(LoweredMetalTape {
        bucket_m,
        num_arena_slots,
        commands,
    })
}

/// Lower one non-`Loop` instruction. Returns `None` for metadata-only
/// instructions (`Reshape`/`Alias`/`Free`) that don't emit a Metal
/// dispatch.
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
fn lower_one<W: CanonicalParams>(
    inst: &Instruction<W>,
    index: usize,
    bucket_m: u32,
    layer_offset: u32,
) -> Result<Option<LoweredCommand<W>>, LoweringError> {
    use Instruction as I;

    let cmd = match inst {
        // ── Token embedding ────────────────────────────────────────
        I::Embed(out_slot, wt_fn) => LoweredCommand {
            kernel: KernelId::Embed,
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

        // ── Fused gate-up SwiGLU MLP (GEMM + SwiGLU in one dispatch) ──
        //
        // Two specialized variants behind `KernelId::FusedGateUpSiluMul`:
        //
        // - **Decode (`bucket_m == 1`)**: dispatches
        //   `fused_gate_up_silu_mul_decode_f16_specialized`, which
        //   uses simd_sum dot products. 256 threads/group =
        //   8 simdgroups × 32 lanes; one simdgroup owns one output.
        //   Threadgroup grid: `(ceil(N/8), 1, 1)` — 8 outputs/group.
        //
        // - **Prefill (`bucket_m >= 2`)**: dispatches
        //   `fused_gate_up_silu_mul_gemm_f16_specialized`, which uses
        //   `simdgroup_matrix<half, 8, 8>` MMA tiles. 32 threads =
        //   one simdgroup per threadgroup, 8x8 output tile.
        //   Threadgroup grid: `(ceil(N/8), ceil(M/8), 1)`.
        //
        // Bindings are identical for both variants: (out, in, weight)
        // at indices 0/1/2. The kernel splits the packed `[gate|up]`
        // weight `[2*N, K]` internally — gate rows [0, N), up rows
        // [N, 2N). The pipeline picker in
        // `interpreter::metal::pipelines::kernel_msl_names` chooses
        // by `bucket_m`.
        I::FusedGateUpSiluMul(in_slot, out_slot, layer, wt_fn) => {
            let inter = W::INTERMEDIATE_SIZE as u32;
            let (threadgroups, threads_per_threadgroup) = if bucket_m == 1 {
                // Decode (MLX gemv port): blockM = BM*SM*TM = 4
                // outputs per threadgroup, 256 threads/group =
                // BN*SN = 8 simdgroups × 32 lanes.
                ((inter.div_ceil(MLP_DECODE_BLOCK_M), 1, 1), (256, 1, 1))
            } else {
                // Prefill: 8x8 output tile, 32 threads.
                let tg_x = inter.div_ceil(MLP_TILE);
                let tg_y = bucket_m.div_ceil(MLP_TILE);
                ((tg_x, tg_y, 1), (32, 1, 1))
            };
            LoweredCommand {
                kernel: KernelId::FusedGateUpSiluMul,
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
                        kind: RuntimeBindingKind::KvCacheK { layer: *layer + layer_offset },
                        binding_index: 6,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV { layer: *layer + layer_offset },
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
            // loop. See `attention_via_cache_v2_f16_specialized` in
            // `attention.metal`.
            let n_q_heads = W::NUM_Q_HEADS;
            LoweredCommand {
                kernel: KernelId::AttentionViaCache,
                dispatch: DispatchShape {
                    threadgroups: (bucket_m, n_q_heads, 1),
                    threads_per_threadgroup: (W::HEAD_DIM, 1, 1),
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
                        kind: RuntimeBindingKind::KvCacheK { layer: *layer + layer_offset },
                        binding_index: 4,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::KvCacheV { layer: *layer + layer_offset },
                        binding_index: 5,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── Prefill-bucket attention (contiguous Q/K/V tiles) ──────
        I::AttentionPrefillContiguous(q_slot, k_slot, v_slot, out_slot, _is_causal) => {
            let n_q_heads = W::NUM_Q_HEADS;
            LoweredCommand {
                kernel: KernelId::AttentionPrefillContiguous,
                dispatch: DispatchShape {
                    threadgroups: (bucket_m.div_ceil(PREFILL_TILE_Q), n_q_heads, 1),
                    threads_per_threadgroup: (W::HEAD_DIM, 1, 1),
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
                    Binding::ArenaSlot {
                        slot: *k_slot,
                        binding_index: 2,
                    },
                    Binding::ArenaSlot {
                        slot: *v_slot,
                        binding_index: 3,
                    },
                    Binding::Runtime {
                        kind: RuntimeBindingKind::CuSeqlensQ,
                        binding_index: 4,
                    },
                ],
                gemm_dims: None,
            }
        }

        // ── Plain residual add ─────────────────────────────────────
        I::Add(delta_slot, residual_slot) => LoweredCommand {
            kernel: KernelId::Add,
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
            return Ok(None);
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

    Ok(Some(cmd))
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

/// Prefill attention Q-axis tile. Each threadgroup handles
/// `PREFILL_TILE_Q` query tokens for one head; chosen to fit the
/// hand-rolled tile shader's K-axis budget.
const PREFILL_TILE_Q: u32 = 16;

/// Output tile dim for the fused MLP kernel
/// (`fused_gate_up_silu_mul_gemm_f16_specialized`). One simdgroup
/// per threadgroup writes an 8×8 output tile. Match the shader's
/// `TILE` constant.
const MLP_TILE: u32 = 8;

/// Output rows per threadgroup for the M=1 decode variant
/// (`fused_gate_up_silu_mul_decode_f16_specialized`). Matches the
/// MLX gemv port's `blockM = BM*SM*TM = 4`.
const MLP_DECODE_BLOCK_M: u32 = 4;
