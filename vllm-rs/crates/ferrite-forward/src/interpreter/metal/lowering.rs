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
                        if let Some(cmd) =
                            lower_one(inst, body_start + offset, bucket_m, iter as u32)?
                        {
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
            function: pick_specialized_symbol(
                "rmsnorm_f16_specialized",
                "rmsnorm_bf16_specialized",
                W::METAL_DTYPE,
            ),
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
                function: pick_specialized_symbol(
                    "fused_add_rmsnorm_f16_specialized",
                    "fused_add_rmsnorm_bf16_specialized",
                    W::METAL_DTYPE,
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
