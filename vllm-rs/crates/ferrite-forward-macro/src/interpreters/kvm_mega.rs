// SPDX-License-Identifier: Apache-2.0
//! KVM megakernel emitter (Phase 2).
//!
//! Vendored `~/Megakernels/demos/cross-gpu-llama` template
//! instantiated per arch. Warp specialization (controller / loader
//! / consumers / storer / launcher), instruction pipelining, page
//! virtual memory, per-SM instruction tape — see
//! `third_party/megakernels/include/controller/instruction_fetch.cuh`.
//! Where the perf comes from. Builds on Phase 1's host /
//! DeviceCallable infra; adds KVM-specific authoring.
//!
//! ── Encoder ─────────────────────────────────────────────────────
//!
//! [`try_encode_bucket`] takes the same `Vec<OpInstance>` the host
//! and prim_mega emitters consume and produces a [`KvmEncodedBucket`]
//! — a flat `Vec<KvmEncodedRow>`. The closed match per
//! `Instruction<W>` variant returns `None` for any kvm-ineligible
//! variant, which marks the canonical kvm-ineligible at codegen
//! time (no `_` catch-all, no runtime "refused" returns —
//! `feedback_no_refusal_chasing`).
//!
//! Row layout pinned against vendor source. See `KVM_MAPPING.md`
//! at the worktree root for the full bidirectional table; the
//! per-opcode `parsed_instruction` reader in
//! `third_party/megakernels/demos/cross-gpu-llama/*.cu` is the
//! ground truth for which i32 slots carry which fields.
//!
//! ── Phase state machine (Q4 of KVM_MAPPING.md) ──────────────────
//!
//! `CutlassGemmAdd` maps to either `OPCODE_O_ProjResidual` (5) or
//! `OPCODE_DownProjResidual` (9) depending on tape position; the
//! IR variant alone doesn't say. `RmsNorm` likewise spreads across
//! `OPCODE_AttnNorm` (1), `OPCODE_MlpNorm` (6), and
//! `OPCODE_LM_HeadNorm` (10).
//!
//! After loop unrolling the IR has exactly `2 * num_layers + 1`
//! `RmsNorm` instances and `2 * num_layers` `CutlassGemmAdd`
//! instances in fixed alternation:
//!
//!   per layer L: AttnNorm[L] · QKV · attn · O_Proj[L] ·
//!                MlpNorm[L] · GateUp · DownProj[L]
//!   final:       LM_HeadNorm · LM_Head
//!
//! The encoder counts norms / gemmadds upfront, then assigns each
//! position structurally: norm #0 = AttnNorm[0], #1 = MlpNorm[0],
//! #2 = AttnNorm[1], …, #(2L) = LM_HeadNorm; gemmadd #(2k) =
//! O_Proj[k], #(2k+1) = DownProj[k]. No mid-encode peek-ahead.

#![allow(dead_code)]

use proc_macro2::TokenStream;
use syn::Lit;

use crate::impl_lib::OpInstance;
use crate::interpreters::host::ArchOpcodes;

// ── Opcode constants (mirror llama.cuh exactly) ────────────────

pub const OPCODE_ATTN_NORM: i32 = 1;
pub const OPCODE_QKV_ROPE_APPEND: i32 = 2;
pub const OPCODE_GQA_ATTENTION_PREFILL: i32 = 3;
pub const OPCODE_GQA_ATTENTION_DECODE: i32 = 4;
pub const OPCODE_O_PROJ_RESIDUAL: i32 = 5;
pub const OPCODE_MLP_NORM: i32 = 6;
pub const OPCODE_GATE_SILU: i32 = 7;
pub const OPCODE_UP_MATMUL: i32 = 8;
pub const OPCODE_DOWN_PROJ_RESIDUAL: i32 = 9;
pub const OPCODE_LM_HEAD_NORM: i32 = 10;
pub const OPCODE_LM_HEAD: i32 = 11;
// 12 (`OPCODE_Barrier_Inc`) and 13 (`OPCODE_AllDeviceBarrier`)
// are multi-GPU only; not emitted in P2 single-GPU bring-up.

/// Pinned to vendor's `base_llama_config::INSTRUCTION_WIDTH = 32`
/// (`llama.cuh:84`). Each row is `[i32; 32]`; trailing slots are
/// zero-padded by the emitter.
pub const INSTRUCTION_WIDTH: usize = 32;

// ── Row + bucket types ──────────────────────────────────────────

/// One TK-shape `[i32; 32]` row. `payload[0]` is the opcode;
/// `payload[1..]` is opcode-specific (see per-opcode encoders).
/// Trailing zeros are valid (the kernel's `parsed_instruction`
/// readers only touch the slots they care about).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvmEncodedRow {
    pub payload: [i32; INSTRUCTION_WIDTH],
}

impl KvmEncodedRow {
    /// New row with `opcode` in slot 0 and the rest zero. Per-arm
    /// encoders fill `payload[1..]` directly.
    pub fn new(opcode: i32) -> Self {
        let mut payload = [0i32; INSTRUCTION_WIDTH];
        payload[0] = opcode;
        Self { payload }
    }

    /// Opcode in slot 0 (the kernel's `instruction[0]` read).
    pub fn opcode(&self) -> i32 {
        self.payload[0]
    }
}

/// Encoded TK throughput tape — flat list of fixed-shape rows.
/// No pointer plan, no runtime-fill list: the kernel reads pointers
/// from `globals_t` (`llama.cuh:158-294`) populated by the launcher
/// at call time, not from per-row pt[]. That's the structural
/// difference from prim_mega's bucket and why this type is
/// genuinely simpler.
#[derive(Clone, Debug)]
pub struct KvmEncodedBucket {
    pub rows: Vec<KvmEncodedRow>,
}

/// Codegen-time inputs every encoder run needs that don't live on
/// `OpInstance`. Resolved by the caller from the workload point +
/// per-canonical [`CanonicalParams`] consts (mirrors prim_mega's
/// `EncodeCtx`).
#[derive(Clone, Debug)]
pub struct EncodeCtx {
    /// Total decode/prefill tokens for this bucket. Maps to
    /// `globals_t::batch_size` at launch (with padding to a
    /// multiple of `matmul_batch_block_size`).
    pub num_tokens: u32,
    /// `<Weights as CanonicalParams>::HIDDEN_SIZE`.
    pub hidden_size: u32,
    /// Q + 2*KV head dims summed. Used for QKV fan-out.
    pub q_size: u32,
    pub kv_size: u32,
    pub head_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub num_kv_heads: u32,
    /// Padded batch — what the kernel sees as `g.batch_size`. Norm
    /// rows are emitted for every padded position so the per-op
    /// `Bar` counters reach their expected count.
    pub batch_size: u32,
    pub matmul_batch_block_size: u32,
    pub matmul_out_block_size: u32,
}

// ── Phase state machine ─────────────────────────────────────────

/// Where we are in the structural phase pattern of a llama-style
/// forward. Set by structural counters in [`classify_positions`];
/// read by per-arm encoders to choose between AttnNorm/MlpNorm/
/// LM_HeadNorm and O_ProjResidual/DownProjResidual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormRole {
    /// First norm of a layer (post-residual into hidden_states).
    /// Emits `OPCODE_AttnNorm`.
    Attn,
    /// Second norm of a layer (post-O_Proj's residual fold).
    /// Emits `OPCODE_MlpNorm`.
    Mlp,
    /// Final norm after all layers (post-DownProj's residual fold).
    /// Emits `OPCODE_LM_HeadNorm`.
    LmHead,
}

/// Position role for a `CutlassGemmAdd` IR instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmAddRole {
    /// Post-attention projection. Emits `OPCODE_O_ProjResidual`.
    OProj,
    /// Post-MLP down projection. Emits `OPCODE_DownProjResidual`.
    DownProj,
}

/// Per-position role tags. Indexed by the IR-instance position
/// after loop unrolling. Computed once per bucket up front so each
/// arm can read its role without maintaining state through the
/// encode walk. `None` entries are positions that aren't
/// `RmsNorm` / `CutlassGemmAdd` (or roles we don't classify here).
#[derive(Clone, Debug, Default)]
pub struct PositionRoles {
    pub norm: Vec<Option<NormRole>>,
    pub gemm_add: Vec<Option<GemmAddRole>>,
}

/// Walk the unrolled instance list and tag each `RmsNorm` /
/// `CutlassGemmAdd` with its structural role. The DSL invariant
/// (`2*L+1` norms, `2*L` gemmadds in alternation) drives the
/// tagging; returns `None` when the structure deviates so the
/// caller can mark the canonical kvm-ineligible. Codegen
/// integration relies on this being a clean fall-through, not a
/// panic — many real canonicals have shapes that aren't llama-
/// style (e.g., LM_Head-only buckets, MLA decoders, etc.).
pub fn classify_positions(instances: &[OpInstance]) -> Option<PositionRoles> {
    let n = instances.len();
    let mut norm: Vec<Option<NormRole>> = vec![None; n];
    let mut gemm_add: Vec<Option<GemmAddRole>> = vec![None; n];

    let total_norms = instances.iter().filter(|i| i.name == "RmsNorm").count();
    let total_gemm_adds = instances
        .iter()
        .filter(|i| i.name == "CutlassGemmAdd")
        .count();

    if total_norms == 0 && total_gemm_adds == 0 {
        // Trivial bucket (no norms, no residuals). Empty roles
        // leave both Vecs as all-None; harmless.
        return Some(PositionRoles { norm, gemm_add });
    }

    // Invariant: norms = 2*L+1, gemmadds = 2*L for some L >= 0.
    // Derived from the llama-style DSL structure. Anything else
    // is kvm-ineligible.
    if total_norms < 1 || total_norms.is_multiple_of(2) {
        // Need odd count: 2*L + 1 norms (per layer + final lm_head).
        return None;
    }
    let l_from_norms = (total_norms - 1) / 2;
    if total_gemm_adds != 2 * l_from_norms {
        return None;
    }

    let mut norm_idx = 0usize;
    let mut gemm_add_idx = 0usize;
    for (pos, inst) in instances.iter().enumerate() {
        if inst.name == "RmsNorm" {
            let role = if norm_idx == total_norms - 1 {
                NormRole::LmHead
            } else if norm_idx.is_multiple_of(2) {
                NormRole::Attn
            } else {
                NormRole::Mlp
            };
            norm[pos] = Some(role);
            norm_idx += 1;
        } else if inst.name == "CutlassGemmAdd" {
            let role = if gemm_add_idx.is_multiple_of(2) {
                GemmAddRole::OProj
            } else {
                GemmAddRole::DownProj
            };
            gemm_add[pos] = Some(role);
            gemm_add_idx += 1;
        }
    }

    Some(PositionRoles { norm, gemm_add })
}

// ── Encoder driver ──────────────────────────────────────────────

/// Encode a whole bucket. Returns `None` if any instance is
/// kvm-ineligible (no encoder arm, OR a field-extraction failure).
/// Handles `Loop` rows by unrolling at encode time; skips `Free`
/// and `Alias` rows. Mirrors prim_mega's discipline.
///
/// `prefill_seq_info` is the per-call `(q_len, token_offset)` list
/// the prefill attention arm needs to compute its row count and
/// payloads. Pass `None` for a pure-decode bucket; pass
/// `Some(&seq_info)` for a prefill bucket.
pub fn try_encode_bucket(
    arch_opcodes: &ArchOpcodes,
    instances: &[OpInstance],
    ctx: &EncodeCtx,
    prefill_seq_info: Option<&[(usize, usize)]>,
) -> Option<KvmEncodedBucket> {
    // Loop unrolling. Mirrors prim_mega's `try_encode_bucket`
    // structure: walk with explicit cursor so `Loop` can advance
    // past its body. We unroll into a flat `Vec<OpInstance>` so
    // role classification + per-arm encoding both see the post-
    // unroll instance list.
    let mut unrolled: Vec<(OpInstance, u32 /* layer_offset */)> =
        Vec::with_capacity(instances.len());
    let mut i = 0;
    while i < instances.len() {
        let inst = &instances[i];
        match inst.name.to_string().as_str() {
            "Loop" => {
                let count = parse_u32_literal(&inst.field_values[0])?;
                let body_len = parse_u32_literal(&inst.field_values[1])? as usize;
                let body_start = i + 1;
                let body_end = body_start + body_len;
                if body_end > instances.len() {
                    return None;
                }
                for iter in 0..count {
                    for body_inst in &instances[body_start..body_end] {
                        unrolled.push((body_inst.clone(), iter));
                    }
                }
                i = body_end;
            }
            "Free" | "Alias" => {
                // No per-row free/alias in mega — pt[] is fixed
                // for the cooperative launch (and TK reads from
                // globals_t, not pt[]).
                i += 1;
            }
            _ => {
                unrolled.push((inst.clone(), 0));
                i += 1;
            }
        }
    }

    // Drop the layer_offset wrapper for classification (it doesn't
    // depend on per-iter offsets).
    let flat: Vec<OpInstance> = unrolled.iter().map(|(inst, _)| inst.clone()).collect();
    let roles = classify_positions(&flat)?;

    let mut rows: Vec<KvmEncodedRow> = Vec::new();
    let _ = arch_opcodes; // shape registry consumed in P2-2 step 2 (sanity-check upfront)

    for (pos, (inst, layer_offset)) in unrolled.iter().enumerate() {
        let opt_row = encode_op_arm(inst, *layer_offset, &roles, pos, ctx, prefill_seq_info)?;
        if let Some(row_or_rows) = opt_row {
            rows.extend(row_or_rows);
        }
    }

    Some(KvmEncodedBucket { rows })
}

/// Per-variant arm. Returns:
/// - `Some(Some(rows))` — variant encoded successfully (one or
///   more rows; e.g., `FusedGateUpSiluMul` fans out into two
///   opcodes' worth of rows).
/// - `Some(None)` — variant intentionally produces no row
///   (`Embed`: handled pre-megakernel-launch by the launcher).
/// - `None` — variant has no arm (canonical kvm-ineligible) or
///   field extraction failed (codegen invariant violation).
///
/// Closed match: every IR variant must be explicitly named per
/// `feedback_no_refusal_chasing`. Adding a new variant in
/// `instr.rs` shows up as a compile error here, not a silent
/// runtime fall-through.
fn encode_op_arm(
    inst: &OpInstance,
    layer_offset: u32,
    roles: &PositionRoles,
    pos: usize,
    ctx: &EncodeCtx,
    prefill_seq_info: Option<&[(usize, usize)]>,
) -> Option<Option<Vec<KvmEncodedRow>>> {
    match inst.name.to_string().as_str() {
        "RmsNorm" => {
            let role = roles.norm[pos].expect("kvm_mega: RmsNorm position must have a NormRole");
            Some(Some(encode_rms_norm(inst, layer_offset, role, ctx)?))
        }
        "CutlassGemmAdd" => {
            let role = roles.gemm_add[pos]
                .expect("kvm_mega: CutlassGemmAdd position must have a GemmAddRole");
            Some(Some(encode_cutlass_gemm_add(
                inst,
                layer_offset,
                role,
                ctx,
            )?))
        }
        "Embed" => {
            // Embedding gather runs pre-megakernel-launch (vendor's
            // `tp_generate.py` populates `g.hidden_states` before
            // the kernel kicks). Encoder emits no row; the launcher
            // handles it host-side. KVM_MAPPING.md §"Bidirectional
            // table" footnote.
            Some(None)
        }

        "FusedQkvRopeCache" => Some(Some(encode_fused_qkv_rope(
            inst,
            layer_offset,
            ctx,
            /*is_prefill=*/ false,
        )?)),
        "FusedQkvRopePrefill" => Some(Some(encode_fused_qkv_rope(
            inst,
            layer_offset,
            ctx,
            /*is_prefill=*/ true,
        )?)),
        "FusedGateUpSiluMul" => Some(Some(encode_fused_gate_up_silu_mul(
            inst,
            layer_offset,
            ctx,
        )?)),
        "CutlassGemm" => Some(Some(encode_cutlass_gemm_lm_head(inst, layer_offset, ctx)?)),
        "AttentionViaCache" | "FlashInferAttentionDecode" => {
            Some(Some(encode_attention_decode(inst, layer_offset, ctx)?))
        }

        "AttentionPrefillContiguous" | "FlashInferAttentionPrefill" => Some(Some(
            encode_attention_prefill(inst, layer_offset, ctx, prefill_seq_info)?,
        )),

        // ── Kvm-ineligible (no TK opcode) — see KVM_MAPPING.md ──
        //
        // These return `None` (the inner Option), marking the
        // canonical kvm-ineligible. NO `_` catch-all here: every
        // IR variant must be explicitly named so adding a new
        // variant in `instr.rs` shows up as a compile error here.
        "LayerNorm"
        | "Reshape"
        | "Add"
        | "ScalarMul"
        | "TanhSoftCap"
        | "FusedAddRmsNorm"
        | "FusedAddRmsNormWithOffset"
        | "ScalarOffsetRmsNorm"
        | "Gemm"
        | "FusedGemmBias"
        | "FusedGateUpGeluMul"
        | "FusedQkvQkNormRopeCache"
        | "SlidingAttentionViaCache"
        | "SlidingAttentionPrefillContiguous"
        | "RopeAppend"
        | "MlaSplit"
        | "MlaAttention"
        | "DeepSeekMoe"
        | "CutlassGemmSplitK"
        | "CutlassGemv"
        | "CutlassFusedGemmBias"
        | "CutlassFusedGateUpSiluMul"
        | "MarlinGemm"
        | "MarlinFusedGateUpSiluMul"
        | "MarlinFusedGateUpGeluMul"
        | "MarlinFusedQkvRopeCache"
        | "MarlinFusedQkvRopePrefill"
        | "Bnb4Gemm"
        | "Bnb4FusedGateUpSiluMul"
        | "Bnb4FusedGateUpGeluMul"
        | "Bnb4FusedQkvRopeCache"
        | "Bnb4FusedQkvRopePrefill"
        | "Fp8Gemm"
        | "Fp8FusedGemmBias"
        | "Fp8FusedGateUpSiluMul"
        | "Fp8FusedGateUpGeluMul"
        | "Fp8FusedQkvRopeCache"
        | "Fp8FusedQkvRopePrefill" => None,

        other => panic!(
            "kvm_mega encoder: unknown variant `{other}` — add an \
             arm explicitly listing it as kvm-eligible (return \
             Some) or kvm-ineligible (return None). No `_` \
             catch-all by design (`feedback_no_refusal_chasing`)."
        ),
    }
}

// ── Per-variant arms ────────────────────────────────────────────

/// `RmsNorm(in_slot, out_slot, layer, weight_fn)` →
/// `OPCODE_AttnNorm` / `OPCODE_MlpNorm` / `OPCODE_LM_HeadNorm` per
/// [`NormRole`]. Row layout per
/// `third_party/megakernels/demos/cross-gpu-llama/batched_rms_norm.cu:24-46`:
///
/// ```text
/// [opcode, layer_idx, num_items=1, local_batch_idx_0, 0, …, 0]
/// ```
///
/// Fan-out: one row per padded batch position (`ctx.batch_size`
/// rows total). Each row carries `num_items=1` and one batch
/// index; the kernel can batch via `num_items > 1` but we emit
/// the simpler shape. Batch padding must be honored exactly
/// because downstream `Bar` counters depend on it.
///
/// LM_HeadNorm pins `layer_idx = 0` by convention.
fn encode_rms_norm(
    inst: &OpInstance,
    layer_offset: u32,
    role: NormRole,
    ctx: &EncodeCtx,
) -> Option<Vec<KvmEncodedRow>> {
    // RmsNorm(in_slot, out_slot, layer, weight_fn) — see
    // `instr.rs:88`. We don't need in_slot/out_slot/weight_fn
    // here: the TK kernel reads from `g.hidden_states` and writes
    // to `g.rms_*_intermediates` (per-role buffer pinned in the
    // kernel's `rms_op` template specialization at
    // `batched_rms_norm.cu:315-349`). Slot mapping is the
    // launcher's concern (P2-4).
    let _in_slot = parse_u32_literal(&inst.field_values[0])?;
    let _out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = layer_baseline + layer_offset;

    let opcode = match role {
        NormRole::Attn => OPCODE_ATTN_NORM,
        NormRole::Mlp => OPCODE_MLP_NORM,
        NormRole::LmHead => OPCODE_LM_HEAD_NORM,
    };
    let layer_field = match role {
        NormRole::LmHead => 0, // LM_Head runs once at end-of-forward; layer slot is unused
        _ => layer as i32,
    };

    let mut out = Vec::with_capacity(ctx.batch_size as usize);
    for bidx in 0..ctx.batch_size as i32 {
        let mut row = KvmEncodedRow::new(opcode);
        row.payload[1] = layer_field;
        row.payload[2] = 1; // num_items
        row.payload[3] = bidx;
        out.push(row);
    }
    Some(out)
}

/// `CutlassGemmAdd(in_slot, residual_slot, layer, weight_fn,
///                 tile_m, tile_n, stages)` →
/// `OPCODE_O_ProjResidual` / `OPCODE_DownProjResidual` per
/// [`GemmAddRole`]. Row layout per
/// `matmul_adds.cu:18-26`:
///
/// ```text
/// [opcode, layer, local_row, local_col, row, col, 0, …, 0]
/// ```
///
/// Fan-out: `num_batch_blocks × num_output_blocks` rows. For
/// single-GPU `local_row == row` and `local_col == col`. Tile
/// hint fields (`tile_m`/`tile_n`/`stages`) are CUTLASS-shape
/// metadata not used by the TK matmul (TK has its own block-shape
/// statics in `matmul_pipeline.cuh`); the encoder reads them only
/// to keep the IR-field-extraction signature parallel to the
/// host side.
fn encode_cutlass_gemm_add(
    inst: &OpInstance,
    layer_offset: u32,
    role: GemmAddRole,
    ctx: &EncodeCtx,
) -> Option<Vec<KvmEncodedRow>> {
    let _in_slot = parse_u32_literal(&inst.field_values[0])?;
    let _residual_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = (layer_baseline + layer_offset) as i32;
    // weight_fn (field_values[3]) read in P2-4 launcher; tile
    // hints (4..6) read solely for parity with host's parse —
    // TK matmul block shape comes from `matmul_pipeline.cuh`
    // statics, not the row.
    let _tile_m = parse_u32_literal(&inst.field_values[4])?;
    let _tile_n = parse_u32_literal(&inst.field_values[5])?;
    let _stages = parse_u32_literal(&inst.field_values[6])?;

    let opcode = match role {
        GemmAddRole::OProj => OPCODE_O_PROJ_RESIDUAL,
        GemmAddRole::DownProj => OPCODE_DOWN_PROJ_RESIDUAL,
    };

    // hidden_dim / matmul_out_block_size — same on both ends
    // (o_proj output is `hidden_dim`, down_proj output is also
    // `hidden_dim`).
    let num_batch_blocks = (ctx.batch_size / ctx.matmul_batch_block_size) as i32;
    let num_output_blocks = (ctx.hidden_size / ctx.matmul_out_block_size) as i32;

    let mut out = Vec::with_capacity((num_batch_blocks * num_output_blocks) as usize);
    for batch_block in 0..num_batch_blocks {
        for out_block in 0..num_output_blocks {
            let mut row = KvmEncodedRow::new(opcode);
            row.payload[1] = layer;
            row.payload[2] = batch_block; // local_row
            row.payload[3] = out_block; // local_col
            row.payload[4] = batch_block; // row (single-GPU: equal)
            row.payload[5] = out_block; // col
            out.push(row);
        }
    }
    Some(out)
}

/// `FusedQkvRopeCache(in, out, layer, weight_fn, cos_sin_fn,
/// biased, interleaved)` and `FusedQkvRopePrefill(in, out, layer,
/// prefill_qo_indptr, prefill_kv_indptr, weight_fn, cos_sin_fn,
/// biased, interleaved)` both → `OPCODE_QKV_RopeAppend`. Row
/// layout per `qkv_rope_append.cu:32-39`:
///
/// ```text
/// [opcode, layer, local_row, local_col, row, col, 0, …, 0]
/// ```
///
/// Fan-out: `num_batch_blocks × num_qkv_blocks`, where
/// `qkv_dim = q_size + 2 * kv_size`. The kernel toggles between
/// prefill and decode at runtime via `g.num_prefill_tokens`
/// (KVM_MAPPING.md Q2) — encoder emits the same row layout for
/// both variants. `is_prefill` is recorded for documentation
/// only; the kernel doesn't read it from the row.
fn encode_fused_qkv_rope(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
    is_prefill: bool,
) -> Option<Vec<KvmEncodedRow>> {
    let _in_slot = parse_u32_literal(&inst.field_values[0])?;
    let _out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = (layer_baseline + layer_offset) as i32;
    // `FusedQkvRopePrefill` carries two extra slot fields at
    // positions 3 + 4 (prefill_qo_indptr / prefill_kv_indptr).
    // The kernel reads those tensors from `globals_t` directly
    // (`g.prefill_qo_indptr` etc.), not from the row payload, so
    // the encoder discards them. Validating the field-extraction
    // would catch DSL drift but adds nothing for kvm correctness.

    let qkv_dim = ctx.q_size + 2 * ctx.kv_size;
    let num_batch_blocks = (ctx.batch_size / ctx.matmul_batch_block_size) as i32;
    let num_qkv_blocks = (qkv_dim / ctx.matmul_out_block_size) as i32;

    let mut out = Vec::with_capacity((num_batch_blocks * num_qkv_blocks) as usize);
    for batch_block in 0..num_batch_blocks {
        for qkv_block in 0..num_qkv_blocks {
            let mut row = KvmEncodedRow::new(OPCODE_QKV_ROPE_APPEND);
            row.payload[1] = layer;
            row.payload[2] = batch_block; // local_row
            row.payload[3] = qkv_block; // local_col
            row.payload[4] = batch_block; // row
            row.payload[5] = qkv_block; // col
            out.push(row);
        }
    }
    let _ = is_prefill; // toggled at launch via g.num_prefill_tokens
    Some(out)
}

/// `FusedGateUpSiluMul(in, out, layer, weight_fn)` → split into
/// **two** TK opcodes' worth of rows: `OPCODE_GateSiLU` (gate
/// projection + SiLU activation) followed by `OPCODE_UpMatmul`
/// (up projection + elementwise mul into silu_out).
///
/// Row layout per `gate_silu.cu:22-28` and `up_matmul.cu:28-34`:
///
/// ```text
/// [opcode, layer, local_row, local_col, row, col, 0, …, 0]
/// ```
///
/// Fan-out per opcode:
/// `num_batch_blocks × num_intermediate_blocks`. Total rows = 2×.
/// Gate rows precede up rows in the tape — that's the order TK's
/// `Bar`-counter chain expects (UpMatmul's loader spin-waits on
/// the GateSiLU `Bar`; see `up_matmul.cu` gmem_waiter pattern).
fn encode_fused_gate_up_silu_mul(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<Vec<KvmEncodedRow>> {
    let _in_slot = parse_u32_literal(&inst.field_values[0])?;
    let _out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = (layer_baseline + layer_offset) as i32;

    let num_batch_blocks = (ctx.batch_size / ctx.matmul_batch_block_size) as i32;
    let num_intermediate_blocks = (ctx.intermediate_size / ctx.matmul_out_block_size) as i32;

    let total = 2 * (num_batch_blocks * num_intermediate_blocks) as usize;
    let mut out = Vec::with_capacity(total);
    for opcode in [OPCODE_GATE_SILU, OPCODE_UP_MATMUL] {
        for batch_block in 0..num_batch_blocks {
            for block in 0..num_intermediate_blocks {
                let mut row = KvmEncodedRow::new(opcode);
                row.payload[1] = layer;
                row.payload[2] = batch_block;
                row.payload[3] = block;
                row.payload[4] = batch_block;
                row.payload[5] = block;
                out.push(row);
            }
        }
    }
    Some(out)
}

/// `CutlassGemm(in, out, layer, weight_fn, tile_m, tile_n,
/// stages)` at the LM_Head position → `OPCODE_LM_Head`. Row
/// layout per `lm_head.cu:21-27`:
///
/// ```text
/// [opcode, layer=0, local_row, local_col, row, col, 0, …, 0]
/// ```
///
/// Fan-out: `num_batch_blocks × num_logit_blocks`, where
/// `num_logit_blocks = vocab_size / matmul_out_block_size`. By
/// convention slot 1 (`layer`) is `0` — LM_Head runs once at
/// end-of-forward; the kernel doesn't read this slot.
///
/// Why this arm is "LM_Head only" without an explicit role tag:
/// every other `Gemm` in a llama-style DSL gets claimed by a
/// fusion Impl (`FusedQkvRopeCache` swallows q/k/v projections;
/// `FusedGateUpSiluMul` swallows gate/up; `CutlassGemmAddImpl`
/// swallows o_proj+add and down_proj+add). After picks the only
/// `CutlassGemm` instance left in the unrolled IR is the LM_Head
/// one. Caveat: if the DSL gains a non-fusable Gemm, we want this
/// to fail loudly — see assertion below.
fn encode_cutlass_gemm_lm_head(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<Vec<KvmEncodedRow>> {
    let _in_slot = parse_u32_literal(&inst.field_values[0])?;
    let _out_slot = parse_u32_literal(&inst.field_values[1])?;
    // Layer field is unused by the kernel for LM_Head; we still
    // parse for IR-shape validity. layer_offset zero-base for the
    // single LM_Head call (post-loop).
    let _layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let _ = layer_offset;
    let _tile_m = parse_u32_literal(&inst.field_values[4])?;
    let _tile_n = parse_u32_literal(&inst.field_values[5])?;
    let _stages = parse_u32_literal(&inst.field_values[6])?;

    let num_batch_blocks = (ctx.batch_size / ctx.matmul_batch_block_size) as i32;
    let num_logit_blocks = (ctx.vocab_size / ctx.matmul_out_block_size) as i32;

    let mut out = Vec::with_capacity((num_batch_blocks * num_logit_blocks) as usize);
    for batch_block in 0..num_batch_blocks {
        for logit_block in 0..num_logit_blocks {
            let mut row = KvmEncodedRow::new(OPCODE_LM_HEAD);
            row.payload[1] = 0; // layer slot unused by lm_head kernel
            row.payload[2] = batch_block;
            row.payload[3] = logit_block;
            row.payload[4] = batch_block;
            row.payload[5] = logit_block;
            out.push(row);
        }
    }
    Some(out)
}

/// `AttentionViaCache(in, out, layer, cos_sin_fn, biased)` and
/// `FlashInferAttentionDecode(in, out, layer, cos_sin_fn, ?, ?)`
/// → `OPCODE_GQA_AttentionDecode`. Variable-length payload per
/// `attention_decode.cu:70-81`:
///
/// ```text
/// [opcode, layer, num_entries=2*num_pairs,
///  seq_0, kv_0, seq_1, kv_1, …]
/// ```
///
/// where each `(seq_idx, kv_head_idx)` pair occupies two i32s
/// after the 3-int header. The kernel reads
/// `s.instruction()[2] / 2` to get the pair count
/// (`attention_decode.cu:71`).
///
/// Row capacity: `(INSTRUCTION_WIDTH - 3) / 2 = 14` pairs per
/// row. Total pairs = `num_tokens × num_kv_heads`; the encoder
/// chunks into 14-pair rows.
fn encode_attention_decode(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<Vec<KvmEncodedRow>> {
    let _in_slot = parse_u32_literal(&inst.field_values[0])?;
    let _out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = (layer_baseline + layer_offset) as i32;

    // FIXME: max-pairs-per-row is unverified. The 32-int row can
    // physically fit `(32 - 3) / 2 = 14` pairs, but vendor's Python
    // scheduler at `Megakernels/megakernels/demos/tp_throughput/
    // scheduler.py` uses `group_size = 8` — likely the kernel's
    // actual semaphore budget. Pin against vendor before this arm
    // is wired into codegen.
    const MAX_PAIRS_PER_INST: usize = (INSTRUCTION_WIDTH - 3) / 2;

    // Build the (seq_idx, kv_head_idx) pair list, then chunk.
    let mut pairs: Vec<(i32, i32)> =
        Vec::with_capacity((ctx.num_tokens * ctx.num_kv_heads) as usize);
    for seq_idx in 0..ctx.num_tokens as i32 {
        for kv_head in 0..ctx.num_kv_heads as i32 {
            pairs.push((seq_idx, kv_head));
        }
    }

    let mut out: Vec<KvmEncodedRow> = Vec::with_capacity(pairs.len().div_ceil(MAX_PAIRS_PER_INST));
    for chunk in pairs.chunks(MAX_PAIRS_PER_INST) {
        let mut row = KvmEncodedRow::new(OPCODE_GQA_ATTENTION_DECODE);
        row.payload[1] = layer;
        row.payload[2] = (chunk.len() * 2) as i32; // num_entries
        for (i, &(seq, kv)) in chunk.iter().enumerate() {
            row.payload[3 + 2 * i] = seq;
            row.payload[3 + 2 * i + 1] = kv;
        }
        out.push(row);
    }
    Some(out)
}

/// `AttentionPrefillContiguous(in, out, layer, ?, ?)` and
/// `FlashInferAttentionPrefill(in, out, layer, ?, ?, ?, ?)`
/// → `OPCODE_GQA_AttentionPrefill`. Fixed 6-int payload per
/// `attention_prefill.cu:42-46`:
///
/// ```text
/// [opcode, layer, seq_idx, prefill_block_idx,
///  prefill_token_offset, kv_head_idx, 0, …, 0]
/// ```
///
/// One row per `(seq_idx, q_block_of_16, kv_head_idx)` triple.
/// Loop order: outer `seq_idx` → `kv_head` → `q_block`.
///
/// ### `prefill_seq_info`
///
/// `prefill_seq_info[seq_idx] = (q_len, token_offset)` —
/// `token_offset = seqused_k - num_q_tokens` for that seq
/// (post-history-append). This data is per-call.
///
/// `None` → return an empty Vec (the canonical is structurally
/// kvm-eligible; row construction is deferred). `Some(&[])` →
/// also empty Vec (no sequences to encode).
fn encode_attention_prefill(
    inst: &OpInstance,
    layer_offset: u32,
    _ctx: &EncodeCtx,
    prefill_seq_info: Option<&[(usize, usize)]>,
) -> Option<Vec<KvmEncodedRow>> {
    let _in_slot = parse_u32_literal(&inst.field_values[0])?;
    let _out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = (layer_baseline + layer_offset) as i32;

    let Some(seq_info) = prefill_seq_info else {
        // No SeqInfo at codegen time — structural eligibility
        // confirmed; rows built at runtime by the launcher. Return
        // empty Vec (NOT `None`, which would mark this canonical
        // kvm-ineligible).
        return Some(Vec::new());
    };

    // num_kv_heads is on EncodeCtx but the decode arm threads it via
    // the same struct; mirror that.
    let num_kv_heads = _ctx.num_kv_heads as usize;

    let mut out = Vec::new();
    for (seq_idx, &(q_len, token_offset)) in seq_info.iter().enumerate() {
        let num_q_blocks = q_len.div_ceil(16);
        for kv_head in 0..num_kv_heads {
            for q_block in 0..num_q_blocks {
                let mut row = KvmEncodedRow::new(OPCODE_GQA_ATTENTION_PREFILL);
                row.payload[1] = layer;
                row.payload[2] = seq_idx as i32;
                row.payload[3] = q_block as i32;
                row.payload[4] = token_offset as i32;
                row.payload[5] = kv_head as i32;
                out.push(row);
            }
        }
    }
    Some(out)
}

// ── Program static emission ────────────────────────────────────

/// Emit a `static <ident>: [[i32; 32]; N] = [...]` from an
/// already-resolved [`KvmEncodedBucket`]. The C++ kernel reads
/// instruction rows out of this static at runtime via the
/// launcher's `cudaMemcpyAsync` to a device tape buffer.
pub fn emit_kvm_program(
    static_ident: &syn::Ident,
    bucket: &KvmEncodedBucket,
) -> proc_macro2::TokenStream {
    use quote::quote;
    let n = bucket.rows.len();
    let rows = bucket.rows.iter().map(|row| {
        let cells = row.payload.iter().map(|v| {
            let lit = proc_macro2::Literal::i32_unsuffixed(*v);
            quote! { #lit }
        });
        quote! { [ #(#cells),* ] }
    });
    let width_lit = proc_macro2::Literal::usize_unsuffixed(INSTRUCTION_WIDTH);
    let n_lit = proc_macro2::Literal::usize_unsuffixed(n);
    quote! {
        #[cfg(feature = "cuda")]
        #[allow(dead_code)]
        static #static_ident: [[i32; #width_lit]; #n_lit] = [ #(#rows),* ];
    }
}

// ── Helpers (private) ───────────────────────────────────────────

/// Extract a `u32` literal from a TokenStream that's expected to
/// hold one. Mirrors `prim_mega::parse_u32_literal` exactly; small
/// helper duplicated rather than re-exported from prim_mega until
/// a shared utility module emerges naturally.
fn parse_u32_literal(ts: &TokenStream) -> Option<u32> {
    let lit: Lit = syn::parse2(ts.clone()).ok()?;
    match lit {
        Lit::Int(li) => li.base10_parse::<u32>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_lib::{OpInstance, OpcodeShape};
    use proc_macro2::Span;
    use syn::Ident;

    fn op(name: &str, fields: Vec<TokenStream>) -> OpInstance {
        OpInstance::new(Ident::new(name, Span::call_site()), fields)
    }

    /// Llama-3.2-1B-ish: 16 layers × 2k hidden × 8k intermediate.
    /// `batch_size = matmul_batch_block_size` (single block) so
    /// row counts stay readable in tests.
    fn ctx_llama_1b() -> EncodeCtx {
        EncodeCtx {
            num_tokens: 1,
            hidden_size: 2048,
            q_size: 2048,
            kv_size: 512,
            head_size: 64,
            intermediate_size: 8192,
            vocab_size: 128_256,
            num_kv_heads: 8,
            batch_size: 128,
            matmul_batch_block_size: 128,
            matmul_out_block_size: 256,
        }
    }

    fn arch_with_norm_and_gemm_add() -> ArchOpcodes {
        let mut a = ArchOpcodes::new();
        a.register(OpcodeShape::new(
            "RmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
            ],
        ));
        a.register(OpcodeShape::new(
            "CutlassGemmAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
            ],
        ));
        a
    }

    /// Build a u32 token literal in the same form prim_mega's tests
    /// use (`<n>u32`). `quote!` interpolates via `#var` which emits
    /// the unsuffixed integer; we re-parse with the explicit suffix
    /// so `parse_u32_literal` matches.
    fn u32_lit(v: u32) -> TokenStream {
        format!("{v}u32").parse().unwrap()
    }

    fn rms_norm_inst(in_slot: u32, out_slot: u32, layer: u32, weight: &str) -> OpInstance {
        let weight_path: TokenStream = weight.parse().unwrap();
        op(
            "RmsNorm",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                weight_path,
            ],
        )
    }

    fn gemm_add_inst(
        in_slot: u32,
        residual_slot: u32,
        layer: u32,
        weight: &str,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
    ) -> OpInstance {
        let weight_path: TokenStream = weight.parse().unwrap();
        op(
            "CutlassGemmAdd",
            vec![
                u32_lit(in_slot),
                u32_lit(residual_slot),
                u32_lit(layer),
                weight_path,
                u32_lit(tile_m),
                u32_lit(tile_n),
                u32_lit(stages),
            ],
        )
    }

    fn qkv_rope_cache_inst(in_slot: u32, out_slot: u32, layer: u32) -> OpInstance {
        let weight_path: TokenStream = "Weights::self_attn_qkv".parse().unwrap();
        let cos_sin_path: TokenStream = "Weights::rotary".parse().unwrap();
        op(
            "FusedQkvRopeCache",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                weight_path,
                cos_sin_path,
                "false".parse().unwrap(),
                "false".parse().unwrap(),
            ],
        )
    }

    fn qkv_rope_prefill_inst(in_slot: u32, out_slot: u32, layer: u32) -> OpInstance {
        let weight_path: TokenStream = "Weights::self_attn_qkv".parse().unwrap();
        let cos_sin_path: TokenStream = "Weights::rotary".parse().unwrap();
        op(
            "FusedQkvRopePrefill",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                u32_lit(0), // prefill_qo_indptr slot
                u32_lit(0), // prefill_kv_indptr slot
                weight_path,
                cos_sin_path,
                "false".parse().unwrap(),
                "false".parse().unwrap(),
            ],
        )
    }

    fn gate_up_silu_mul_inst(in_slot: u32, out_slot: u32, layer: u32) -> OpInstance {
        let weight_path: TokenStream = "Weights::mlp_gate_up".parse().unwrap();
        op(
            "FusedGateUpSiluMul",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                weight_path,
            ],
        )
    }

    fn attention_via_cache_inst(in_slot: u32, out_slot: u32, layer: u32) -> OpInstance {
        let cos_sin_path: TokenStream = "Weights::rotary".parse().unwrap();
        op(
            "AttentionViaCache",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                cos_sin_path,
                "false".parse().unwrap(), // biased
            ],
        )
    }

    fn flashinfer_attention_decode_inst(in_slot: u32, out_slot: u32, layer: u32) -> OpInstance {
        let cos_sin_path: TokenStream = "Weights::rotary".parse().unwrap();
        op(
            "FlashInferAttentionDecode",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                cos_sin_path,
                u32_lit(0), // window/extra slot
                "false".parse().unwrap(),
            ],
        )
    }

    fn attention_prefill_contiguous_inst(in_slot: u32, out_slot: u32, layer: u32) -> OpInstance {
        op(
            "AttentionPrefillContiguous",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                u32_lit(0),               // ?
                "false".parse().unwrap(), // ?
            ],
        )
    }

    fn flashinfer_attention_prefill_inst(in_slot: u32, out_slot: u32, layer: u32) -> OpInstance {
        op(
            "FlashInferAttentionPrefill",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                u32_lit(0),
                u32_lit(0),
                u32_lit(0),
                "false".parse().unwrap(),
            ],
        )
    }

    fn cutlass_gemm_inst(
        in_slot: u32,
        out_slot: u32,
        layer: u32,
        weight: &str,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
    ) -> OpInstance {
        let weight_path: TokenStream = weight.parse().unwrap();
        op(
            "CutlassGemm",
            vec![
                u32_lit(in_slot),
                u32_lit(out_slot),
                u32_lit(layer),
                weight_path,
                u32_lit(tile_m),
                u32_lit(tile_n),
                u32_lit(stages),
            ],
        )
    }

    /// One layer: AttnNorm · O_Proj · MlpNorm · DownProj · LM_HeadNorm.
    /// Verifies the role classifier tags positions correctly.
    #[test]
    fn classify_one_layer_plus_lm_head() {
        let prog = vec![
            rms_norm_inst(0, 1, 0, "Weights::input_layernorm"),
            gemm_add_inst(2, 3, 0, "Weights::self_attn_o_proj", 128, 128, 3),
            rms_norm_inst(4, 5, 0, "Weights::post_attention_layernorm"),
            gemm_add_inst(6, 7, 0, "Weights::mlp_down_proj", 128, 128, 3),
            rms_norm_inst(8, 9, 0, "Weights::norm"),
        ];
        let roles = classify_positions(&prog).unwrap();
        assert_eq!(roles.norm[0], Some(NormRole::Attn));
        assert_eq!(roles.norm[2], Some(NormRole::Mlp));
        assert_eq!(roles.norm[4], Some(NormRole::LmHead));
        assert_eq!(roles.gemm_add[1], Some(GemmAddRole::OProj));
        assert_eq!(roles.gemm_add[3], Some(GemmAddRole::DownProj));
    }

    /// Two layers + lm head — the alternation must hold across
    /// layer boundaries.
    #[test]
    fn classify_two_layers_alternates() {
        let mut prog: Vec<OpInstance> = Vec::new();
        for layer in 0..2u32 {
            prog.push(rms_norm_inst(0, 1, layer, "Weights::input_layernorm"));
            prog.push(gemm_add_inst(
                2,
                3,
                layer,
                "Weights::self_attn_o_proj",
                128,
                128,
                3,
            ));
            prog.push(rms_norm_inst(
                4,
                5,
                layer,
                "Weights::post_attention_layernorm",
            ));
            prog.push(gemm_add_inst(
                6,
                7,
                layer,
                "Weights::mlp_down_proj",
                128,
                128,
                3,
            ));
        }
        prog.push(rms_norm_inst(8, 9, 0, "Weights::norm"));
        let roles = classify_positions(&prog).unwrap();
        // Norms: positions 0, 2, 4, 6, 8 — Attn, Mlp, Attn, Mlp, LmHead.
        assert_eq!(roles.norm[0], Some(NormRole::Attn));
        assert_eq!(roles.norm[2], Some(NormRole::Mlp));
        assert_eq!(roles.norm[4], Some(NormRole::Attn));
        assert_eq!(roles.norm[6], Some(NormRole::Mlp));
        assert_eq!(roles.norm[8], Some(NormRole::LmHead));
        // GemmAdds: positions 1, 3, 5, 7 — OProj, DownProj, OProj, DownProj.
        assert_eq!(roles.gemm_add[1], Some(GemmAddRole::OProj));
        assert_eq!(roles.gemm_add[3], Some(GemmAddRole::DownProj));
        assert_eq!(roles.gemm_add[5], Some(GemmAddRole::OProj));
        assert_eq!(roles.gemm_add[7], Some(GemmAddRole::DownProj));
    }

    /// Even-count norms don't match llama-shape (need 2*L+1 odd).
    /// `classify_positions` returns `None` so the canonical falls
    /// through to host as kvm-ineligible — no panic at codegen.
    #[test]
    fn classify_rejects_even_norm_count() {
        let prog = vec![
            rms_norm_inst(0, 1, 0, "Weights::input_layernorm"),
            rms_norm_inst(2, 3, 0, "Weights::post_attention_layernorm"),
        ];
        assert!(classify_positions(&prog).is_none());
    }

    /// AttnNorm row carries opcode 1 + layer + num_items=1 + batch
    /// index, fanning out to `batch_size` rows.
    #[test]
    fn rms_norm_attn_emits_per_token_rows() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![
            rms_norm_inst(0, 1, 7, "Weights::input_layernorm"),
            gemm_add_inst(2, 3, 7, "Weights::self_attn_o_proj", 128, 128, 3),
            rms_norm_inst(4, 5, 7, "Weights::post_attention_layernorm"),
            gemm_add_inst(6, 7, 7, "Weights::mlp_down_proj", 128, 128, 3),
            rms_norm_inst(8, 9, 0, "Weights::norm"),
        ];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b(), None)
            .expect("AttnNorm + GemmAdd has kvm arms");
        // Row counts: norms = 3 × batch_size (128) = 384;
        // gemm_adds = 2 × num_batch_blocks(1) × num_output_blocks(8) = 16.
        // Total = 400.
        assert_eq!(bucket.rows.len(), 384 + 16);
        // First 128 rows are AttnNorm @ layer=7.
        for (i, row) in bucket.rows[..128].iter().enumerate() {
            assert_eq!(row.opcode(), OPCODE_ATTN_NORM);
            assert_eq!(row.payload[1], 7);
            assert_eq!(row.payload[2], 1);
            assert_eq!(row.payload[3], i as i32);
        }
    }

    /// O_ProjResidual rows after the first AttnNorm; DownProjResidual
    /// rows after the MlpNorm. Verifies the GemmAdd phase machine.
    #[test]
    fn cutlass_gemm_add_routes_by_phase() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![
            rms_norm_inst(0, 1, 3, "Weights::input_layernorm"),
            gemm_add_inst(2, 3, 3, "Weights::self_attn_o_proj", 128, 128, 3),
            rms_norm_inst(4, 5, 3, "Weights::post_attention_layernorm"),
            gemm_add_inst(6, 7, 3, "Weights::mlp_down_proj", 128, 128, 3),
            rms_norm_inst(8, 9, 0, "Weights::norm"),
        ];
        let bucket =
            try_encode_bucket(&arch, &prog, &ctx_llama_1b(), None).expect("kvm arms registered");
        // batch_size=128 norm rows, then 8 O_Proj rows, then 128
        // norm rows, then 8 DownProj rows, then 128 LmHeadNorm rows.
        let o_proj_start = 128;
        let o_proj_end = o_proj_start + 8;
        for row in &bucket.rows[o_proj_start..o_proj_end] {
            assert_eq!(row.opcode(), OPCODE_O_PROJ_RESIDUAL);
            assert_eq!(row.payload[1], 3);
        }
        let down_proj_start = o_proj_end + 128; // skip the MlpNorm fan-out
        let down_proj_end = down_proj_start + 8;
        for row in &bucket.rows[down_proj_start..down_proj_end] {
            assert_eq!(row.opcode(), OPCODE_DOWN_PROJ_RESIDUAL);
            assert_eq!(row.payload[1], 3);
        }
    }

    /// LM_HeadNorm pins layer=0 even when the IR's layer field is
    /// nonzero. The kernel doesn't read layer for LM_HeadNorm.
    #[test]
    fn lm_head_norm_pins_layer_to_zero() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![
            rms_norm_inst(0, 1, 0, "Weights::input_layernorm"),
            gemm_add_inst(2, 3, 0, "Weights::self_attn_o_proj", 128, 128, 3),
            rms_norm_inst(4, 5, 0, "Weights::post_attention_layernorm"),
            gemm_add_inst(6, 7, 0, "Weights::mlp_down_proj", 128, 128, 3),
            // IR carries layer=42 here; the encoder must override
            // to 0 because role==LmHead.
            rms_norm_inst(8, 9, 42, "Weights::norm"),
        ];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b(), None).unwrap();
        let lm_head_start = bucket.rows.len() - 128;
        for row in &bucket.rows[lm_head_start..] {
            assert_eq!(row.opcode(), OPCODE_LM_HEAD_NORM);
            assert_eq!(row.payload[1], 0); // pinned to 0
        }
    }

    /// `FusedQkvRopeCache` standalone (decode mode) → fan out to
    /// `num_batch_blocks × num_qkv_blocks` `OPCODE_QKV_RopeAppend`
    /// rows. With ctx_llama_1b: B=128, Bblock=128, Q=2048, KV=512,
    /// out_block=256 → qkv_dim=3072, num_qkv_blocks=12,
    /// num_batch_blocks=1 → 12 rows.
    #[test]
    fn qkv_rope_cache_emits_qkv_rope_append() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![qkv_rope_cache_inst(0, 1, 5)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b(), None)
            .expect("FusedQkvRopeCache has a kvm arm");
        assert_eq!(bucket.rows.len(), 12);
        for (i, row) in bucket.rows.iter().enumerate() {
            assert_eq!(row.opcode(), OPCODE_QKV_ROPE_APPEND);
            assert_eq!(row.payload[1], 5); // layer
            assert_eq!(row.payload[2], 0); // local_row (single batch block)
            assert_eq!(row.payload[3], i as i32); // local_col
            assert_eq!(row.payload[4], 0); // row
            assert_eq!(row.payload[5], i as i32); // col
        }
    }

    /// `FusedQkvRopePrefill` shares opcode + row layout with
    /// `FusedQkvRopeCache`. The kernel toggles via
    /// `g.num_prefill_tokens` (KVM_MAPPING.md Q2). Tape-side, the
    /// only difference is that the prefill IR variant has two extra
    /// slot fields (qo_indptr / kv_indptr) that the encoder
    /// discards.
    #[test]
    fn qkv_rope_prefill_uses_same_opcode_and_layout() {
        let arch = arch_with_norm_and_gemm_add();
        let decode_bucket = try_encode_bucket(
            &arch,
            &[qkv_rope_cache_inst(0, 1, 7)],
            &ctx_llama_1b(),
            None,
        )
        .unwrap();
        let prefill_bucket = try_encode_bucket(
            &arch,
            &[qkv_rope_prefill_inst(0, 1, 7)],
            &ctx_llama_1b(),
            None,
        )
        .unwrap();
        assert_eq!(decode_bucket.rows, prefill_bucket.rows);
    }

    /// `FusedGateUpSiluMul` splits into 2× rows (gate first, then
    /// up). Per-opcode fan-out:
    /// `num_batch_blocks × num_intermediate_blocks`. With
    /// ctx_llama_1b: I=8192, out_block=256 → 32 inter blocks ×
    /// 1 batch block = 32 rows per opcode, 64 total.
    #[test]
    fn gate_up_silu_mul_splits_into_two_opcodes_in_order() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![gate_up_silu_mul_inst(0, 1, 11)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b(), None)
            .expect("FusedGateUpSiluMul has a kvm arm");
        assert_eq!(bucket.rows.len(), 64);
        for row in &bucket.rows[..32] {
            assert_eq!(row.opcode(), OPCODE_GATE_SILU);
            assert_eq!(row.payload[1], 11);
        }
        for row in &bucket.rows[32..] {
            assert_eq!(row.opcode(), OPCODE_UP_MATMUL);
            assert_eq!(row.payload[1], 11);
        }
        // Block indices ascend within each opcode partition.
        for (i, row) in bucket.rows[..32].iter().enumerate() {
            assert_eq!(row.payload[3], i as i32);
        }
        for (i, row) in bucket.rows[32..].iter().enumerate() {
            assert_eq!(row.payload[3], i as i32);
        }
    }

    /// Decode attention with one batched payload row. With
    /// num_tokens=1 + num_kv_heads=8 → 8 pairs, fits in one row,
    /// `num_entries = 16`.
    #[test]
    fn attention_decode_emits_single_batched_row_for_small_workload() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![attention_via_cache_inst(0, 1, 9)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b(), None)
            .expect("AttentionViaCache has a kvm arm");
        assert_eq!(bucket.rows.len(), 1);
        let row = &bucket.rows[0];
        assert_eq!(row.opcode(), OPCODE_GQA_ATTENTION_DECODE);
        assert_eq!(row.payload[1], 9); // layer
        assert_eq!(row.payload[2], 16); // num_entries = 8 pairs × 2
        // First pair: (seq=0, kv=0); second: (seq=0, kv=1); …
        for kv in 0..8 {
            assert_eq!(row.payload[3 + 2 * kv], 0); // seq_idx
            assert_eq!(row.payload[3 + 2 * kv + 1], kv as i32); // kv_head
        }
    }

    /// num_tokens=4 × num_kv_heads=8 → 32 pairs → ⌈32/14⌉ = 3
    /// rows. First two rows have 14 pairs (num_entries=28), last
    /// has 4 pairs (num_entries=8).
    #[test]
    fn attention_decode_chunks_pairs_into_14_per_row() {
        let arch = arch_with_norm_and_gemm_add();
        let mut ctx = ctx_llama_1b();
        ctx.num_tokens = 4;
        let prog = vec![attention_via_cache_inst(0, 1, 0)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx, None).unwrap();
        assert_eq!(bucket.rows.len(), 3);
        assert_eq!(bucket.rows[0].payload[2], 28); // 14 pairs × 2
        assert_eq!(bucket.rows[1].payload[2], 28);
        assert_eq!(bucket.rows[2].payload[2], 8); // 4 pairs × 2
        // Verify chunk boundaries: row 0 covers pairs 0..14, row 1
        // covers 14..28, row 2 covers 28..32. Pairs are
        // (seq=p/8, kv=p%8) where p is the pair index.
        let pair_at = |row_idx: usize, slot: usize| -> (i32, i32) {
            let r = &bucket.rows[row_idx];
            (r.payload[3 + 2 * slot], r.payload[3 + 2 * slot + 1])
        };
        assert_eq!(pair_at(0, 0), (0, 0));
        assert_eq!(pair_at(0, 13), (1, 5)); // pair 13 = seq 1, kv 5
        assert_eq!(pair_at(1, 0), (1, 6)); // pair 14
        assert_eq!(pair_at(2, 0), (3, 4)); // pair 28 = seq 3, kv 4
        assert_eq!(pair_at(2, 3), (3, 7)); // pair 31 (final)
    }

    /// `FlashInferAttentionDecode` shares the row layout with
    /// `AttentionViaCache`. The legacy and FA-based paths are two
    /// IR variants but one TK opcode.
    #[test]
    fn flashinfer_attention_decode_uses_same_opcode_and_layout() {
        let arch = arch_with_norm_and_gemm_add();
        let legacy = try_encode_bucket(
            &arch,
            &[attention_via_cache_inst(0, 1, 7)],
            &ctx_llama_1b(),
            None,
        )
        .unwrap();
        let fi = try_encode_bucket(
            &arch,
            &[flashinfer_attention_decode_inst(0, 1, 7)],
            &ctx_llama_1b(),
            None,
        )
        .unwrap();
        assert_eq!(legacy.rows, fi.rows);
    }

    /// `CutlassGemm` at LM_Head position → `OPCODE_LM_Head` with
    /// layer pinned to 0 regardless of the IR's layer value
    /// (kernel doesn't read layer for LM_Head). Fan-out:
    /// num_batch_blocks × num_logit_blocks. With ctx_llama_1b:
    /// vocab=128_256, out_block=256, but vocab/256 = 501 — not
    /// integer-clean. Use a vocab-aligned variant for this test
    /// to keep the row count predictable.
    #[test]
    fn cutlass_gemm_emits_lm_head_with_layer_pinned() {
        let arch = arch_with_norm_and_gemm_add();
        let mut ctx = ctx_llama_1b();
        ctx.vocab_size = 32_768; // 128 × 256, integer-clean
        let prog = vec![cutlass_gemm_inst(0, 1, 42, "Weights::lm_head", 128, 128, 3)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx, None)
            .expect("CutlassGemm has a kvm arm at LM_Head position");
        let num_logit_blocks = 32_768 / 256; // 128
        assert_eq!(bucket.rows.len(), num_logit_blocks);
        for (i, row) in bucket.rows.iter().enumerate() {
            assert_eq!(row.opcode(), OPCODE_LM_HEAD);
            assert_eq!(row.payload[1], 0); // layer pinned to 0
            assert_eq!(row.payload[2], 0); // local_row (single batch block)
            assert_eq!(row.payload[3], i as i32); // local_col
        }
    }

    /// `None` SeqInfo for a prefill arm = canonical structurally
    /// kvm-eligible, rows deferred to launcher. Empty Vec, NOT
    /// `None` from `try_encode_bucket` (which would mark
    /// kvm-ineligible).
    #[test]
    fn attention_prefill_with_none_seq_info_returns_empty() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![attention_prefill_contiguous_inst(0, 1, 0)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b(), None)
            .expect("AttentionPrefillContiguous is structurally kvm-eligible");
        assert!(
            bucket.rows.is_empty(),
            "no SeqInfo at codegen → empty rows; launcher rebuilds at runtime"
        );
    }

    /// Synthetic SeqInfo: 2 sequences, q_lens [16, 24], offsets
    /// [0, 32]. With ctx num_kv_heads=8 and loop order
    /// outer seq → kv_head → q_block:
    ///   - Seq 0: 16 q-tokens → ⌈16/16⌉=1 q-block × 8 kv heads = 8
    ///   - Seq 1: 24 q-tokens → ⌈24/16⌉=2 q-blocks × 8 kv heads = 16
    ///
    /// Total: 24 rows.
    #[test]
    fn attention_prefill_with_seq_info_emits_per_q_block_rows() {
        let arch = arch_with_norm_and_gemm_add();
        let seq_info: Vec<(usize, usize)> = vec![(16, 0), (24, 32)];
        let prog = vec![attention_prefill_contiguous_inst(0, 1, 5)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b(), Some(&seq_info)).unwrap();
        assert_eq!(bucket.rows.len(), 8 + 16);
        // Seq 0's 8 rows: q_block=0 fixed, kv_head varies 0..8.
        for kv_head in 0..8 {
            let row = &bucket.rows[kv_head];
            assert_eq!(row.opcode(), OPCODE_GQA_ATTENTION_PREFILL);
            assert_eq!(row.payload[1], 5); // layer
            assert_eq!(row.payload[2], 0); // seq_idx
            assert_eq!(row.payload[3], 0); // q_block_idx
            assert_eq!(row.payload[4], 0); // token_offset
            assert_eq!(row.payload[5], kv_head as i32);
        }
        // Seq 1's 16 rows: kv_head=0 first (q_block 0 then 1), kv_head=1 next, …
        let row = &bucket.rows[8]; // first row of seq 1: kv_head=0, q_block=0
        assert_eq!(row.payload[2], 1); // seq_idx
        assert_eq!(row.payload[3], 0); // q_block_idx
        assert_eq!(row.payload[4], 32); // token_offset
        assert_eq!(row.payload[5], 0); // kv_head
        let row = &bucket.rows[9]; // q_block=1 within same kv_head
        assert_eq!(row.payload[3], 1);
        assert_eq!(row.payload[5], 0);
        let row = &bucket.rows[10]; // back to q_block=0, kv_head=1
        assert_eq!(row.payload[3], 0);
        assert_eq!(row.payload[5], 1);
    }

    /// `FlashInferAttentionPrefill` shares opcode + row layout with
    /// `AttentionPrefillContiguous`. Same SeqInfo → same rows.
    #[test]
    fn flashinfer_attention_prefill_uses_same_opcode_and_layout() {
        let arch = arch_with_norm_and_gemm_add();
        let seq_info: Vec<(usize, usize)> = vec![(8, 0)];
        let legacy = try_encode_bucket(
            &arch,
            &[attention_prefill_contiguous_inst(0, 1, 3)],
            &ctx_llama_1b(),
            Some(&seq_info),
        )
        .unwrap();
        let fi = try_encode_bucket(
            &arch,
            &[flashinfer_attention_prefill_inst(0, 1, 3)],
            &ctx_llama_1b(),
            Some(&seq_info),
        )
        .unwrap();
        assert_eq!(legacy.rows, fi.rows);
    }

    /// `emit_kvm_program` renders a `static <ident>: [[i32; 32]; N]`
    /// declaration with one row per `KvmEncodedRow`. Verify the
    /// rendered TokenStream parses as a `static` item with the
    /// right shape.
    #[test]
    fn emit_kvm_program_renders_static_array() {
        let arch = arch_with_norm_and_gemm_add();
        let prog = vec![
            rms_norm_inst(0, 1, 0, "Weights::input_layernorm"),
            gemm_add_inst(2, 3, 0, "Weights::self_attn_o_proj", 128, 128, 3),
            rms_norm_inst(4, 5, 0, "Weights::post_attention_layernorm"),
            gemm_add_inst(6, 7, 0, "Weights::mlp_down_proj", 128, 128, 3),
            rms_norm_inst(8, 9, 0, "Weights::norm"),
        ];
        let mut ctx = ctx_llama_1b();
        ctx.batch_size = 8;
        ctx.matmul_batch_block_size = 8;
        let bucket = try_encode_bucket(&arch, &prog, &ctx, None).unwrap();
        let static_ident = Ident::new("KVM_TEST", Span::call_site());
        let ts = emit_kvm_program(&static_ident, &bucket);
        let item: syn::ItemStatic = syn::parse2(ts).expect("renders as a parseable static item");
        assert_eq!(item.ident, "KVM_TEST");
    }
}
