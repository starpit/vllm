// SPDX-License-Identifier: Apache-2.0
//! Primitive megakernel emitter (Phase 1).
//!
//! One persistent `__global__` per arch; body is a `switch` over
//! `[i32; 32]` opcodes; each arm calls a `__device__` fn that
//! wraps a vendor kernel (cutlass / flashinfer / TK). No warp
//! specialization, no page virtual memory, no controller / loader
//! / storer split. Per-op handoffs are `__syncthreads()` or
//! grid sync. Job: prove the infra (DeviceCallable Impl audit,
//! encoder / scheduler / launcher plumbing) end-to-end with the
//! smallest possible surface. Probably perf-flat or slightly worse
//! than host on Phase 1.
//!
//! ── Encoder ─────────────────────────────────────────────────────
//!
//! [`try_encode_bucket`] takes the same `Vec<OpInstance>` the host
//! emitter consumes and produces an [`EncodedBucket`] — the static
//! tape rows + a pointer-plan + a runtime-fill list. The encoder is
//! a closed match per variant; variants without an arm cause
//! [`try_encode_bucket`] to return `None`, marking the canonical
//! mega-ineligible at codegen time (no `_` catch-all, no runtime
//! "refused" returns).
//!
//! Slot conventions match `vllm-cuda/csrc/megakernel/prim_mega.cu`'s
//! per-op `run_*` docstrings + `dc_cutlass.cuh`'s per-template
//! caller contracts. Each row is up to 32 i32 entries; trailing
//! slots are zero-padded by the emitter (step 7).
//!
//! Loop unrolling: the host emitter compresses repeating sub-runs
//! via `apply_loop_compression`. Vendor's controller has no LOOP
//! opcode in Phase 1 (see MEGA_HANDOFF.md §"Loops"), so the encoder
//! re-expands the compressed form here, threading the iteration
//! index through as `layer = baseline + iter`.
//!
//! Free / Alias rows: pt[] is fixed for the duration of one
//! cooperative launch, so neither variant has runtime effect inside
//! the megakernel. The encoder skips them; alias relationships are
//! folded into the pointer-plan instead (the launcher resolves
//! TileSlot indices through the colored slot map).
//!
//! The launcher emission (step 7) and FFI runtime types are
//! deferred — the encoder is testable against synthetic OpInstances
//! without them. Once step 7 lands, [`EncodedBucket`] is consumed
//! by the per-arch globals struct + launcher fn macro emit.

#![allow(dead_code)]

use std::collections::BTreeMap;

use proc_macro2::TokenStream;
use quote::quote;

use crate::impl_lib::{OpInstance, OpcodeShape};
use crate::interpreters::host::ArchOpcodes;

// ── Opcode constants (mirror prim_mega.cu's enum Opcode) ────────

pub const OP_END: i32 = -1;
pub const OP_NOP: i32 = 0;
pub const OP_RMS_NORM: i32 = 1;
pub const OP_FUSED_ADD_RMS_NORM: i32 = 2;
pub const OP_QKV_ROPE_CACHE: i32 = 3;
pub const OP_SILU_AND_MUL: i32 = 4;
pub const OP_GEMV: i32 = 5;
pub const OP_CUTLASS_GEMM: i32 = 6;
pub const OP_CUTLASS_GEMM_SPLITK: i32 = 7;

/// Maximum width of one tape row in i32 entries. Pinned to match
/// `prim_mega::INSTRUCTION_WIDTH` in `prim_mega.cu` and vendor's
/// `INSTRUCTION_WIDTH` in `~/Megakernels/include/config.cuh`.
pub const INSTRUCTION_WIDTH: usize = 32;

// ── Pointer-plan + runtime-fill types ───────────────────────────

/// Which device-side field of the host accessor's return value
/// the launcher takes a pointer to.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WeightField {
    /// `(weight_fn)(W, layer).weight` — the dense Tensor on
    /// `RmsNorm` / dense `LinearLayer`.
    Weight,
    /// `(weight_fn)(W, layer).dense_weight()` — the dense path
    /// accessor for `LinearLayer` returning `&OwnedTensor`.
    DenseWeight,
    /// `(cos_sin_fn)(W, layer)` — the function returns a
    /// [`GpuTensor`] by value; launcher takes its data pointer.
    AsGpuTensor,
}

/// Per-program workspace allocations the launcher pre-allocates
/// once per cooperative launch.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WorkspaceKind {
    /// `[split_k, M, N]` float scratch for a split-k GEMM
    /// reduction. Keyed by row index in the program because two
    /// different split-k rows may want different shapes.
    SplitKScratch { row_idx: u32 },
}

/// Forward-context-level pointers (per call, but uniform across
/// the whole tape).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ForwardField {
    /// `*(ctx.fwd.input_ids)` — `uint32_t*` positions for
    /// rope/qkv cache write. Used by `OP_QKV_ROPE_CACHE`.
    Positions,
    /// `*(ctx.fwd.slot_mapping)` — `int64_t*` slot mapping for
    /// kv-cache write. Used by `OP_QKV_ROPE_CACHE`.
    SlotMapping,
}

/// One pointer slot the launcher fills before kicking the kernel.
/// pt[] is built once per cooperative launch from these specs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PtrSpec {
    /// `pt[i] := tile_table[slot].as_mut_ptr()`.
    TileSlot(u32),
    /// `pt[i] := (Weights::<fn_ident>)(W, <layer>).<field>.as_ptr()`.
    /// `fn_ident` is the field accessor on `Weights` (e.g.
    /// `input_layernorm`); layer is post-unroll concrete (baseline
    /// + iter offset).
    Weight {
        fn_ident: String,
        layer: u32,
        field: WeightField,
    },
    /// Per-program workspace allocation.
    Workspace(WorkspaceKind),
    /// Forward-context-level pointer.
    Forward(ForwardField),
}

/// Runtime-filled scalar in a tape row. The static tape rows hold
/// 0 in these positions; the launcher patches them per call after
/// copying the template tape to the device buffer.
///
/// Why not bake at codegen: scalars like `eps` come from
/// safetensors-loaded `Weights` fields, not from `ModelParams` or
/// `CanonicalParams` consts. Two checkpoints sharing one canonical
/// can have different `.eps`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeSource {
    /// `f32::to_bits((Weights::<fn_ident>)(W, <layer>).eps) as i32`.
    WeightEpsBits { fn_ident: String, layer: u32 },
}

/// One slot of a `[i32; 32]` row. Trailing slots not present in
/// `EncodedRow.slots` are zero-padded at emit time.
#[derive(Clone, Debug)]
pub enum RowSlot {
    /// Codegen-time literal.
    Const(i32),
    /// Codegen-time const expression rendered verbatim by the
    /// emitter (e.g. `<Weights as CanonicalParams>::HIDDEN_SIZE
    /// as i32`). Used for shape constants the encoder doesn't
    /// know but the per-canonical impl block does.
    ConstExpr(TokenStream),
    /// Pointer index into pt[]; resolved post-pass by
    /// [`EncodedBucket::assign_ptr_indices`].
    Ptr(PtrSpec),
    /// Slot the launcher fills at call time from runtime data.
    Runtime(RuntimeSource),
}

/// One encoded `[i32; 32]` row, pre-pointer-resolution.
#[derive(Clone, Debug)]
pub struct EncodedRow {
    pub slots: Vec<RowSlot>,
}

impl EncodedRow {
    /// Start a new row with `opcode` in slot 0.
    pub fn new(opcode: i32) -> Self {
        Self {
            slots: vec![RowSlot::Const(opcode)],
        }
    }
    pub fn push_const(&mut self, v: i32) {
        self.slots.push(RowSlot::Const(v));
    }
    pub fn push_const_expr(&mut self, ts: TokenStream) {
        self.slots.push(RowSlot::ConstExpr(ts));
    }
    pub fn push_ptr(&mut self, p: PtrSpec) {
        self.slots.push(RowSlot::Ptr(p));
    }
    pub fn push_runtime(&mut self, r: RuntimeSource) {
        self.slots.push(RowSlot::Runtime(r));
    }
    /// Bit-pattern of an f32 as i32 — for `__int_as_float(row[i])`
    /// readers in the .cu (alpha, beta, eps when known at codegen).
    pub fn push_f32_bits(&mut self, v: f32) {
        self.slots.push(RowSlot::Const(v.to_bits() as i32));
    }
}

/// Codegen-time inputs every encoder run needs that don't live on
/// `OpInstance`. Resolved by the caller from the workload point
/// (M / sk_bucket) + per-canonical [`CanonicalParams`] consts.
#[derive(Clone, Debug)]
pub struct EncodeCtx {
    /// Number of decode/prefill tokens for this bucket — runs as
    /// `num_rows` for elementwise-per-token kernels and as `M`
    /// for the GEMM family. Equal to the workload point's
    /// `num_tokens` value.
    pub num_tokens: u32,
    /// `<Weights as CanonicalParams>::HIDDEN_SIZE`. The encoder
    /// renders it as a [`RowSlot::ConstExpr`] to avoid hard-coding
    /// per-arch constants on the macro side; we hand the raw
    /// integer for the encoder's own scalar-shaped slots and the
    /// emitter substitutes the const expression as needed.
    pub hidden_size: u32,
    /// `<Weights as CanonicalParams>::Q_SIZE`.
    pub q_size: u32,
    /// `<Weights as CanonicalParams>::KV_SIZE`.
    pub kv_size: u32,
    /// `<Weights as CanonicalParams>::HEAD_SIZE`.
    pub head_size: u32,
    /// `<Weights as CanonicalParams>::INTERMEDIATE_SIZE`.
    pub intermediate_size: u32,
}

/// Encoded bucket — list of unrolled rows + ordered pointer plan
/// + ordered runtime-fill list. Consumed by step 7's launcher emit.
#[derive(Clone, Debug)]
pub struct EncodedBucket {
    /// Rows in unrolled order. Length is the number of opcodes the
    /// kernel will execute (no `Loop`, no `Free`, no `Alias`).
    pub rows: Vec<EncodedRow>,
    /// pt[index] is filled from `ptr_plan[index]` at launch time.
    /// Stable order (insertion order from row scan); deduped — two
    /// equal `PtrSpec`s share one index.
    pub ptr_plan: Vec<PtrSpec>,
    /// Runtime patches: `(row_idx, slot_idx, source)`. Applied to
    /// the device-resident tape buffer after the static template
    /// is copied in but before launch.
    pub runtime_fills: Vec<(u32, u32, RuntimeSource)>,
}

impl EncodedBucket {
    /// Walk every `RowSlot::Ptr(spec)` and replace it with
    /// `RowSlot::Const(idx)` where idx is the assigned pt[] index.
    /// Populates `ptr_plan` in insertion order. Idempotent on
    /// already-assigned rows (Const(idx) passes through unchanged).
    /// Also populates `runtime_fills` from `RowSlot::Runtime`.
    pub fn assign_ptr_indices(&mut self) {
        let mut plan: Vec<PtrSpec> = Vec::new();
        let mut runtime: Vec<(u32, u32, RuntimeSource)> = Vec::new();
        for (row_idx, row) in self.rows.iter_mut().enumerate() {
            for (slot_idx, slot) in row.slots.iter_mut().enumerate() {
                match slot {
                    RowSlot::Ptr(spec) => {
                        let idx = plan.iter().position(|s| s == spec).unwrap_or_else(|| {
                            plan.push(spec.clone());
                            plan.len() - 1
                        });
                        *slot = RowSlot::Const(idx as i32);
                    }
                    RowSlot::Runtime(src) => {
                        runtime.push((row_idx as u32, slot_idx as u32, src.clone()));
                        *slot = RowSlot::Const(0);
                    }
                    _ => {}
                }
            }
        }
        self.ptr_plan = plan;
        self.runtime_fills = runtime;
    }
}

// ── Encoder driver ──────────────────────────────────────────────

/// Encode a whole bucket. Returns `None` if any instance is
/// mega-ineligible (no encoder arm, OR the variant has an arm but
/// some field's literal extraction failed). Handles `Loop` rows by
/// unrolling at encode time; skips `Free` and `Alias` rows.
pub fn try_encode_bucket(
    arch_opcodes: &ArchOpcodes,
    instances: &[OpInstance],
    ctx: &EncodeCtx,
) -> Option<EncodedBucket> {
    let shapes = arch_opcodes.shapes_by_name();
    let mut rows: Vec<EncodedRow> = Vec::with_capacity(instances.len());

    // Walk with explicit cursor so `Loop` can advance past its body.
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
                        if let Some(row) = encode_op_arm(&shapes, body_inst, iter, ctx)? {
                            rows.push(row);
                        }
                    }
                }
                i = body_end;
            }
            "Free" | "Alias" => {
                // Mega has fixed pt[] — no per-row free/alias.
                i += 1;
            }
            _ => {
                if let Some(row) = encode_op_arm(&shapes, inst, 0, ctx)? {
                    rows.push(row);
                }
                i += 1;
            }
        }
    }

    let mut bucket = EncodedBucket {
        rows,
        ptr_plan: Vec::new(),
        runtime_fills: Vec::new(),
    };
    bucket.assign_ptr_indices();
    Some(bucket)
}

/// Per-variant arm. Returns:
/// - `Some(Some(row))` — variant encoded successfully.
/// - `Some(None)` — variant intentionally produces no row
///   (structural-only IR variants).
/// - `None` — variant has no arm (canonical mega-ineligible) or
///   field extraction failed (codegen invariant violation).
///
/// The outer `Option` carries fail-the-bucket; the inner carries
/// emit-no-row. This keeps the call site clean and lets the closed
/// match below stay flat.
fn encode_op_arm(
    shapes: &BTreeMap<String, OpcodeShape>,
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<Option<EncodedRow>> {
    // Sanity: variant must be registered. This catches stale
    // `arch_opcodes` (caller forgot a `register` somewhere) before
    // any field-shape unwrap below blows up with a less-clear
    // panic.
    let _ = shapes.get(&inst.name.to_string()).or_else(|| {
        panic!(
            "prim_mega encoder: variant `{}` not in arch_opcodes — \
             caller missed an `ArchOpcodes::register` for the picked Impl",
            inst.name,
        )
    });

    match inst.name.to_string().as_str() {
        "RmsNorm" => Some(Some(encode_rms_norm(inst, layer_offset, ctx)?)),
        "FusedAddRmsNorm" => Some(Some(encode_fused_add_rms_norm(inst, layer_offset, ctx)?)),
        "FusedQkvRopeCache" => Some(Some(encode_fused_qkv_rope_cache(inst, layer_offset, ctx)?)),
        "CutlassGemm" => Some(Some(encode_cutlass_gemm(
            inst,
            layer_offset,
            ctx,
            /*beta=*/ 0.0,
        )?)),
        "CutlassGemmAdd" => {
            // Same `device::Gemm` underlying class as bare GEMM —
            // residual-add is a runtime `beta=1.0` epilogue. Map
            // `residual_slot` to both the C and beta-source ptr.
            Some(Some(encode_cutlass_gemm_add(inst, layer_offset, ctx)?))
        }
        "CutlassGemmSplitK" => Some(Some(encode_cutlass_gemm_splitk(inst, layer_offset, ctx)?)),
        "CutlassGemv" => Some(Some(encode_cutlass_gemv(inst, layer_offset, ctx)?)),
        // Variants with no prim_mega arm — mark canonical
        // mega-ineligible. NO `_` catch-all here: every variant
        // must be explicitly named so adding a new Instruction
        // variant in `ferrite-forward/src/instr.rs` shows up as
        // a compile error here, not a silent runtime fall-through.
        // Variants that exist but have no DC sibling C++ kernel:
        "Embed"
        | "LayerNorm"
        | "Reshape"
        | "Add"
        | "ScalarMul"
        | "TanhSoftCap"
        | "FusedAddRmsNormWithOffset"
        | "ScalarOffsetRmsNorm"
        | "Gemm"
        | "FusedGemmBias"
        | "FusedGateUpSiluMul"
        | "FusedGateUpGeluMul"
        | "FusedQkvQkNormRopeCache"
        | "FusedQkvRopePrefill"
        | "AttentionViaCache"
        | "AttentionPrefillContiguous"
        | "SlidingAttentionViaCache"
        | "SlidingAttentionPrefillContiguous"
        | "FlashInferAttentionDecode"
        | "FlashInferAttentionPrefill"
        | "RopeAppend"
        | "MlaSplit"
        | "MlaAttention"
        | "DeepSeekMoe"
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
            "prim_mega encoder: unknown variant `{other}` — add an arm \
             explicitly listing it as mega-eligible (return Some) or \
             mega-ineligible (return None). No `_` catch-all by design."
        ),
    }
}

// ── Per-variant arms ────────────────────────────────────────────

/// `RmsNorm(in_slot, out_slot, layer, weight_fn)` →
/// `OP_RMS_NORM` with row layout per `prim_mega.cu::run_rms_norm`:
/// `[op, ptr_out, ptr_in, ptr_w, eps_bits, hidden, num_rows,
///   smem_off]`.
fn encode_rms_norm(inst: &OpInstance, layer_offset: u32, ctx: &EncodeCtx) -> Option<EncodedRow> {
    let in_slot = parse_u32_literal(&inst.field_values[0])?;
    let out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = layer_baseline + layer_offset;
    let fn_ident = path_last_segment(&inst.field_values[3])?;

    let mut row = EncodedRow::new(OP_RMS_NORM);
    row.push_ptr(PtrSpec::TileSlot(out_slot));
    row.push_ptr(PtrSpec::TileSlot(in_slot));
    row.push_ptr(PtrSpec::Weight {
        fn_ident: fn_ident.clone(),
        layer,
        field: WeightField::Weight,
    });
    row.push_runtime(RuntimeSource::WeightEpsBits { fn_ident, layer });
    row.push_const(ctx.hidden_size as i32);
    row.push_const(ctx.num_tokens as i32);
    row.push_const(0); // smem_off — single-op layout (step 7 sizes it)
    Some(row)
}

/// `FusedAddRmsNorm(delta_slot, residual_slot, layer, weight_fn)`
/// → `OP_FUSED_ADD_RMS_NORM`. Row layout per
/// `prim_mega.cu::run_fused_add_rms_norm`:
/// `[op, ptr_input, ptr_residual, ptr_w, eps_bits, hidden,
///   num_rows, smem_off]`. The `input` slot in `dc_fused_add_rms_norm`
/// is the delta (in/out — gets normalized output written back).
fn encode_fused_add_rms_norm(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<EncodedRow> {
    let delta_slot = parse_u32_literal(&inst.field_values[0])?;
    let residual_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = layer_baseline + layer_offset;
    let fn_ident = path_last_segment(&inst.field_values[3])?;

    let mut row = EncodedRow::new(OP_FUSED_ADD_RMS_NORM);
    row.push_ptr(PtrSpec::TileSlot(delta_slot));
    row.push_ptr(PtrSpec::TileSlot(residual_slot));
    row.push_ptr(PtrSpec::Weight {
        fn_ident: fn_ident.clone(),
        layer,
        field: WeightField::Weight,
    });
    row.push_runtime(RuntimeSource::WeightEpsBits { fn_ident, layer });
    row.push_const(ctx.hidden_size as i32);
    row.push_const(ctx.num_tokens as i32);
    row.push_const(0);
    Some(row)
}

/// `FusedQkvRopeCache(in_slot, out_slot, layer, weight_fn,
///                    cos_sin_fn, biased, interleaved)` →
/// `OP_QKV_ROPE_CACHE`. Row layout per
/// `prim_mega.cu::run_qkv_rope_cache`:
/// `[op, ptr_q_out, ptr_k_cache, ptr_v_cache, ptr_qkv,
///   ptr_positions, ptr_cos_sin, ptr_slot_mapping,
///   q_size, kv_size, head_size, num_rows]`.
///
/// `ptr_qkv` here is the QKV-projection output landing pad —
/// resolved from `in_slot` / `out_slot` per the host counterpart's
/// alias semantics: the rope-append writes Q in place at `out_slot`
/// and writes K/V into the kv_cache extern storage. For Phase 1
/// the encoder maps `ptr_qkv` to the in-tile (the QKV proj output)
/// and `ptr_q_out` to the out-tile (Q after rope, in place over
/// the same memory). The kv_cache pointers live on the Weights
/// type — encoder takes them via `WeightField::DenseWeight` on
/// the kv_cache accessor, which step 7's launcher resolves.
///
/// **TODO step 7**: kv_cache accessor isn't `weight_fn` — it's
/// a separate per-arch Weights field. Phase 1 leaves placeholder
/// PtrSpecs that the launcher must intercept. The encoder pins
/// the row layout; the launcher fills these from the
/// `ferrite_kernels::layers::KvCache` struct on Weights.
fn encode_fused_qkv_rope_cache(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<EncodedRow> {
    let in_slot = parse_u32_literal(&inst.field_values[0])?;
    let out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = layer_baseline + layer_offset;
    let weight_fn_ident = path_last_segment(&inst.field_values[3])?;
    let cos_sin_fn_ident = path_last_segment(&inst.field_values[4])?;
    // biased / interleaved consumed by step 7 launcher
    let _biased = parse_bool_literal(&inst.field_values[5])?;
    let _interleaved = parse_bool_literal(&inst.field_values[6])?;

    let mut row = EncodedRow::new(OP_QKV_ROPE_CACHE);
    // ptr_q_out = out_slot (rope writes Q in place over the QKV
    // proj output; encoder treats `out_slot` as the Q landing pad).
    row.push_ptr(PtrSpec::TileSlot(out_slot));
    // ptr_k_cache / ptr_v_cache = WeightField::DenseWeight against
    // the per-layer KvCache accessor. The Weights field name is
    // not in OpInstance — placeholder uses a synthesized accessor
    // name that step 7's launcher pattern-matches. Pinning the
    // row positions per the run_qkv_rope_cache contract is what
    // matters for step 6.
    row.push_ptr(PtrSpec::Weight {
        fn_ident: format!("kv_cache_{layer}_k"),
        layer,
        field: WeightField::DenseWeight,
    });
    row.push_ptr(PtrSpec::Weight {
        fn_ident: format!("kv_cache_{layer}_v"),
        layer,
        field: WeightField::DenseWeight,
    });
    // ptr_qkv = in_slot (QKV proj output landing pad).
    row.push_ptr(PtrSpec::TileSlot(in_slot));
    // ptr_positions / ptr_cos_sin / ptr_slot_mapping
    row.push_ptr(PtrSpec::Forward(ForwardField::Positions));
    row.push_ptr(PtrSpec::Weight {
        fn_ident: cos_sin_fn_ident,
        layer,
        field: WeightField::AsGpuTensor,
    });
    row.push_ptr(PtrSpec::Forward(ForwardField::SlotMapping));
    // dims
    row.push_const(ctx.q_size as i32);
    row.push_const(ctx.kv_size as i32);
    row.push_const(ctx.head_size as i32);
    row.push_const(ctx.num_tokens as i32);
    // Suppress unused warnings: weight_fn_ident is recorded but
    // not yet routed (the QKV proj LinearLayer's `.dense_weight()`
    // belongs to a previous CutlassGemm row, not this one). Step 7
    // links them via the alias chain.
    let _ = weight_fn_ident;
    Some(row)
}

/// `CutlassGemm(in_slot, out_slot, layer, weight_fn,
///              tile_m, tile_n, stages)` → `OP_CUTLASS_GEMM`.
/// Row layout per `prim_mega.cu::run_cutlass_gemm`:
/// `[op, ptr_C, ptr_A, ptr_B, M, N, K, alpha_bits, beta_bits,
///   smem_off, config_id]`.
///
/// `M / N / K` left as runtime here — N / K come from
/// `LinearLayer::weight.shape()` which is per-Weights, and the
/// encoder doesn't have it. Step 7's launcher resolves N / K from
/// the same accessor's weight tensor at call time.
///
/// `beta` taken as a parameter so `encode_cutlass_gemm_add` can
/// reuse the same body with `beta=1.0`.
fn encode_cutlass_gemm(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
    beta: f32,
) -> Option<EncodedRow> {
    let in_slot = parse_u32_literal(&inst.field_values[0])?;
    let out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = layer_baseline + layer_offset;
    let fn_ident = path_last_segment(&inst.field_values[3])?;
    let tile_m = parse_u32_literal(&inst.field_values[4])?;
    let tile_n = parse_u32_literal(&inst.field_values[5])?;
    let stages = parse_u32_literal(&inst.field_values[6])?;
    let config_id = cutlass_gemm_config_id(tile_m, tile_n, stages)?;

    let mut row = EncodedRow::new(OP_CUTLASS_GEMM);
    row.push_ptr(PtrSpec::TileSlot(out_slot));
    row.push_ptr(PtrSpec::TileSlot(in_slot));
    row.push_ptr(PtrSpec::Weight {
        fn_ident,
        layer,
        field: WeightField::DenseWeight,
    });
    row.push_const(ctx.num_tokens as i32); // M
    row.push_const(0); // N — TODO step 7: ptr_idx_dim or from accessor
    row.push_const(0); // K — same
    row.push_f32_bits(1.0); // alpha
    row.push_f32_bits(beta);
    row.push_const(0); // smem_off
    row.push_const(config_id);
    Some(row)
}

/// `CutlassGemmAdd(in_slot, residual_slot, layer, weight_fn,
///                 tile_m, tile_n, stages)` → `OP_CUTLASS_GEMM`
/// with `beta=1.0`. Same kernel as bare GEMM; the residual-add is
/// the epilogue. Maps `residual_slot` → ptr_C; `in_slot` → ptr_A.
/// Reuses `encode_cutlass_gemm`'s body.
fn encode_cutlass_gemm_add(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<EncodedRow> {
    // CutlassGemmAdd already orders its fields as (in_slot,
    // residual_slot=C, layer, weight_fn, tile_m, tile_n, stages) —
    // exactly what `encode_cutlass_gemm` reads as (in_slot,
    // out_slot=C, ...). The kernel's `out = alpha*A*B + beta*C`
    // semantics turn into a residual-add when `beta=1.0` and `C` is
    // the residual buffer. Pass-through.
    encode_cutlass_gemm(inst, layer_offset, ctx, /*beta=*/ 1.0)
}

/// `CutlassGemmSplitK(in_slot, out_slot, layer, weight_fn,
///                    tile_m, tile_n, stages, split_k)` →
/// `OP_CUTLASS_GEMM_SPLITK`. Row layout per
/// `prim_mega.cu::run_cutlass_gemm_splitk`:
/// `[op, ptr_C, ptr_A, ptr_B, ptr_workspace, M, N, K,
///   alpha_bits, beta_bits, smem_off, config_id, split_k]`.
fn encode_cutlass_gemm_splitk(
    inst: &OpInstance,
    layer_offset: u32,
    ctx: &EncodeCtx,
) -> Option<EncodedRow> {
    let in_slot = parse_u32_literal(&inst.field_values[0])?;
    let out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = layer_baseline + layer_offset;
    let fn_ident = path_last_segment(&inst.field_values[3])?;
    let tile_m = parse_u32_literal(&inst.field_values[4])?;
    let tile_n = parse_u32_literal(&inst.field_values[5])?;
    let stages = parse_u32_literal(&inst.field_values[6])?;
    let split_k = parse_u32_literal(&inst.field_values[7])?;
    let config_id = cutlass_splitk_config_id(tile_m, tile_n, stages, split_k)?;

    let mut row = EncodedRow::new(OP_CUTLASS_GEMM_SPLITK);
    row.push_ptr(PtrSpec::TileSlot(out_slot));
    row.push_ptr(PtrSpec::TileSlot(in_slot));
    row.push_ptr(PtrSpec::Weight {
        fn_ident,
        layer,
        field: WeightField::DenseWeight,
    });
    // workspace — assigned per-row by `WorkspaceKind::SplitKScratch`
    // because two split-k rows may want different shapes.
    row.push_ptr(PtrSpec::Workspace(WorkspaceKind::SplitKScratch {
        // row_idx is patched by `assign_ptr_indices` indirectly via
        // ptr_plan ordering — for the per-row uniqueness we encode
        // the (tile, split_k, layer) signature into a synthetic id.
        // Step 7 emits the actual scratch alloc per ptr_plan entry.
        row_idx: layer * 1000 + split_k * 100 + tile_m + tile_n,
    }));
    row.push_const(ctx.num_tokens as i32); // M
    row.push_const(0); // N — TODO step 7
    row.push_const(0); // K — same
    row.push_f32_bits(1.0); // alpha
    row.push_f32_bits(0.0); // beta
    row.push_const(0); // smem_off
    row.push_const(config_id);
    row.push_const(split_k as i32);
    Some(row)
}

/// `CutlassGemv(in_slot, out_slot, layer, weight_fn)` → `OP_GEMV`.
/// Row layout per `prim_mega.cu::run_gemv`:
/// `[op, ptr_out, ptr_x, ptr_W, N, K, alpha_bits, beta_bits,
///   smem_off]`. M is implicit (=1) — kernel rejects M != 1.
fn encode_cutlass_gemv(
    inst: &OpInstance,
    layer_offset: u32,
    _ctx: &EncodeCtx,
) -> Option<EncodedRow> {
    let in_slot = parse_u32_literal(&inst.field_values[0])?;
    let out_slot = parse_u32_literal(&inst.field_values[1])?;
    let layer_baseline = parse_u32_literal(&inst.field_values[2])?;
    let layer = layer_baseline + layer_offset;
    let fn_ident = path_last_segment(&inst.field_values[3])?;

    let mut row = EncodedRow::new(OP_GEMV);
    row.push_ptr(PtrSpec::TileSlot(out_slot));
    row.push_ptr(PtrSpec::TileSlot(in_slot));
    row.push_ptr(PtrSpec::Weight {
        fn_ident,
        layer,
        field: WeightField::DenseWeight,
    });
    row.push_const(0); // N — TODO step 7 (from accessor's weight shape)
    row.push_const(0); // K — same
    row.push_f32_bits(1.0); // alpha
    row.push_f32_bits(0.0); // beta
    row.push_const(0); // smem_off
    Some(row)
}

// ── Program static emission (step 7a) ───────────────────────────

/// Emit a `static <ident>: [[i32; 32]; N] = [...]` from an
/// already-resolved [`EncodedBucket`]. The C++ kernel reads
/// `tape + pc * INSTRUCTION_WIDTH` (`prim_mega.cu:401`), so the row
/// width is pinned to 32 here; trailing slots not present on the
/// [`EncodedRow`] zero-pad.
///
/// Pre-condition: the bucket has been through
/// [`EncodedBucket::assign_ptr_indices`] — every `RowSlot::Ptr` has
/// been replaced by `RowSlot::Const(idx)` and every
/// `RowSlot::Runtime` by `RowSlot::Const(0)` (the launcher patches
/// the zeroed slot per-call from `runtime_fills`). Encountering an
/// unresolved variant is a codegen invariant violation; the function
/// panics so the failure surfaces at macro expansion, not at link
/// time when the host side ends up writing nonsense into pt[].
///
/// This is the static-template half of the launcher: the launcher fn
/// (step 7b/7c, follow-up commit) consumes the same bucket's
/// `ptr_plan` + `runtime_fills` to fill `pt[]` and patch the
/// device-resident copy of this template before kicking the kernel.
pub fn emit_prim_mega_program(static_ident: &syn::Ident, bucket: &EncodedBucket) -> TokenStream {
    let n = bucket.rows.len();
    let rows = bucket.rows.iter().map(|row| {
        let cells = (0..INSTRUCTION_WIDTH).map(|i| match row.slots.get(i) {
            Some(RowSlot::Const(c)) => {
                let lit = proc_macro2::Literal::i32_unsuffixed(*c);
                quote! { #lit }
            }
            Some(RowSlot::ConstExpr(ts)) => quote! { (#ts) },
            Some(RowSlot::Ptr(spec)) => panic!(
                "emit_prim_mega_program: unresolved RowSlot::Ptr({:?}) — \
                 caller must run EncodedBucket::assign_ptr_indices first",
                spec
            ),
            Some(RowSlot::Runtime(src)) => panic!(
                "emit_prim_mega_program: unresolved RowSlot::Runtime({:?}) — \
                 caller must run EncodedBucket::assign_ptr_indices first",
                src
            ),
            None => quote! { 0 },
        });
        quote! { [ #(#cells),* ] }
    });
    // INSTRUCTION_WIDTH is `usize` for indexing; render as a bare
    // `32` literal here so the static type reads `[[i32; 32]; N]`
    // verbatim — the kernel contract pins the literal width, not a
    // suffixed const.
    let width_lit = proc_macro2::Literal::usize_unsuffixed(INSTRUCTION_WIDTH);
    let n_lit = proc_macro2::Literal::usize_unsuffixed(n);
    quote! {
        #[cfg(feature = "cuda")]
        #[allow(dead_code)]
        static #static_ident: [[i32; #width_lit]; #n_lit] = [ #(#rows),* ];
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Parse a TokenStream of shape `<n>u32` or `<n>` (un-suffixed)
/// into a u32. Mirrors the host emitter's same-named helper.
fn parse_u32_literal(ts: &TokenStream) -> Option<u32> {
    let s = ts.to_string();
    let s = s.trim();
    let s = s.strip_suffix("u32").unwrap_or(s);
    s.parse::<u32>().ok()
}

/// Parse a TokenStream of shape `true` / `false` into a bool.
fn parse_bool_literal(ts: &TokenStream) -> Option<bool> {
    let s = ts.to_string();
    let s = s.trim();
    match s {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Extract the last `::`-separated segment of a path-shaped
/// TokenStream — `Weights::input_layernorm` → `"input_layernorm"`.
/// Used to recover the Weights-field accessor name from the
/// codegen-emitted `Weights::<ident>` path.
fn path_last_segment(ts: &TokenStream) -> Option<String> {
    let s = ts.to_string();
    // TokenStream stringification spaces tokens: `Weights :: foo`.
    // Strip whitespace, then take the last `::`-separated piece.
    let stripped: String = s.split_whitespace().collect();
    let last = stripped.rsplit("::").next()?;
    if last.is_empty() {
        return None;
    }
    Some(last.to_string())
}

/// Map a `(tile_m, tile_n, stages)` tuple to the C++
/// `enum CutlassConfig` id — the encoder side of the
/// `CUTLASS_DC_GEMM_LIST` X-macro expansion in
/// `cutlass_gemm_configs.cuh`. Stable ordering: matches the X-macro
/// list verbatim. New rows append; never reorder existing rows or
/// the runtime ids drift.
fn cutlass_gemm_config_id(tile_m: u32, tile_n: u32, stages: u32) -> Option<i32> {
    // Mirrors the C++ X-macro entries in
    // `cutlass_gemm_configs.cuh::CUTLASS_DC_GEMM_LIST`. Order pinned
    // by the `dc_zoo_equals_host_zoo` test on the host side.
    let id = match (tile_m, tile_n, stages) {
        (32, 64, 3) => 0,
        (32, 64, 4) => 1,
        (32, 128, 3) => 2,
        (32, 128, 4) => 3,
        (32, 256, 3) => 4,
        (64, 64, 3) => 5,
        (64, 64, 4) => 6,
        (64, 128, 3) => 7,
        (64, 128, 4) => 8,
        (128, 64, 3) => 9,
        (128, 64, 4) => 10,
        (128, 128, 3) => 11,
        (128, 128, 4) => 12,
        (128, 256, 3) => 13,
        (256, 64, 3) => 14,
        (256, 64, 4) => 15,
        _ => return None,
    };
    Some(id)
}

/// Map a `(tile_m, tile_n, stages, split_k)` tuple to the C++
/// `enum CutlassSplitKConfig` id — mirror of
/// `CUTLASS_DC_SPLITK_LIST` in `cutlass_gemm_configs.cuh`.
fn cutlass_splitk_config_id(tile_m: u32, tile_n: u32, stages: u32, split_k: u32) -> Option<i32> {
    // SplitK list is currently 12 entries — three workhorse tiles
    // × four split_k factors. Pinned by `dc_splitk_zoo_equals_host_zoo`.
    let id = match (tile_m, tile_n, stages, split_k) {
        (64, 64, 3, 2) => 0,
        (64, 64, 3, 4) => 1,
        (64, 64, 3, 8) => 2,
        (64, 64, 3, 16) => 3,
        (64, 128, 3, 2) => 4,
        (64, 128, 3, 4) => 5,
        (64, 128, 3, 8) => 6,
        (64, 128, 3, 16) => 7,
        (128, 128, 3, 2) => 8,
        (128, 128, 3, 4) => 9,
        (128, 128, 3, 8) => 10,
        (128, 128, 3, 16) => 11,
        _ => return None,
    };
    Some(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreters::host::{alias_instance, free_instance, loop_instance};
    use proc_macro2::Span;
    use quote::quote;
    use syn::Ident;

    /// Build an OpInstance directly. Test-only helper that mirrors
    /// what each Impl's `fan_out` produces for the variant.
    fn op(name: &str, fields: Vec<TokenStream>) -> OpInstance {
        OpInstance::new(Ident::new(name, Span::call_site()), fields)
    }

    /// Test-only ArchOpcodes prefilled with the seven mega-eligible
    /// variant shapes. Real codegen builds this incrementally as
    /// each Impl's `opcode_shape()` registers.
    fn arch_with_seven_shapes() -> ArchOpcodes {
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
            "FusedAddRmsNorm",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
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
            "FusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("biased", syn::parse_quote!(bool)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        ));
        a.register(OpcodeShape::new(
            "CutlassGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
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
        a.register(OpcodeShape::new(
            "CutlassGemmSplitK",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
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
                ("split_k", syn::parse_quote!(u32)),
            ],
        ));
        a.register(OpcodeShape::new(
            "CutlassGemv",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
            ],
        ));
        a
    }

    fn ctx_llama_1b() -> EncodeCtx {
        EncodeCtx {
            num_tokens: 1,
            hidden_size: 2048,
            q_size: 2048,
            kv_size: 512,
            head_size: 64,
            intermediate_size: 8192,
        }
    }

    /// RmsNorm encodes to OP_RMS_NORM with three pointer slots
    /// (out, in, weight) plus the eps Runtime + scalar dims.
    /// Pin: variant ident → opcode + ptr-plan dedup.
    #[test]
    fn rms_norm_encodes_to_op_rms_norm() {
        let arch = arch_with_seven_shapes();
        let inst = op(
            "RmsNorm",
            vec![
                quote! { 3u32 },
                quote! { 5u32 },
                quote! { 0u32 },
                quote! { Weights::input_layernorm },
            ],
        );
        let bucket =
            try_encode_bucket(&arch, &[inst], &ctx_llama_1b()).expect("RmsNorm has a mega arm");
        assert_eq!(bucket.rows.len(), 1);
        let row = &bucket.rows[0];
        assert!(matches!(row.slots[0], RowSlot::Const(c) if c == OP_RMS_NORM));
        // Three ptr indices in order: out, in, weight.
        assert_eq!(bucket.ptr_plan.len(), 3);
        assert_eq!(bucket.ptr_plan[0], PtrSpec::TileSlot(5));
        assert_eq!(bucket.ptr_plan[1], PtrSpec::TileSlot(3));
        assert_eq!(
            bucket.ptr_plan[2],
            PtrSpec::Weight {
                fn_ident: "input_layernorm".into(),
                layer: 0,
                field: WeightField::Weight,
            }
        );
        // One runtime fill (eps) at slot index 4.
        assert_eq!(bucket.runtime_fills.len(), 1);
        assert_eq!(bucket.runtime_fills[0].0, 0); // row_idx
        assert_eq!(bucket.runtime_fills[0].1, 4); // slot_idx
        assert_eq!(
            bucket.runtime_fills[0].2,
            RuntimeSource::WeightEpsBits {
                fn_ident: "input_layernorm".into(),
                layer: 0,
            }
        );
        // hidden + num_rows + smem_off in slots 5..8.
        assert!(matches!(row.slots[5], RowSlot::Const(2048)));
        assert!(matches!(row.slots[6], RowSlot::Const(1)));
        assert!(matches!(row.slots[7], RowSlot::Const(0)));
    }

    /// Loop unrolling threads `iter` through `layer = baseline +
    /// iter`. Two RmsNorm rows in a body × 3 iters → 6 rows.
    /// Each iteration's weight ptr_plan entry has a distinct
    /// `layer` field so they dedupe per-iter.
    #[test]
    fn loop_unrolls_with_layer_offset() {
        let arch = arch_with_seven_shapes();
        let body = vec![
            op(
                "RmsNorm",
                vec![
                    quote! { 0u32 },
                    quote! { 1u32 },
                    quote! { 0u32 },
                    quote! { Weights::input_layernorm },
                ],
            ),
            op(
                "RmsNorm",
                vec![
                    quote! { 2u32 },
                    quote! { 3u32 },
                    quote! { 0u32 },
                    quote! { Weights::post_attention_layernorm },
                ],
            ),
        ];
        let mut prog: Vec<OpInstance> = vec![loop_instance(3, 2)];
        prog.extend(body);

        let bucket =
            try_encode_bucket(&arch, &prog, &ctx_llama_1b()).expect("loop body is mega-eligible");
        assert_eq!(bucket.rows.len(), 6, "3 iters × 2 body rows");
        // Distinct layers across iterations: ptr_plan has weight
        // entries for both `input_layernorm` and `post_*` × 3
        // layers each = 6 weight ptrs + 4 tile slots = 10 total.
        let weight_ptrs: Vec<&PtrSpec> = bucket
            .ptr_plan
            .iter()
            .filter(|p| matches!(p, PtrSpec::Weight { .. }))
            .collect();
        assert_eq!(weight_ptrs.len(), 6);
        // Layers should be 0, 1, 2 for each accessor.
        let layers: std::collections::BTreeSet<u32> = weight_ptrs
            .iter()
            .map(|p| match p {
                PtrSpec::Weight { layer, .. } => *layer,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(layers, [0u32, 1, 2].into_iter().collect());
    }

    /// Loop body with a non-zero `layer` baseline (the host
    /// emitter's per-row baseline preservation in
    /// `apply_loop_compression`) must add `iter_offset` to the
    /// baseline, not replace it.
    #[test]
    fn loop_layer_baseline_preserved() {
        let arch = arch_with_seven_shapes();
        let body = vec![op(
            "RmsNorm",
            vec![
                quote! { 0u32 },
                quote! { 1u32 },
                quote! { 5u32 }, // baseline 5, not 0
                quote! { Weights::input_layernorm },
            ],
        )];
        let mut prog: Vec<OpInstance> = vec![loop_instance(2, 1)];
        prog.extend(body);
        let bucket =
            try_encode_bucket(&arch, &prog, &ctx_llama_1b()).expect("loop body is mega-eligible");
        assert_eq!(bucket.rows.len(), 2);
        let layers: Vec<u32> = bucket
            .ptr_plan
            .iter()
            .filter_map(|p| match p {
                PtrSpec::Weight { layer, .. } => Some(*layer),
                _ => None,
            })
            .collect();
        // iter 0 → baseline+0 = 5; iter 1 → baseline+1 = 6.
        assert_eq!(layers, vec![5, 6]);
    }

    /// Free + Alias rows are pt-plan-only — the megakernel walks a
    /// fixed ptr table per cooperative launch, so no runtime
    /// equivalent of either exists. Both are silently skipped at
    /// encode time.
    #[test]
    fn free_and_alias_skipped() {
        let arch = arch_with_seven_shapes();
        let prog = vec![
            alias_instance(7, 3),
            op(
                "RmsNorm",
                vec![
                    quote! { 3u32 },
                    quote! { 5u32 },
                    quote! { 0u32 },
                    quote! { Weights::input_layernorm },
                ],
            ),
            free_instance(99),
        ];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b())
            .expect("only the RmsNorm row survives encoding");
        assert_eq!(bucket.rows.len(), 1);
        assert!(matches!(bucket.rows[0].slots[0], RowSlot::Const(c) if c == OP_RMS_NORM));
    }

    /// Variants with no prim_mega arm produce `None` from
    /// [`try_encode_bucket`] — the canonical is mega-ineligible.
    /// Pin: `Embed` has no DC sibling C++ kernel.
    #[test]
    fn unsupported_variant_returns_none() {
        let mut arch = ArchOpcodes::new();
        arch.register(OpcodeShape::new(
            "Embed",
            vec![
                ("out_slot", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Embedding
                    ),
                ),
            ],
        ));
        let prog = vec![op(
            "Embed",
            vec![quote! { 0u32 }, quote! { Weights::token_embed }],
        )];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b());
        assert!(
            bucket.is_none(),
            "Embed has no prim_mega arm — should reject"
        );
    }

    /// CutlassGemm encodes to OP_CUTLASS_GEMM with a config_id
    /// matching the C++ X-macro position (128x128 stages=3 → id=11).
    #[test]
    fn cutlass_gemm_config_id_matches_xmacro_order() {
        assert_eq!(cutlass_gemm_config_id(128, 128, 3), Some(11));
        assert_eq!(cutlass_gemm_config_id(32, 64, 3), Some(0));
        assert_eq!(cutlass_gemm_config_id(256, 64, 4), Some(15));
        // Tile not in the zoo → None.
        assert_eq!(cutlass_gemm_config_id(64, 32, 5), None);
    }

    /// CutlassGemmAdd reuses the GEMM body with `beta=1.0`. The
    /// emitted f32 bit-pattern at slot 8 must be `1.0_f32.to_bits()`.
    #[test]
    fn cutlass_gemm_add_uses_beta_one() {
        let arch = arch_with_seven_shapes();
        let inst = op(
            "CutlassGemmAdd",
            vec![
                quote! { 3u32 },
                quote! { 5u32 },
                quote! { 0u32 },
                quote! { Weights::o_proj },
                quote! { 128u32 },
                quote! { 128u32 },
                quote! { 3u32 },
            ],
        );
        let bucket = try_encode_bucket(&arch, &[inst], &ctx_llama_1b())
            .expect("CutlassGemmAdd has a mega arm");
        let row = &bucket.rows[0];
        // alpha at slot 7, beta at slot 8 (per run_cutlass_gemm).
        assert!(matches!(row.slots[7], RowSlot::Const(c) if c == 1.0_f32.to_bits() as i32));
        assert!(matches!(row.slots[8], RowSlot::Const(c) if c == 1.0_f32.to_bits() as i32));
    }

    /// CutlassGemmSplitK's split_k workspace gets a unique ptr_plan
    /// entry per row (two split-k rows in the same bucket → two
    /// distinct workspaces, not aliased).
    #[test]
    fn splitk_workspace_per_row() {
        let arch = arch_with_seven_shapes();
        let make = |layer: u32, sk: u32| {
            let layer_lit = proc_macro2::Literal::u32_suffixed(layer);
            let sk_lit = proc_macro2::Literal::u32_suffixed(sk);
            op(
                "CutlassGemmSplitK",
                vec![
                    quote! { 0u32 },
                    quote! { 1u32 },
                    quote! { #layer_lit },
                    quote! { Weights::down_proj },
                    quote! { 128u32 },
                    quote! { 128u32 },
                    quote! { 3u32 },
                    quote! { #sk_lit },
                ],
            )
        };
        let prog = vec![make(0, 4), make(1, 8)];
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b())
            .expect("CutlassGemmSplitK is mega-eligible");
        let workspaces: Vec<&PtrSpec> = bucket
            .ptr_plan
            .iter()
            .filter(|p| matches!(p, PtrSpec::Workspace(_)))
            .collect();
        assert_eq!(workspaces.len(), 2, "two distinct split-k workspaces");
    }

    /// Path-last-segment helper recovers `input_layernorm` from
    /// `Weights::input_layernorm` regardless of intra-token spacing
    /// (TokenStream stringification spaces `::`).
    #[test]
    fn path_last_segment_handles_spaces() {
        let ts = quote! { Weights::input_layernorm };
        assert_eq!(path_last_segment(&ts), Some("input_layernorm".into()));
        let ts2 = quote! { Weights::nested::deep_field };
        assert_eq!(path_last_segment(&ts2), Some("deep_field".into()));
    }

    // ── Program static emission (step 7a) ───────────────────────

    /// Build a one-row encoded bucket with the seven mega-eligible
    /// shapes registered, then `assign_ptr_indices` so emission
    /// pre-conditions hold. Used by every emission test below.
    fn rms_norm_bucket() -> EncodedBucket {
        let arch = arch_with_seven_shapes();
        let inst = op(
            "RmsNorm",
            vec![
                quote! { 3u32 },
                quote! { 5u32 },
                quote! { 0u32 },
                quote! { Weights::input_layernorm },
            ],
        );
        try_encode_bucket(&arch, &[inst], &ctx_llama_1b()).expect("RmsNorm has a mega arm")
    }

    fn ident(name: &str) -> syn::Ident {
        syn::Ident::new(name, Span::call_site())
    }

    /// Emitted token stream parses as a Rust `static` item with the
    /// expected outer shape: `[[i32; 32]; N]`. Pin the
    /// kernel-contract row width here — the C++ side reads
    /// `tape + pc * 32`.
    #[test]
    fn emit_program_renders_static_with_correct_outer_shape() {
        let bucket = rms_norm_bucket();
        let ts = emit_prim_mega_program(&ident("MEGA_PROGRAM_M_1"), &bucket);
        let item: syn::ItemStatic =
            syn::parse2(ts).expect("emit_prim_mega_program produces a parseable static item");
        assert_eq!(item.ident, "MEGA_PROGRAM_M_1");
        let n_lit = proc_macro2::Literal::usize_unsuffixed(bucket.rows.len());
        let expected_ty: syn::Type = syn::parse_quote!([[i32; 32]; #n_lit]);
        let actual_ty = &*item.ty;
        assert_eq!(
            quote! { #expected_ty }.to_string(),
            quote! { #actual_ty }.to_string(),
        );
    }

    /// Each emitted row has exactly 32 i32 cells, regardless of how
    /// many slots the encoder pushed. Trailing cells beyond
    /// `row.slots.len()` zero-pad. Pin the kernel contract.
    #[test]
    fn emit_program_pads_rows_to_32_cells() {
        let bucket = rms_norm_bucket();
        let ts = emit_prim_mega_program(&ident("MEGA_PROGRAM_M_1"), &bucket);
        // Render to string, count `,`-separated cells inside the
        // outermost row literal. The static body is one row of 32
        // cells; the simplest robust check is to parse the static
        // and walk the expression tree.
        let item: syn::ItemStatic = syn::parse2(ts).unwrap();
        let outer = match &*item.expr {
            syn::Expr::Array(a) => a,
            other => panic!("expected outer Expr::Array, got {:?}", other),
        };
        assert_eq!(outer.elems.len(), 1, "one row");
        let row = match &outer.elems[0] {
            syn::Expr::Array(a) => a,
            other => panic!("expected row Expr::Array, got {:?}", other),
        };
        assert_eq!(
            row.elems.len(),
            INSTRUCTION_WIDTH,
            "row must zero-pad to {INSTRUCTION_WIDTH}"
        );
    }

    /// Slot 0 of an RmsNorm row holds OP_RMS_NORM (=1). Pin the
    /// opcode-render path: i32 literals come through verbatim.
    #[test]
    fn emit_program_renders_opcode_in_slot_zero() {
        let bucket = rms_norm_bucket();
        let ts = emit_prim_mega_program(&ident("MEGA_PROGRAM_M_1"), &bucket);
        let item: syn::ItemStatic = syn::parse2(ts).unwrap();
        let outer = match &*item.expr {
            syn::Expr::Array(a) => a,
            _ => unreachable!(),
        };
        let row = match &outer.elems[0] {
            syn::Expr::Array(a) => a,
            _ => unreachable!(),
        };
        let cell0_lit = match &row.elems[0] {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(li),
                ..
            }) => li.base10_parse::<i32>().unwrap(),
            other => panic!("expected int literal at slot 0, got {:?}", other),
        };
        assert_eq!(cell0_lit, OP_RMS_NORM);
    }

    /// `RowSlot::ConstExpr` token streams render verbatim as a
    /// parenthesized expression. Pin the path the encoder uses for
    /// per-canonical const-prop (`<Weights as
    /// CanonicalParams>::HIDDEN_SIZE as i32`); the encoder doesn't
    /// produce these today, but the emit path must support them
    /// so step-7 follow-ups can route shape constants through them
    /// without a new RowSlot variant.
    #[test]
    fn emit_program_renders_const_expr_verbatim() {
        let mut row = EncodedRow::new(OP_NOP);
        row.push_const_expr(quote! {
            <Weights as ::ferrite_kernels::canonical::CanonicalParams>::HIDDEN_SIZE as i32
        });
        let bucket = EncodedBucket {
            rows: vec![row],
            ptr_plan: Vec::new(),
            runtime_fills: Vec::new(),
        };
        let ts = emit_prim_mega_program(&ident("MEGA_PROGRAM_T"), &bucket);
        let s = ts.to_string();
        assert!(
            s.contains("HIDDEN_SIZE"),
            "ConstExpr token stream must appear verbatim in emitted static; got:\n{s}"
        );
    }

    /// After `assign_ptr_indices`, `RowSlot::Runtime(eps)` becomes
    /// `RowSlot::Const(0)` and gets recorded in `runtime_fills`.
    /// The emitted static therefore has a literal `0` at the eps
    /// position; the launcher (step 7b/7c) patches it per call.
    #[test]
    fn emit_program_renders_zero_for_runtime_filled_slot() {
        let bucket = rms_norm_bucket();
        // Sanity: encoder did record exactly one runtime fill at the
        // eps position (slot 4) before we go check the emitted const.
        assert_eq!(bucket.runtime_fills.len(), 1);
        assert_eq!(
            bucket.runtime_fills[0],
            (
                0,
                4,
                RuntimeSource::WeightEpsBits {
                    fn_ident: "input_layernorm".into(),
                    layer: 0,
                }
            )
        );
        let ts = emit_prim_mega_program(&ident("MEGA_PROGRAM_M_1"), &bucket);
        let item: syn::ItemStatic = syn::parse2(ts).unwrap();
        let outer = match &*item.expr {
            syn::Expr::Array(a) => a,
            _ => unreachable!(),
        };
        let row = match &outer.elems[0] {
            syn::Expr::Array(a) => a,
            _ => unreachable!(),
        };
        let cell4_lit = match &row.elems[4] {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(li),
                ..
            }) => li.base10_parse::<i32>().unwrap(),
            other => panic!("expected int literal at slot 4, got {:?}", other),
        };
        assert_eq!(cell4_lit, 0);
    }

    /// Calling `emit_prim_mega_program` without first running
    /// `assign_ptr_indices` panics — the resolved-rows pre-condition
    /// is a hard codegen invariant. Surfaces at macro expansion, not
    /// at link time when the launcher would otherwise feed garbage
    /// into pt[]. Pin both branches (Ptr + Runtime).
    #[test]
    #[should_panic(expected = "unresolved RowSlot::Ptr")]
    fn emit_program_panics_on_unresolved_ptr() {
        let mut row = EncodedRow::new(OP_RMS_NORM);
        row.push_ptr(PtrSpec::TileSlot(7));
        let bucket = EncodedBucket {
            rows: vec![row],
            ptr_plan: Vec::new(),
            runtime_fills: Vec::new(),
        };
        let _ = emit_prim_mega_program(&ident("MEGA_PROGRAM_T"), &bucket);
    }

    #[test]
    #[should_panic(expected = "unresolved RowSlot::Runtime")]
    fn emit_program_panics_on_unresolved_runtime() {
        let mut row = EncodedRow::new(OP_RMS_NORM);
        row.push_runtime(RuntimeSource::WeightEpsBits {
            fn_ident: "x".into(),
            layer: 0,
        });
        let bucket = EncodedBucket {
            rows: vec![row],
            ptr_plan: Vec::new(),
            runtime_fills: Vec::new(),
        };
        let _ = emit_prim_mega_program(&ident("MEGA_PROGRAM_T"), &bucket);
    }

    /// Multi-row buckets render as `[[..32..], [..32..], ...]` —
    /// outer length is `bucket.rows.len()`. Pin loop-unrolled
    /// programs (every per-iter row gets its own outer entry).
    #[test]
    fn emit_program_outer_length_matches_rows() {
        let arch = arch_with_seven_shapes();
        let body = vec![op(
            "RmsNorm",
            vec![
                quote! { 0u32 },
                quote! { 1u32 },
                quote! { 0u32 },
                quote! { Weights::input_layernorm },
            ],
        )];
        let mut prog: Vec<OpInstance> = vec![loop_instance(4, 1)];
        prog.extend(body);
        let bucket = try_encode_bucket(&arch, &prog, &ctx_llama_1b()).unwrap();
        assert_eq!(bucket.rows.len(), 4);
        let ts = emit_prim_mega_program(&ident("MEGA_PROGRAM_M_1"), &bucket);
        let item: syn::ItemStatic = syn::parse2(ts).unwrap();
        let outer = match &*item.expr {
            syn::Expr::Array(a) => a,
            _ => unreachable!(),
        };
        assert_eq!(outer.elems.len(), 4);
    }
}
