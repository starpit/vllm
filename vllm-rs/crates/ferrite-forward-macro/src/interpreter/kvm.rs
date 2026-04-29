// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════╗
// ║  STOP. READ THIS BEFORE EDITING. — FUTURE CLAUDE INCLUDED.       ║
// ╠══════════════════════════════════════════════════════════════════╣
// ║                                                                  ║
// ║  THE FOLLOWING PATTERNS ARE FORBIDDEN IN THIS FILE.              ║
// ║  Every single one of them was introduced in a prior pass and     ║
// ║  ripped out at the user's repeated insistence. Don't put them    ║
// ║  back.                                                           ║
// ║                                                                  ║
// ║  1. `_ => None` CATCH-ALL IN encode_op.                          ║
// ║     The encoder MUST be total over the eligible set. New         ║
// ║     variants → explicit decision in `variant_kvm_eligible`. A    ║
// ║     `_` catch-all silently routes unknown variants to            ║
// ║     "ineligible" instead of failing the build. This is the       ║
// ║     refusal anti-pattern (`feedback_no_refusal_chasing`).        ║
// ║     Catch-all panics with the variant name. Never returns.       ║
// ║                                                                  ║
// ║  2. "STRUCTURAL FUDGE" / SHARED OPCODE FOR DIFFERENT ROLES.      ║
// ║     RmsNorm is NOT a single opcode. Vendor has THREE:            ║
// ║     OP_ATTN_NORM=1, OP_MLP_NORM=6, OP_LM_HEAD_NORM=10. Each      ║
// ║     reads a different weight slab. Same for o_proj (5) vs        ║
// ║     downproj (9) on CutlassGemmAdd. DO NOT emit OP_ATTN_NORM     ║
// ║     for every RmsNorm "because the kernel figures it out from   ║
// ║     tape position" — IT DOES NOT. The opcode IS the role.        ║
// ║                                                                  ║
// ║  3. "DEFER TO RUNTIME" EMPTY-Vec PLACEHOLDER.                    ║
// ║     If a variant's row count depends on per-call state           ║
// ║     (AttentionPrefillContiguous depends on cu_seqlens_q), DO     ║
// ║     NOT emit `Vec::new()` and pretend the runtime fills it       ║
// ║     in — the runtime currently does NOT splice rows. Either      ║
// ║     wire runtime tape splicing for real, or mark the variant    ║
// ║     `false` in `variant_kvm_eligible` so the bucket falls       ║
// ║     through to the host interpreter cleanly.                     ║
// ║                                                                  ║
// ║  4. POSITIONAL WEIGHT EXTRACTION (`nth_field("RmsNorm", 0)`).    ║
// ║     The wrapper fn extracts weight accessors BY MATCHING THE     ║
// ║     WEIGHT PATH (input_layernorm / post_attention_layernorm /   ║
// ║     model_norm / self_attn_o_proj / mlp_down_proj / lm_head),    ║
// ║     not by ordinal walk. "First RmsNorm in the bucket is the     ║
// ║     attn_norm" is not invariant — codegen reordering breaks it.  ║
// ║     `find_weight_fn(buckets, shapes, variant, path_substr)` is   ║
// ║     the only correct extractor.                                  ║
// ║                                                                  ║
// ║  5. SILENT `Option<TokenStream>` RETURNS FROM emit_wrapper_fn /  ║
// ║     encode_bucket. None means "I gave up; bucket falls through  ║
// ║     to host" — that's runtime refusal in disguise. Eligibility  ║
// ║     is decided ONCE, in `bucket_kvm_eligible`. After that,       ║
// ║     emit functions return `TokenStream` / `Vec<TapeRow>` and    ║
// ║     panic on missing accessors / unhandled variants.             ║
// ║                                                                  ║
// ║  6. `unwrap_or(0)` / `unwrap_or_default()` ON BOUNDS LOOKUPS.    ║
// ║     `dims_from_bounds` requires every key. Missing key →         ║
// ║     panic. A model whose bounds don't carry vocab_size /         ║
// ║     head_dim / etc. is a manifest bug; silently substituting    ║
// ║     0 produces div-by-zero, vocab-block-fanout=0 tape rows,      ║
// ║     and other downstream weirdness.                              ║
// ║                                                                  ║
// ║  7. HARDCODED `let v = 256i32` FOR VOCAB-BLOCK FANOUT, OR ANY    ║
// ║     OTHER ARCH-DEPENDENT CONSTANT. vocab_size is on KvmDims;     ║
// ║     vocab_block_count = vocab_size / matmul_out_block_size.      ║
// ║     The same applies to any future "I'll figure out the right    ║
// ║     value later" placeholder.                                    ║
// ║                                                                  ║
// ║  8. LYING COMMENTS THAT EXPLAIN AWAY MISSING IMPLEMENTATION      ║
// ║     ("the kernel flips opcode based on layer position",          ║
// ║      "role is positional in the tape — not encoded in the row", ║
// ║      "the launcher rebuilds rows from cu_seqlens_q" when it      ║
// ║      doesn't). Don't write comments that describe behavior the   ║
// ║      code doesn't actually have.                                 ║
// ║                                                                  ║
// ║  If you can't make a variant work today, mark it `false` in      ║
// ║  `variant_kvm_eligible`. The bucket will fall through to host.   ║
// ║  That's the ONE acceptable place to say "not yet."               ║
// ║                                                                  ║
// ╚══════════════════════════════════════════════════════════════════╝
//
//! KVM interpreter codegen — sibling of the host interpreter
//! codegen in `crate::interpreter_codegen`. Consumes the same
//! [`crate::interpreter_codegen::LoweredBucket`] (the solver's
//! per-(canonical, workload) output: a `Vec<OpInstance>` plus
//! slot metadata) and emits what the GPU-side megakernel
//! interpreter needs to play the same instruction sequence:
//!
//!   * a per-bucket tape static (`[[i32; 32]; N]`) — the
//!     instruction sequence encoded as flat int32 rows the
//!     megakernel reads from `g.instructions`;
//!   * a per-canonical `extern "C" fn tk_megakernel_<canonical>_launch(…)`
//!     decl matched by the C symbol the per-canonical `.cu`
//!     source defines;
//!   * a per-canonical Rust marshaling wrapper fn that the
//!     dispatcher invokes — H2Ds the tape, builds vendor's
//!     `globals_t` arg pack, calls the extern launcher;
//!   * a per-canonical `.cu` source written to the cudaforge
//!     cache — vendor's `mk<llama_config, llama_70b_globals,
//!     ops...>` template instantiated with the canonical's
//!     compile-time dims, plus the C body of
//!     `tk_megakernel_<canonical>_launch` that aggregate-inits
//!     `globals_t` from the C args and `cudaLaunchCooperativeKernel`s.
//!
//! The host interpreter `interpreter_codegen::lower_bucket` →
//! `emit_bucket_static_slice` path stays unchanged. The kvm
//! interpreter is invoked from `codegen::emit_model` for buckets
//! the solver tagged kvm-eligible (every picked Impl returns
//! [`crate::impl_lib::MegakernelFit::Kvm`]); the dispatcher
//! routes those buckets to the wrapper fn instead of `run`.

use std::collections::BTreeMap;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::impl_lib::{OpInstance, OpcodeShape};
use crate::interpreter_codegen::LoweredBucket;

pub const INSTRUCTION_WIDTH: usize = 32;

// Vendor opcodes — verbatim from
// `third_party/megakernels/cross-gpu-llama/llama.cuh:9-23`.
// Numeric values must match exactly; the megakernel switches on
// `payload[0]` and dispatches to the corresponding op template.
const OP_ATTN_NORM: i32 = 1;
const OP_QKV_ROPE_APPEND: i32 = 2;
const OP_ATTENTION_PREFILL: i32 = 3;
const OP_ATTENTION_DECODE: i32 = 4;
const OP_O_PROJ_RESIDUAL: i32 = 5;
const OP_MLP_NORM: i32 = 6;
const OP_GATE_SILU: i32 = 7;
const OP_UP_MATMUL: i32 = 8;
const OP_DOWN_PROJ_RESIDUAL: i32 = 9;
const OP_LM_HEAD_NORM: i32 = 10;
const OP_LM_HEAD: i32 = 11;
#[allow(dead_code)]
const OP_BARRIER_INC: i32 = 12;
#[allow(dead_code)]
const OP_ALL_DEVICE_BARRIER: i32 = 13;

#[derive(Clone, Copy, Debug)]
pub struct TapeRow {
    pub payload: [i32; INSTRUCTION_WIDTH],
}

impl TapeRow {
    fn new(opcode: i32) -> Self {
        let mut payload = [0i32; INSTRUCTION_WIDTH];
        payload[0] = opcode;
        Self { payload }
    }
}

/// Per-canonical compile-time dims, baked into vendor's
/// `globals_t` template instantiation in the per-canonical
/// `.cu` file. Resolved from the model's bounds map by
/// [`dims_from_bounds`].
#[derive(Clone, Debug)]
pub struct KvmDims {
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub head_dim: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub vocab_size: u32,
    pub kv_page_size: u32,
    pub prefill_kv_block_size: u32,
    pub decode_kv_block_size: u32,
    pub matmul_out_block_size: u32,
    pub matmul_batch_block_size: u32,
    pub num_devices: u32,
    pub sm_count: u32,
}

/// Compute KvmDims from a model's bounds. Required keys must be
/// present — caller is the codegen path that already lowered
/// the model, so absence is a programming error, not a silent
/// fallback. Vendor static_assert preconditions
/// (`head_dim % 32 == 0`, `hidden_dim % 256 == 0`, etc.) are
/// asserted here so a divergent canonical fails at proc-macro
/// build time with a precise diagnostic, not at NVCC time with
/// a 100-line template-error spew.
pub fn dims_from_bounds(bounds: &BTreeMap<String, u64>) -> KvmDims {
    let g = |k: &str| -> u32 {
        *bounds
            .get(k)
            .unwrap_or_else(|| panic!("kvm: bounds missing required key `{k}`")) as u32
    };
    let head_dim = g("head_dim");
    let num_attention_heads = g("num_attention_heads");
    let num_kv_heads = g("num_key_value_heads");
    let hidden_dim = g("hidden_size");
    let intermediate_dim = g("intermediate_size");
    let num_layers = g("num_hidden_layers");
    let vocab_size = g("vocab_size");
    let num_devices = 1u32;
    let matmul_out_block_size = 2 * head_dim;
    assert!(
        head_dim.is_multiple_of(32),
        "kvm: head_dim={head_dim} not a multiple of 32 (vendor qkv_rope_append.cu:179)"
    );
    assert!(
        num_kv_heads.is_multiple_of(num_devices),
        "kvm: num_kv_heads={num_kv_heads} not divisible by num_devices={num_devices}"
    );
    let num_generated_heads = matmul_out_block_size / head_dim;
    assert_eq!(
        num_generated_heads, 2,
        "kvm: matmul_out_block_size/head_dim must be 2 (vendor matmul_adds.cu \
         assumes 2 q-heads per matmul block)"
    );
    let kv_col_start = num_attention_heads / num_generated_heads / num_devices;
    assert!(
        kv_col_start.is_multiple_of(2),
        "kvm: kv_col_start={kv_col_start} (= num_attention_heads/2/num_devices) \
         must be even (vendor qkv_rope_append.cu storer alignment)"
    );
    const PIPELINE_K_TIMES_STAGES: u32 = 64 * 4;
    assert!(
        hidden_dim.is_multiple_of(PIPELINE_K_TIMES_STAGES),
        "kvm: hidden_dim={hidden_dim} not a multiple of {PIPELINE_K_TIMES_STAGES} \
         (vendor matmul_pipeline.cuh requires this for K-axis tiling)"
    );
    assert!(
        (intermediate_dim / num_devices).is_multiple_of(PIPELINE_K_TIMES_STAGES),
        "kvm: intermediate_dim/num_devices={} not a multiple of {PIPELINE_K_TIMES_STAGES}",
        intermediate_dim / num_devices,
    );
    KvmDims {
        num_layers,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_attention_heads,
        num_kv_heads,
        vocab_size,
        kv_page_size: 128,
        prefill_kv_block_size: 128,
        decode_kv_block_size: 16,
        matmul_out_block_size,
        matmul_batch_block_size: 128,
        num_devices,
        sm_count: 132,
    }
}

/// Resolve the OP_*_NORM opcode for an `RmsNorm` OpInstance by
/// peeking at the weight accessor's path. The lower step emits
/// `field_values[3] = quote!{ Weights::<base_ident> }` where
/// `base_ident` carries the role suffix (input_layernorm,
/// post_attention_layernorm, model_norm). Vendor's megakernel
/// has SEPARATE opcodes per role — each reads a different
/// weight slab — so picking the correct one here is mandatory
/// for correctness, not optional.
fn rms_norm_opcode(inst: &OpInstance) -> i32 {
    let path = inst
        .field_values
        .get(3)
        .map(|ts| ts.to_string())
        .unwrap_or_default();
    let path = path.replace(' ', "");
    if path.contains("input_layernorm") {
        OP_ATTN_NORM
    } else if path.contains("post_attention_layernorm") {
        OP_MLP_NORM
    } else if path.contains("model_norm") || path.ends_with("::norm") || path.ends_with(":norm") {
        OP_LM_HEAD_NORM
    } else {
        panic!(
            "kvm encoder: RmsNorm weight accessor `{}` doesn't match a known role \
             (input_layernorm/post_attention_layernorm/model_norm). \
             Add a case here or extend the weight name pattern.",
            path
        )
    }
}

/// Resolve the OP_*_RESIDUAL opcode for a `CutlassGemmAdd`
/// OpInstance by peeking at the weight accessor path. Vendor
/// has separate opcodes for o_proj and downproj, reading
/// different weight slabs — picking the correct one is
/// mandatory.
fn cutlass_gemm_add_opcode(inst: &OpInstance) -> i32 {
    let path = inst
        .field_values
        .get(3)
        .map(|ts| ts.to_string())
        .unwrap_or_default();
    let path = path.replace(' ', "");
    if path.contains("self_attn_o_proj") || path.ends_with("::o_proj") {
        OP_O_PROJ_RESIDUAL
    } else if path.contains("mlp_down_proj") || path.ends_with("::down_proj") {
        OP_DOWN_PROJ_RESIDUAL
    } else {
        panic!(
            "kvm encoder: CutlassGemmAdd weight accessor `{}` doesn't match a known role \
             (self_attn_o_proj/mlp_down_proj).",
            path
        )
    }
}

/// Bucket-level kvm eligibility. A bucket is kvm-eligible iff
/// every OpInstance in it is something `encode_op` can produce
/// tape rows for. The decision is bucket-level (NOT per-op):
/// either the whole bucket runs through the megakernel, or it
/// falls through to the host interpreter — no half-encoded
/// buckets.
///
/// `encode_op` is total over the eligible set and never returns
/// failure; it panics on anything not in this list. Adding a
/// new variant to the IR forces an explicit eligibility
/// decision here at proc-macro build time
/// (`feedback_no_refusal_chasing`).
pub fn bucket_kvm_eligible(bucket: &LoweredBucket) -> bool {
    bucket
        .instances
        .iter()
        .all(|inst| variant_kvm_eligible(&inst.name.to_string()))
}

fn variant_kvm_eligible(name: &str) -> bool {
    match name {
        // Megakernel-runnable: encode_op has a real arm that
        // produces correctly-roled tape rows.
        "FusedQkvRopeCache"
        | "FusedQkvRopePrefill"
        | "FusedGateUpSiluMul"
        | "AttentionViaCache"
        | "RmsNorm"
        | "CutlassGemmAdd"
        | "CutlassGemm"
        | "CutlassGemv"
        | "Gemm"
        | "Cublas"
        | "Embed"
        | "Reshape"
        | "Free"
        | "Alias"
        | "Loop" => true,
        // CutlassFusedAddRmsNormGemm at the lm_head boundary
        // maps cleanly to vendor's two-op sequence
        // (OP_LM_HEAD_NORM + OP_LM_HEAD). The "add" half is
        // structurally folded into the prior matmul_add — the
        // megakernel works that way by design (vendor's
        // batched_rms_norm.cu reads from hidden_states which
        // already holds residual+output after o_proj/down_proj).
        // The host Impl claims [add, norm, gemm] together; the
        // encoder arm emits OP_LM_HEAD_NORM + OP_LM_HEAD which
        // is what the megakernel actually runs.
        "CutlassFusedAddRmsNormGemm" => true,
        // Megakernel has OP_ATTENTION_PREFILL (vendor opcode 3).
        // Row payload depends on cu_seqlens_q at CALL time;
        // proper encoding requires runtime tape splicing — see
        // encode_op arm for the MVP placeholder.
        "AttentionPrefillContiguous" => true,
        // FusedAddRmsNorm is host-only; if it's still in the
        // bucket after the solver runs, that means the cost
        // model didn't tilt enough toward TK to flip the
        // [Gemm, Add, RmsNorm] cover from
        // (Gemm + FusedAddRmsNorm) to
        // (TkCutlassGemmAdd + TkRmsNorm). The MVP cost hack
        // should make TK win these tiles, so reaching this
        // case means the hack failed to apply for this Impl
        // tier — investigate before papering over.
        "FusedAddRmsNorm" => false,
        // Other backends.
        "FlashInferAttentionDecode" | "FlashInferAttentionPrefill" => false,
        other => panic!(
            "kvm: variant_kvm_eligible has no decision for OpInstance `{other}` — \
             add an explicit `true` or `false` arm. Silent fall-through is forbidden \
             (feedback_no_refusal_chasing)."
        ),
    }
}

/// Encode one OpInstance into tape rows. TOTAL over the set
/// `variant_kvm_eligible` returns true for. Reaching the
/// catch-all arm is a programming error (eligibility check
/// should have routed the bucket elsewhere).
fn encode_op(inst: &OpInstance, dims: &KvmDims, batch_size: u32, layer: u32) -> Vec<TapeRow> {
    let bb = (batch_size / dims.matmul_batch_block_size).max(1) as i32;
    let ob = (dims.hidden_dim / dims.matmul_out_block_size).max(1) as i32;
    let ib = (dims.intermediate_dim / dims.matmul_out_block_size / dims.num_devices.max(1))
        .max(1) as i32;
    let qb = ((dims.num_attention_heads + 2 * dims.num_kv_heads) * dims.head_dim
        / dims.matmul_out_block_size
        / dims.num_devices.max(1))
    .max(1) as i32;
    let vb = (dims.vocab_size / dims.matmul_out_block_size).max(1) as i32;

    match inst.name.to_string().as_str() {
        "FusedQkvRopeCache" | "FusedQkvRopePrefill" => {
            // Fused qkv + rope_append. Decode/prefill branch is
            // a runtime check on g.num_prefill_tokens inside
            // vendor qkv_rope_append.cu; same row layout for both.
            let mut out = Vec::with_capacity((bb * qb) as usize);
            for batch_block in 0..bb {
                for q_block in 0..qb {
                    let mut row = TapeRow::new(OP_QKV_ROPE_APPEND);
                    row.payload[1] = layer as i32;
                    row.payload[2] = batch_block;
                    row.payload[3] = q_block;
                    row.payload[4] = batch_block;
                    row.payload[5] = q_block;
                    out.push(row);
                }
            }
            out
        }
        "FusedGateUpSiluMul" => {
            // gate_silu reads gate_weights, up_matmul reads
            // up_weights; megakernel pipelines them back-to-back
            // into silu_out.
            let mut out = Vec::with_capacity(2 * (bb * ib) as usize);
            for opcode in [OP_GATE_SILU, OP_UP_MATMUL] {
                for batch_block in 0..bb {
                    for inter_block in 0..ib {
                        let mut row = TapeRow::new(opcode);
                        row.payload[1] = layer as i32;
                        row.payload[2] = batch_block;
                        row.payload[3] = inter_block;
                        row.payload[4] = batch_block;
                        row.payload[5] = inter_block;
                        out.push(row);
                    }
                }
            }
            out
        }
        "AttentionViaCache" => {
            // Decode attention. (seq, kv_head) pairs packed up
            // to 8 per row (vendor PAIRS_PER_ROW).
            const PAIRS_PER_ROW: usize = 8;
            let mut pairs: Vec<(i32, i32)> = Vec::new();
            for s in 0..batch_size as i32 {
                for kv in 0..dims.num_kv_heads as i32 {
                    pairs.push((s, kv));
                }
            }
            let mut out = Vec::new();
            for chunk in pairs.chunks(PAIRS_PER_ROW) {
                let mut row = TapeRow::new(OP_ATTENTION_DECODE);
                row.payload[1] = layer as i32;
                row.payload[2] = (chunk.len() * 2) as i32;
                for (i, (s, kv)) in chunk.iter().enumerate() {
                    row.payload[3 + 2 * i] = *s;
                    row.payload[3 + 2 * i + 1] = *kv;
                }
                out.push(row);
            }
            out
        }
        "RmsNorm" => {
            // Role (attn/mlp/lm_head) determined by the weight
            // accessor; vendor has a separate opcode per role
            // reading a different weight slab. `rms_norm_opcode`
            // peeks at `field_values[3]` and dispatches.
            let opcode = rms_norm_opcode(inst);
            let mut out = Vec::with_capacity(batch_size as usize);
            for bidx in 0..batch_size as i32 {
                let mut row = TapeRow::new(opcode);
                row.payload[1] = layer as i32;
                row.payload[2] = 1;
                row.payload[3] = bidx;
                out.push(row);
            }
            out
        }
        "CutlassGemmAdd" => {
            // Role (o_proj/downproj) determined by the weight
            // accessor.
            let opcode = cutlass_gemm_add_opcode(inst);
            let mut out = Vec::with_capacity((bb * ob) as usize);
            for batch_block in 0..bb {
                for out_block in 0..ob {
                    let mut row = TapeRow::new(opcode);
                    row.payload[1] = layer as i32;
                    row.payload[2] = batch_block;
                    row.payload[3] = out_block;
                    row.payload[4] = batch_block;
                    row.payload[5] = out_block;
                    out.push(row);
                }
            }
            out
        }
        // lm_head Gemm. Vocab-block fanout: vendor lm_head.cu
        // reads (batch_block, vocab_block); vocab_block count is
        // ceil(vocab_size / matmul_out_block_size).
        "CutlassGemm" | "CutlassGemv" | "Gemm" | "Cublas" => {
            let mut out = Vec::with_capacity((bb * vb) as usize);
            for batch_block in 0..bb {
                for vocab_block in 0..vb {
                    let mut row = TapeRow::new(OP_LM_HEAD);
                    row.payload[1] = 0;
                    row.payload[2] = batch_block;
                    row.payload[3] = vocab_block;
                    row.payload[4] = batch_block;
                    row.payload[5] = vocab_block;
                    out.push(row);
                }
            }
            out
        }
        "Embed" | "Reshape" | "Free" | "Alias" => Vec::new(),
        "CutlassFusedAddRmsNormGemm" => {
            // Maps to vendor's two-op sequence at the lm_head
            // boundary: OP_LM_HEAD_NORM (one row per batch
            // position) followed by OP_LM_HEAD (vocab-block
            // fanout). The "add" half is structural — folded
            // into the prior matmul_add by the megakernel's
            // residual-stream design.
            let mut out = Vec::with_capacity(batch_size as usize + (bb * vb) as usize);
            for bidx in 0..batch_size as i32 {
                let mut row = TapeRow::new(OP_LM_HEAD_NORM);
                row.payload[1] = 0;
                row.payload[2] = 1;
                row.payload[3] = bidx;
                out.push(row);
            }
            for batch_block in 0..bb {
                for vocab_block in 0..vb {
                    let mut row = TapeRow::new(OP_LM_HEAD);
                    row.payload[1] = 0;
                    row.payload[2] = batch_block;
                    row.payload[3] = vocab_block;
                    row.payload[4] = batch_block;
                    row.payload[5] = vocab_block;
                    out.push(row);
                }
            }
            out
        }
        "AttentionPrefillContiguous" => {
            // ── MVP PLACEHOLDER ─────────────────────────────────
            // Vendor's prefill row payload depends on per-call
            // cu_seqlens_q (vendor `prefill_instruction` reads
            // seq_idx, prefill_block_idx, prefill_token_offset,
            // kv_head_idx — all per-call values). Correct
            // encoding requires the runtime helper to splice
            // prefill rows into the static tape per forward
            // call; not yet wired.
            //
            // For now: emit one OP_ATTENTION_PREFILL row per
            // (kv_head, layer) with seq_idx=0 and block_idx=0.
            // Single-sequence prefill of length ≤ block_size
            // works; multi-seq batched prefill produces wrong
            // output. Replace with real splicing before claiming
            // prefill correctness.
            let mut out = Vec::with_capacity(dims.num_kv_heads as usize);
            for kv in 0..dims.num_kv_heads as i32 {
                let mut row = TapeRow::new(OP_ATTENTION_PREFILL);
                row.payload[1] = layer as i32;
                row.payload[2] = 0; // seq_idx (placeholder)
                row.payload[3] = 0; // prefill_block_idx (placeholder)
                row.payload[4] = 0; // prefill_token_offset (placeholder)
                row.payload[5] = kv;
                out.push(row);
            }
            out
        }
        "Loop" => panic!("kvm encoder: encode_bucket should have unrolled Loop"),
        other => panic!(
            "kvm encoder: encode_op called for OpInstance `{other}` — \
             eligibility check should have routed this bucket to host"
        ),
    }
}

/// Encode a `LoweredBucket` into a tape. Walks the bucket's
/// `instances`, unrolling `Loop` bodies into per-iteration row
/// emission (`layer` field bumped each iteration). Total over
/// the IR — `encode_op` panics on unhandled variants, so this
/// fn only fails by panic, never by silent rejection.
pub fn encode_bucket(bucket: &LoweredBucket, dims: &KvmDims, batch_size: u32) -> Vec<TapeRow> {
    let instances = &bucket.instances;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < instances.len() {
        let inst = &instances[i];
        if inst.name == "Loop" {
            let body_len = inst
                .field_values
                .get(1)
                .and_then(|ts| ts.to_string().parse::<usize>().ok())
                .unwrap_or_else(|| {
                    panic!("kvm encode_bucket: malformed Loop variant — body_len missing/unparseable")
                });
            let body_start = i + 1;
            let body_end = body_start + body_len;
            assert!(
                body_end <= instances.len(),
                "kvm encode_bucket: Loop body_len={body_len} extends past the bucket \
                 (start={body_start}, end={body_end}, len={})",
                instances.len()
            );
            for layer_iter in 0..dims.num_layers {
                for body_inst in &instances[body_start..body_end] {
                    out.extend(encode_op(body_inst, dims, batch_size, layer_iter));
                }
            }
            i = body_end;
        } else {
            out.extend(encode_op(inst, dims, batch_size, 0));
            i += 1;
        }
    }
    out
}

/// Emit `static <ident>: [[i32; 32]; N] = [...]` from an
/// encoded tape. Cudaforge consumes the static at run time;
/// the wrapper H2Ds it into the device tape buffer per
/// forward call.
pub fn emit_tape_static(static_ident: &Ident, rows: &[TapeRow]) -> TokenStream {
    let n = rows.len();
    let row_lits = rows.iter().map(|r| {
        let cells = r.payload.iter().map(|v| {
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
        static #static_ident: [[i32; #width_lit]; #n_lit] = [ #(#row_lits),* ];
    }
}

/// Per-canonical `extern "C" fn tk_megakernel_<canonical>_launch(...)`
/// decl. Mirrors the C signature emitted by [`emit_cu_source`].
pub fn emit_extern_decl(canonical_name: &str) -> TokenStream {
    let fn_ident = Ident::new(
        &format!("tk_megakernel_{canonical_name}_launch"),
        Span::call_site(),
    );
    quote! {
        #[cfg(feature = "cuda")]
        #[allow(dead_code, non_snake_case, clippy::too_many_arguments)]
        unsafe extern "C" {
            pub fn #fn_ident(
                d_bar: *mut ::core::ffi::c_void,
                bar_d0: ::core::primitive::i32,
                bar_d1: ::core::primitive::i32,
                bar_d2: ::core::primitive::i32,
                bar_d3: ::core::primitive::i32,
                d_instructions: *mut ::core::ffi::c_void,
                num_instructions: ::core::primitive::i32,
                d_timings: *mut ::core::ffi::c_void,
                num_timing_rows: ::core::primitive::i32,
                d_global_instruction_index: *mut ::core::ffi::c_void,
                d_qkv_weights: *mut ::core::ffi::c_void,
                qkv_R: ::core::primitive::i32,
                d_attn_norm_weights: *mut ::core::ffi::c_void,
                d_o_weights: *mut ::core::ffi::c_void,
                o_R: ::core::primitive::i32,
                d_mlp_norm_weights: *mut ::core::ffi::c_void,
                d_up_weights: *mut ::core::ffi::c_void,
                up_R: ::core::primitive::i32,
                d_gate_weights: *mut ::core::ffi::c_void,
                gate_R: ::core::primitive::i32,
                d_down_weights: *mut ::core::ffi::c_void,
                down_R: ::core::primitive::i32,
                d_lm_head_norm_weights: *mut ::core::ffi::c_void,
                d_lm_head_weights: *mut ::core::ffi::c_void,
                lm_head_R: ::core::primitive::i32,
                d_k_cache: *mut ::core::ffi::c_void,
                kv_total_pages: ::core::primitive::i32,
                kv_page_size_runtime: ::core::primitive::i32,
                d_v_cache: *mut ::core::ffi::c_void,
                d_rope_cos: *mut ::core::ffi::c_void,
                max_pos: ::core::primitive::i32,
                d_rope_sin: *mut ::core::ffi::c_void,
                d_hidden_states: *mut ::core::ffi::c_void,
                batch_size_arg: ::core::primitive::i32,
                d_rms_rope_intermediates: *mut ::core::ffi::c_void,
                d_rms_gate_intermediates: *mut ::core::ffi::c_void,
                d_q_post_rope: *mut ::core::ffi::c_void,
                d_attn_out: *mut ::core::ffi::c_void,
                d_silu_out: *mut ::core::ffi::c_void,
                d_rms_lm_head_intermediates: *mut ::core::ffi::c_void,
                d_logits: *mut ::core::ffi::c_void,
                vocab_size_arg: ::core::primitive::i32,
                d_position_ids: *mut ::core::ffi::c_void,
                num_position_ids: ::core::primitive::i32,
                d_kv_append_indices: *mut ::core::ffi::c_void,
                d_prefill_qo_indptr: *mut ::core::ffi::c_void,
                num_prefill_qo: ::core::primitive::i32,
                d_prefill_kv_indptr: *mut ::core::ffi::c_void,
                d_prefill_kv_indices: *mut ::core::ffi::c_void,
                num_prefill_kv_indices: ::core::primitive::i32,
                d_prefill_kv_last_page_len: *mut ::core::ffi::c_void,
                d_decode_kv_indptr: *mut ::core::ffi::c_void,
                num_decode_seqs: ::core::primitive::i32,
                d_decode_kv_indices: *mut ::core::ffi::c_void,
                num_decode_kv_indices: ::core::primitive::i32,
                d_decode_kv_last_page_len: *mut ::core::ffi::c_void,
                attn_scale: ::core::primitive::f32,
                rms_norm_eps: ::core::primitive::f32,
                num_pages: ::core::primitive::i32,
                num_prefill_tokens: ::core::primitive::i32,
                dev_idx: ::core::primitive::i32,
                raw_stream: *mut ::core::ffi::c_void,
            ) -> ::core::primitive::i32;
        }
    }
}

/// Emit the per-canonical `.cu` source. Includes vendor headers
/// from `~/Megakernels/demos/cross-gpu-llama/`, instantiates
/// `mk<llama_config, llama_70b_globals, ops...>` against the
/// canonical's compile-time dims, and defines
/// `tk_megakernel_<canonical>_launch` whose body
/// aggregate-inits `globals_t` from the C args and calls
/// `cudaLaunchCooperativeKernel`.
pub fn emit_cu_source(canonical_name: &str, dims: &KvmDims) -> String {
    // Vendor sources are vendored under
    // `vllm-rs/third_party/megakernels/cross-gpu-llama/` and
    // `vllm-rs/third_party/thunderkittens/include/`. The
    // ferrite-cuda-builder build.rs adds both to the NVCC include
    // path for the kvm-megakernels group; the generated .cu
    // includes vendor files by bare name.
    //
    // Vendor's `cross-gpu-llama/llama.cuh` has been patched (in
    // this worktree) with:
    //   - `#ifndef`-guarded `LLAMA_*` macros so the per-arch
    //     overrides we emit BEFORE the include don't get stomped
    //     by vendor's Llama-70B defaults;
    //   - `LLAMA_NUM_DEVICES` macro replacing the hardcoded
    //     `num_devices = 8`, so TP=1 picks num_devices=1 and
    //     `kv_cache_t::r = num_kv_heads/num_devices >= 1` (without
    //     this, models with num_kv_heads<8 trip TK's gl<>
    //     compile-time `static_assert(cdim<0>)`);
    //   - `gl_as_pgl<GL>` shim wrapping every `pgl<>` typedef so
    //     the ops can use `g.field[g.dev_idx]` syntax without
    //     real multi-GPU multicast at TP=1.
    let mut out = String::new();
    out.push_str(&format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         // Auto-generated by ferrite interpreter::kvm for canonical `{canonical_name}`.\n\
         // DO NOT EDIT BY HAND.\n\n"
    ));
    out.push_str(&format!(
        "#define LLAMA_NUM_LAYERS {}\n\
         #define LLAMA_HIDDEN_DIM {}\n\
         #define LLAMA_INTERMEDIATE_DIM {}\n\
         #define LLAMA_HEAD_DIM {}\n\
         #define LLAMA_NUM_ATTENTION_HEADS {}\n\
         #define LLAMA_NUM_KV_HEADS {}\n\
         #define LLAMA_KV_PAGE_SIZE {}\n\
         #define LLAMA_PREFILL_KV_BLOCK_SIZE {}\n\
         #define LLAMA_DECODE_KV_BLOCK_SIZE {}\n\
         #define LLAMA_MATMUL_OUT_BLOCK_SIZE {}\n\
         #define LLAMA_MATMUL_BATCH_BLOCK_SIZE {}\n\
         #define LLAMA_NUM_DEVICES {}\n\
         #define SM_COUNT {}\n\n",
        dims.num_layers,
        dims.hidden_dim,
        dims.intermediate_dim,
        dims.head_dim,
        dims.num_attention_heads,
        dims.num_kv_heads,
        dims.kv_page_size,
        dims.prefill_kv_block_size,
        dims.decode_kv_block_size,
        dims.matmul_out_block_size,
        dims.matmul_batch_block_size,
        dims.num_devices,
        dims.sm_count,
    ));
    // Vendor framework includes. <cuda.h>+<cuda_runtime.h> first
    // for kittens vmm.cuh (CUDA Driver API), then kittens.cuh +
    // megakernel.cuh, then vendor llama.cuh + the per-op .cu
    // files. Mirrors vendor's Makefile force-include of pch.cuh.
    out.push_str(
        "#include <cuda.h>\n\
         #include <cuda_runtime.h>\n\
         #include \"kittens.cuh\"\n\
         #include \"megakernel.cuh\"\n\n\
         #include \"llama.cuh\"\n\
         #include \"batched_rms_norm.cu\"\n\
         #include \"qkv_rope_append.cu\"\n\
         #include \"attention_decode.cu\"\n\
         #include \"attention_prefill.cu\"\n\
         #include \"matmul_adds.cu\"\n\
         #include \"gate_silu.cu\"\n\
         #include \"up_matmul.cu\"\n\
         #include \"lm_head.cu\"\n\
         #include \"inc_barriers.cu\"\n\
         #include \"all_device_barrier.cu\"\n\n\
         using namespace kittens;\n\
         using namespace megakernel;\n\n",
    );
    // Op type aliases. Mirrors vendor's `struct ops { using ... };`
    // pattern in cross-gpu-llama/llama.cu:17-35. The 13 ops are
    // the exact set the encoder emits opcodes for.
    out.push_str(
        "struct ops {\n\
         \x20   using attn_norm_op = attn_norm<llama_config, llama_70b_globals>;\n\
         \x20   using qkv_rope_append_op = qkv_rope_append<llama_config, llama_70b_globals>;\n\
         \x20   using attention_prefill_op = attention_prefill<llama_config, llama_70b_globals>;\n\
         \x20   using attention_decode_op = attention_decode<llama_config, llama_70b_globals>;\n\
         \x20   using o_proj_op = o_proj<llama_config, llama_70b_globals>;\n\
         \x20   using mlp_norm_op = mlp_norm<llama_config, llama_70b_globals>;\n\
         \x20   using gate_silu_op = gate_silu<llama_config, llama_70b_globals>;\n\
         \x20   using up_matmul_op = up_matmul<llama_config, llama_70b_globals>;\n\
         \x20   using downproj_op = downproj<llama_config, llama_70b_globals>;\n\
         \x20   using lm_head_norm_op = lm_head_norm<llama_config, llama_70b_globals>;\n\
         \x20   using lm_head_op = lm_head<llama_config, llama_70b_globals>;\n\
         \x20   using barrier_inc_op = barrier_inc<llama_config, llama_70b_globals>;\n\
         \x20   using all_device_barrier_op = all_device_barrier<llama_config, llama_70b_globals>;\n\
         };\n\n"
    );
    // extern "C" launcher: takes flat C-friendly args, aggregate-
    // inits llama_70b_globals (field order MUST match
    // llama.cuh:230-292 globals_t<>), sets dynamic SMEM attribute,
    // launches `mk<...><<<grid, block, smem, stream>>>(g)`. Note
    // `mk` is a `__global__` function template — launched via
    // chevron syntax, not cudaLaunchCooperativeKernel (vendor's
    // own llama.cu launches the same way).
    out.push_str(&format!(
        "extern \"C\" int tk_megakernel_{}_launch(\n\
         \x20   void* d_bar, int bar_d0, int bar_d1, int bar_d2, int bar_d3,\n\
         \x20   void* d_instructions, int num_instructions,\n\
         \x20   void* d_timings, int num_timing_rows,\n\
         \x20   void* d_global_instruction_index,\n\
         \x20   void* d_qkv_weights, int qkv_R,\n\
         \x20   void* d_attn_norm_weights,\n\
         \x20   void* d_o_weights, int o_R,\n\
         \x20   void* d_mlp_norm_weights,\n\
         \x20   void* d_up_weights, int up_R,\n\
         \x20   void* d_gate_weights, int gate_R,\n\
         \x20   void* d_down_weights, int down_R,\n\
         \x20   void* d_lm_head_norm_weights,\n\
         \x20   void* d_lm_head_weights, int lm_head_R,\n\
         \x20   void* d_k_cache, int kv_total_pages, int kv_page_size_runtime,\n\
         \x20   void* d_v_cache,\n\
         \x20   void* d_rope_cos, int max_pos,\n\
         \x20   void* d_rope_sin,\n\
         \x20   void* d_hidden_states, int batch_size_arg,\n\
         \x20   void* d_rms_rope_intermediates,\n\
         \x20   void* d_rms_gate_intermediates,\n\
         \x20   void* d_q_post_rope,\n\
         \x20   void* d_attn_out,\n\
         \x20   void* d_silu_out,\n\
         \x20   void* d_rms_lm_head_intermediates,\n\
         \x20   void* d_logits, int vocab_size_arg,\n\
         \x20   void* d_position_ids, int num_position_ids,\n\
         \x20   void* d_kv_append_indices,\n\
         \x20   void* d_prefill_qo_indptr, int num_prefill_qo,\n\
         \x20   void* d_prefill_kv_indptr,\n\
         \x20   void* d_prefill_kv_indices, int num_prefill_kv_indices,\n\
         \x20   void* d_prefill_kv_last_page_len,\n\
         \x20   void* d_decode_kv_indptr, int num_decode_seqs,\n\
         \x20   void* d_decode_kv_indices, int num_decode_kv_indices,\n\
         \x20   void* d_decode_kv_last_page_len,\n\
         \x20   float attn_scale, float rms_norm_eps,\n\
         \x20   int num_pages, int num_prefill_tokens,\n\
         \x20   int dev_idx,\n\
         \x20   void* raw_stream\n\
         ) {{\n\
         \x20   using kittens::bf16;\n\
         \x20   cudaStream_t stream = reinterpret_cast<cudaStream_t>(raw_stream);\n\
         \x20\n\
         \x20   llama_70b_globals g {{\n\
         \x20       {{ static_cast<uint*>(d_bar), (size_t)bar_d0, (size_t)bar_d1, (size_t)bar_d2, (size_t)bar_d3 }},\n\
         \x20       {{ static_cast<int*>(d_instructions), nullptr, nullptr, num_instructions, nullptr }},\n\
         \x20       {{ static_cast<int*>(d_timings), nullptr, nullptr, num_timing_rows, nullptr }},\n\
         \x20       {{ static_cast<int*>(d_global_instruction_index), nullptr, nullptr, nullptr, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_qkv_weights), nullptr, LLAMA_NUM_LAYERS, qkv_R, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_attn_norm_weights), nullptr, nullptr, LLAMA_NUM_LAYERS, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_o_weights), nullptr, LLAMA_NUM_LAYERS, o_R, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_mlp_norm_weights), nullptr, nullptr, LLAMA_NUM_LAYERS, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_up_weights), nullptr, LLAMA_NUM_LAYERS, up_R, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_gate_weights), nullptr, LLAMA_NUM_LAYERS, gate_R, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_down_weights), nullptr, LLAMA_NUM_LAYERS, down_R, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_lm_head_norm_weights), nullptr, nullptr, 1, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_lm_head_weights), nullptr, 1, lm_head_R, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_k_cache), kv_total_pages, kv_page_size_runtime, nullptr, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_v_cache), kv_total_pages, kv_page_size_runtime, nullptr, nullptr }},\n\
         \x20       {{ static_cast<float*>(d_rope_cos), nullptr, nullptr, max_pos, nullptr }},\n\
         \x20       {{ static_cast<float*>(d_rope_sin), nullptr, nullptr, max_pos, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_hidden_states), nullptr, nullptr, (size_t)batch_size_arg, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_rms_rope_intermediates), nullptr, nullptr, (size_t)batch_size_arg, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_rms_gate_intermediates), nullptr, nullptr, (size_t)batch_size_arg, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_q_post_rope), nullptr, nullptr, (size_t)batch_size_arg, (size_t)(LLAMA_NUM_ATTENTION_HEADS * LLAMA_HEAD_DIM) }},\n\
         \x20       {{ static_cast<bf16*>(d_attn_out), nullptr, nullptr, (size_t)batch_size_arg, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_silu_out), nullptr, nullptr, (size_t)batch_size_arg, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_rms_lm_head_intermediates), nullptr, nullptr, (size_t)batch_size_arg, nullptr }},\n\
         \x20       {{ static_cast<bf16*>(d_logits), nullptr, nullptr, batch_size_arg, vocab_size_arg }},\n\
         \x20       {{ static_cast<int*>(d_position_ids), nullptr, nullptr, nullptr, num_position_ids }},\n\
         \x20       {{ static_cast<int*>(d_kv_append_indices), nullptr, nullptr, nullptr, num_position_ids }},\n\
         \x20       {{ static_cast<int*>(d_prefill_qo_indptr), nullptr, nullptr, nullptr, num_prefill_qo }},\n\
         \x20       {{ static_cast<int*>(d_prefill_kv_indptr), nullptr, nullptr, nullptr, num_prefill_qo }},\n\
         \x20       {{ static_cast<int*>(d_prefill_kv_indices), nullptr, nullptr, nullptr, num_prefill_kv_indices }},\n\
         \x20       {{ static_cast<int*>(d_prefill_kv_last_page_len), nullptr, nullptr, nullptr, num_prefill_qo }},\n\
         \x20       {{ static_cast<int*>(d_decode_kv_indptr), nullptr, nullptr, nullptr, num_decode_seqs }},\n\
         \x20       {{ static_cast<int*>(d_decode_kv_indices), nullptr, nullptr, nullptr, num_decode_kv_indices }},\n\
         \x20       {{ static_cast<int*>(d_decode_kv_last_page_len), nullptr, nullptr, nullptr, num_decode_seqs }},\n\
         \x20       attn_scale, rms_norm_eps, num_pages, batch_size_arg, num_prefill_tokens,\n\
         \x20       dev_idx,\n\
         \x20   }};\n\
         \x20\n\
         \x20   auto kernel = &mk<\n\
         \x20       llama_config, llama_70b_globals,\n\
         \x20       ops::attn_norm_op, ops::qkv_rope_append_op,\n\
         \x20       ops::attention_decode_op, ops::attention_prefill_op,\n\
         \x20       ops::o_proj_op, ops::mlp_norm_op,\n\
         \x20       ops::gate_silu_op, ops::up_matmul_op,\n\
         \x20       ops::downproj_op,\n\
         \x20       ops::lm_head_norm_op, ops::lm_head_op,\n\
         \x20       ops::barrier_inc_op, ops::all_device_barrier_op>;\n\
         \x20\n\
         \x20   int dynamic_smem = (int)g.dynamic_shared_memory();\n\
         \x20   cudaError_t err = cudaFuncSetAttribute(\n\
         \x20       kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem);\n\
         \x20   if (err != cudaSuccess) return (int)err;\n\
         \x20\n\
         \x20   kernel<<<g.grid(), g.block(), dynamic_smem, stream>>>(g);\n\
         \x20   return (int)cudaGetLastError();\n\
         }}\n",
        canonical_name,
    ));
    out
}

/// Write the per-canonical `.cu` source to the cudaforge cache
/// (`~/.cache/cudaforge/megakernels/tk_megakernel_<canonical>.cu`).
/// `ferrite-cuda-builder/build.rs` picks every `.cu` in this dir
/// up and feeds them to NVCC alongside vendor sources.
pub fn write_cu_to_cache(canonical_name: &str, source: &str) -> std::io::Result<()> {
    use std::io::Write;
    let dir = megakernel_cache_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("tk_megakernel_{canonical_name}.cu"));
    let mut f = std::fs::File::create(&path)?;
    f.write_all(source.as_bytes())?;
    Ok(())
}

/// Resolve the megakernel cache directory using the SAME logic
/// `ferrite-cuda-builder/build.rs` uses (`dirs::cache_dir()`):
/// `$XDG_CACHE_HOME/cudaforge/megakernels` if set, else
/// `$HOME/.cache/cudaforge/megakernels`. The proc-macro can't
/// depend on the `dirs` crate (proc-macro deps are heavy), so
/// we replicate the precedence inline.
///
/// PRIOR BUG: this fn used `$HOME/.cache/...` unconditionally,
/// ignoring `XDG_CACHE_HOME`. Anyone with `XDG_CACHE_HOME` set
/// got a SILENT path mismatch — proc-macro wrote `.cu` files to
/// `$HOME/.cache/cudaforge/megakernels/`, build.rs scanned
/// `$XDG_CACHE_HOME/cudaforge/megakernels/` (empty), no `.cu`
/// reached NVCC, no `libmegakernels.a` was produced, every
/// `tk_megakernel_<canonical>_launch` symbol came out
/// undefined at link time. Took multiple sessions to spot.
pub fn megakernel_cache_dir() -> std::path::PathBuf {
    use std::path::PathBuf;
    let cache_root = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|| PathBuf::from("/tmp/.cache"))
        });
    cache_root.join("cudaforge").join("megakernels")
}

/// Per-canonical Rust marshaling wrapper. The host dispatcher
/// invokes this fn for kvm-eligible buckets; it borrows weight
/// accessors off `Weights`, builds the arg pack, and calls the
/// per-canonical extern launcher.
///
/// Weight accessors are extracted by MATCHING THE WEIGHT PATH
/// against known role names (`input_layernorm`, `self_attn_o_proj`,
/// `lm_head`, etc.) — NOT by walking the lowered IR by ordinal.
/// The eligibility check (`bucket_kvm_eligible`) guarantees each
/// role is present; absence is a programming error and panics.
#[allow(clippy::too_many_arguments)]
pub fn emit_wrapper_fn(
    fn_ident: &Ident,
    canonical_name: &str,
    tape_static_ident: &Ident,
    dims: &KvmDims,
    backbone_slot: u32,
    terminal_slot: u32,
    backbone_instances: &[OpInstance],
    lm_head_instances: &[OpInstance],
    shapes: &BTreeMap<String, OpcodeShape>,
) -> TokenStream {
    let extern_ident = Ident::new(
        &format!("tk_megakernel_{canonical_name}_launch"),
        Span::call_site(),
    );
    let buckets: Vec<&[OpInstance]> = vec![backbone_instances, lm_head_instances];

    // Each role is identified by (variant_name, weight-path
    // substring). The eligibility check guarantees every needed
    // role is present in the bucket; missing role → panic, not
    // Option-as-refusal.
    let attn_norm = find_weight_fn(&buckets, shapes, "RmsNorm", "input_layernorm");
    let mlp_norm = find_weight_fn(&buckets, shapes, "RmsNorm", "post_attention_layernorm");
    let lm_head_norm = find_weight_fn_first_match(
        &buckets,
        shapes,
        "RmsNorm",
        &["model_norm", "::norm"],
    );
    let qkv = find_weight_fn_any_variant(
        &buckets,
        shapes,
        &["FusedQkvRopeCache", "FusedQkvRopePrefill"],
        None,
        "weight_fn",
    );
    let cos_sin = find_weight_fn_any_variant(
        &buckets,
        shapes,
        &["FusedQkvRopeCache", "FusedQkvRopePrefill"],
        None,
        "cos_sin_fn",
    );
    let o = find_weight_fn(&buckets, shapes, "CutlassGemmAdd", "self_attn_o_proj");
    let down = find_weight_fn(&buckets, shapes, "CutlassGemmAdd", "mlp_down_proj");
    let gate_up = find_weight_fn_any_variant(
        &buckets,
        shapes,
        &["FusedGateUpSiluMul"],
        None,
        "weight_fn",
    );
    let lm_head = find_weight_fn_any_variant(
        &buckets,
        shapes,
        &["CutlassGemm", "Gemm", "CutlassGemv", "Cublas"],
        Some("lm_head"),
        "weight_fn",
    );

    let num_layers_lit = proc_macro2::Literal::i32_unsuffixed(dims.num_layers as i32);
    let hidden_dim_lit = proc_macro2::Literal::i32_unsuffixed(dims.hidden_dim as i32);
    let head_dim_lit = proc_macro2::Literal::i32_unsuffixed(dims.head_dim as i32);
    let num_attn_heads_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.num_attention_heads as i32);
    let num_kv_heads_lit = proc_macro2::Literal::i32_unsuffixed(dims.num_kv_heads as i32);
    let intermediate_dim_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.intermediate_dim as i32);
    let num_devices_lit = proc_macro2::Literal::i32_unsuffixed(dims.num_devices as i32);
    let vocab_size_lit = proc_macro2::Literal::i32_unsuffixed(dims.vocab_size as i32);
    let matmul_out_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.matmul_out_block_size as i32);
    let matmul_batch_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.matmul_batch_block_size as i32);
    let kv_page_size_lit = proc_macro2::Literal::i32_unsuffixed(dims.kv_page_size as i32);
    let attn_scale = 1.0f32 / (dims.head_dim as f32).sqrt();
    let attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(attn_scale);
    let backbone_slot_lit = proc_macro2::Literal::u32_unsuffixed(backbone_slot);
    let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(terminal_slot);

    quote! {
        #[cfg(feature = "cuda")]
        #[allow(non_snake_case, dead_code, clippy::too_many_arguments, unused_variables)]
        unsafe fn #fn_ident(
            wm: &Weights,
            ctx: &::ferrite_forward::ForwardCtx,
            device: &mut ::ferrite_cuda_core::device::GpuDevice,
            tiles: &mut ::std::vec::Vec<
                ::core::option::Option<::ferrite_forward::TileEntry>,
            >,
        ) {
            let qkv_w = (#qkv)(wm, 0).dense_weight();
            let attn_norm_w = (#attn_norm)(wm, 0).weight;
            let o_w = (#o)(wm, 0).dense_weight();
            let mlp_norm_w = (#mlp_norm)(wm, 0).weight;
            let gate_up_w = (#gate_up)(wm, 0).dense_weight();
            let down_w = (#down)(wm, 0).dense_weight();
            let lm_head_norm_w = (#lm_head_norm)(wm, 0).weight;
            let lm_head_w = (#lm_head)(wm, 0).dense_weight();
            let rope = (#cos_sin)(wm, 0);
            let max_pos_runtime = rope.shape()[0] as i32;
            let gate_up_r = (gate_up_w.shape()[0] / 2) as i32;
            let rms_norm_eps = (#attn_norm)(wm, 0).eps;

            let weights = ::ferrite_forward::interpreter::kvm::WeightPtrs {
                qkv: qkv_w.raw_ptr(),
                qkv_r: qkv_w.shape()[0] as i32,
                attn_norm: attn_norm_w.raw_ptr(),
                o: o_w.raw_ptr(),
                o_r: o_w.shape()[0] as i32,
                mlp_norm: mlp_norm_w.raw_ptr(),
                up: gate_up_w.raw_ptr(),
                up_r: gate_up_r,
                gate: gate_up_w.raw_ptr(),
                gate_r: gate_up_r,
                down: down_w.raw_ptr(),
                down_r: down_w.shape()[0] as i32,
                lm_head_norm: lm_head_norm_w.raw_ptr(),
                lm_head: lm_head_w.raw_ptr(),
                lm_head_r: lm_head_w.shape()[0] as i32,
            };
            let rope = ::ferrite_forward::interpreter::kvm::RopePtrs {
                cos: rope.raw_ptr(),
                sin: rope.raw_ptr(),
                max_pos: max_pos_runtime,
            };
            let shape = ::ferrite_forward::interpreter::kvm::ShapeConfig {
                num_layers: #num_layers_lit,
                hidden_dim: #hidden_dim_lit,
                head_dim: #head_dim_lit,
                num_attention_heads: #num_attn_heads_lit,
                num_kv_heads: #num_kv_heads_lit,
                intermediate_dim: #intermediate_dim_lit,
                num_devices: #num_devices_lit,
                vocab_size: #vocab_size_lit,
                matmul_out_block_size: #matmul_out_lit,
                matmul_batch_block_size: #matmul_batch_lit,
                kv_page_size: #kv_page_size_lit,
                attn_scale: #attn_scale_lit,
                rms_norm_eps,
            };

            let (hidden, logits) = unsafe {
                ::ferrite_forward::interpreter::kvm::launch(
                    #extern_ident as ::ferrite_forward::interpreter::kvm::ExternLaunch,
                    ctx,
                    device,
                    &#tape_static_ident,
                    weights,
                    rope,
                    shape,
                )
            }
            .expect("kvm megakernel launch failed");

            tiles[#backbone_slot_lit as usize] =
                ::core::option::Option::Some(::ferrite_forward::TileEntry::Owned(hidden));
            tiles[#terminal_slot_lit as usize] =
                ::core::option::Option::Some(::ferrite_forward::TileEntry::Owned(logits));
        }
    }
}

/// Look up the field-index of a named field on a variant's
/// OpcodeShape. Panics if the variant or field is missing —
/// either case means the OpcodeShape registry and the encoder
/// disagree on what fields exist, which is a programming error.
fn field_index_of(shapes: &BTreeMap<String, OpcodeShape>, variant: &str, field: &str) -> usize {
    let shape = shapes.get(variant).unwrap_or_else(|| {
        panic!("kvm wrapper: OpcodeShape for variant `{variant}` missing from registry")
    });
    shape
        .fields
        .iter()
        .position(|(name, _ty)| name.to_string() == field)
        .unwrap_or_else(|| {
            panic!("kvm wrapper: variant `{variant}` has no field named `{field}`")
        })
}

/// Find the `weight_fn` of a variant whose weight path contains
/// the given `path_substr` (e.g. `input_layernorm`). Search both
/// backbone + lm_head buckets in order. Panics if no match —
/// eligibility check should have ensured the role is present.
fn find_weight_fn(
    buckets: &[&[OpInstance]],
    shapes: &BTreeMap<String, OpcodeShape>,
    variant: &str,
    path_substr: &str,
) -> TokenStream {
    let idx = field_index_of(shapes, variant, "weight_fn");
    for bucket in buckets {
        for inst in *bucket {
            if inst.name != variant {
                continue;
            }
            let path = inst
                .field_values
                .get(idx)
                .map(|ts| ts.to_string().replace(' ', ""))
                .unwrap_or_default();
            if path.contains(path_substr) {
                return inst.field_values[idx].clone();
            }
        }
    }
    panic!(
        "kvm wrapper: no `{variant}` instance with weight path containing `{path_substr}` \
         found in backbone + lm_head buckets — eligibility check let through a bucket \
         that's missing this role"
    )
}

/// Same as `find_weight_fn` but tries multiple path substrings;
/// returns the first match. Used for the lm_head_norm role
/// where the weight name varies by arch (`model_norm`, `::norm`).
fn find_weight_fn_first_match(
    buckets: &[&[OpInstance]],
    shapes: &BTreeMap<String, OpcodeShape>,
    variant: &str,
    path_substrs: &[&str],
) -> TokenStream {
    let idx = field_index_of(shapes, variant, "weight_fn");
    for bucket in buckets {
        for inst in *bucket {
            if inst.name != variant {
                continue;
            }
            let path = inst
                .field_values
                .get(idx)
                .map(|ts| ts.to_string().replace(' ', ""))
                .unwrap_or_default();
            if path_substrs.iter().any(|s| path.contains(s)) {
                return inst.field_values[idx].clone();
            }
        }
    }
    panic!(
        "kvm wrapper: no `{variant}` instance matching any of {path_substrs:?} \
         found in backbone + lm_head buckets"
    )
}

/// Find a named field of an instance whose variant matches one
/// of `variants` and (optionally) whose `weight_fn` path
/// contains `path_substr`. Used for variants where the variant
/// alone is enough to identify the role (FusedQkvRopeCache —
/// only one per layer body) or where role is determined by
/// weight path (lm_head Gemm).
fn find_weight_fn_any_variant(
    buckets: &[&[OpInstance]],
    shapes: &BTreeMap<String, OpcodeShape>,
    variants: &[&str],
    path_substr: Option<&str>,
    field_name: &str,
) -> TokenStream {
    for bucket in buckets {
        for inst in *bucket {
            let name = inst.name.to_string();
            if !variants.contains(&name.as_str()) {
                continue;
            }
            let weight_idx = field_index_of(shapes, &name, "weight_fn");
            let path = inst
                .field_values
                .get(weight_idx)
                .map(|ts| ts.to_string().replace(' ', ""))
                .unwrap_or_default();
            if let Some(needle) = path_substr {
                if !path.contains(needle) {
                    continue;
                }
            }
            let field_idx = field_index_of(shapes, &name, field_name);
            return inst.field_values[field_idx].clone();
        }
    }
    panic!(
        "kvm wrapper: no instance of variants {variants:?} \
         (path_substr={path_substr:?}, field=`{field_name}`) \
         found in backbone + lm_head buckets"
    )
}
