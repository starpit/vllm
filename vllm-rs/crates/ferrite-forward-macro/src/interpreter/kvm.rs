// SPDX-License-Identifier: Apache-2.0
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
    pub kv_page_size: u32,
    pub prefill_kv_block_size: u32,
    pub decode_kv_block_size: u32,
    pub matmul_out_block_size: u32,
    pub matmul_batch_block_size: u32,
    pub num_devices: u32,
    pub sm_count: u32,
}

/// Compute KvmDims from a model's bounds, returning `None` if
/// any of vendor's static_assert checks fail (head_dim
/// divisibility, hidden / intermediate multiples of 256, etc.).
pub fn dims_from_bounds(bounds: &BTreeMap<String, u64>) -> Option<KvmDims> {
    let g = |k: &str| -> u64 { bounds.get(k).copied().unwrap_or(0) };
    let head_dim = g("head_dim") as u32;
    let num_attention_heads = g("num_attention_heads") as u32;
    let num_kv_heads = g("num_key_value_heads") as u32;
    let hidden_dim = g("hidden_size") as u32;
    let intermediate_dim = g("intermediate_size") as u32;
    let num_layers = g("num_hidden_layers") as u32;
    let num_devices = 1u32;
    let matmul_out_block_size = 2 * head_dim;
    if head_dim == 0
        || num_attention_heads == 0
        || hidden_dim == 0
        || intermediate_dim == 0
        || num_layers == 0
    {
        return None;
    }
    if !head_dim.is_multiple_of(32) {
        return None;
    }
    if num_kv_heads == 0 || !num_kv_heads.is_multiple_of(num_devices) {
        return None;
    }
    let num_generated_heads = matmul_out_block_size / head_dim;
    if num_generated_heads != 2 {
        return None;
    }
    let kv_col_start = num_attention_heads / num_generated_heads / num_devices;
    if !kv_col_start.is_multiple_of(2) {
        return None;
    }
    const PIPELINE_K_TIMES_STAGES: u32 = 64 * 4;
    if !hidden_dim.is_multiple_of(PIPELINE_K_TIMES_STAGES) {
        return None;
    }
    if !(intermediate_dim / num_devices).is_multiple_of(PIPELINE_K_TIMES_STAGES) {
        return None;
    }
    Some(KvmDims {
        num_layers,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_attention_heads,
        num_kv_heads,
        kv_page_size: 128,
        prefill_kv_block_size: 128,
        decode_kv_block_size: 16,
        matmul_out_block_size,
        matmul_batch_block_size: 128,
        num_devices,
        sm_count: 132,
    })
}

/// Encode one OpInstance into zero-or-more tape rows. Returns
/// `None` if the instance's variant has no megakernel mapping
/// (caller treats the bucket as kvm-ineligible). Variants
/// returning `Some(empty Vec)` are kvm-eligible but emit no
/// rows at codegen time — runtime tape-rebuild path (e.g.
/// AttentionPrefillContiguous needs per-call seq_info).
fn encode_op(
    inst: &OpInstance,
    dims: &KvmDims,
    batch_size: u32,
    layer: u32,
) -> Option<Vec<TapeRow>> {
    let bb = (batch_size / dims.matmul_batch_block_size).max(1) as i32;
    let ob = (dims.hidden_dim / dims.matmul_out_block_size).max(1) as i32;
    let ib =
        (dims.intermediate_dim / dims.matmul_out_block_size / dims.num_devices.max(1)).max(1) as i32;
    let qb = ((dims.num_attention_heads + 2 * dims.num_kv_heads) * dims.head_dim
        / dims.matmul_out_block_size
        / dims.num_devices.max(1))
    .max(1) as i32;

    match inst.name.to_string().as_str() {
        // FusedAddRmsNorm encodes the same rows as a plain RmsNorm
        // — the "add" half is structural in the megakernel: the
        // previous op (o_proj or downproj) is a `CutlassGemmAdd`
        // which fuses the residual add into its accumulator, so by
        // the time the norm runs the residual stream is already
        // resident in hidden_states. The fused-add-rms-norm host
        // Impl claims [add, norm] tiles together; in the TK
        // universe the same two-tile claim stands but the add tile
        // produces no kernel work.
        "RmsNorm" | "FusedAddRmsNorm" => {
            // Per-batch-position fanout. Vendor's
            // `batched_rms_norm` dispatches one warp per row.
            // Role (attn / mlp / lm_head) is positional in the
            // tape — not encoded in the row itself; the kernel
            // uses the order to flow data through the residual
            // stream.
            let mut out = Vec::with_capacity(batch_size as usize);
            for bidx in 0..batch_size as i32 {
                let mut row = TapeRow::new(OP_ATTN_NORM);
                row.payload[1] = layer as i32;
                row.payload[2] = 1;
                row.payload[3] = bidx;
                out.push(row);
            }
            Some(out)
        }
        "CutlassGemmAdd" => {
            // o_proj or downproj fanout. Same row layout for
            // both — vendor uses a separate opcode for each
            // role; here we use OP_O_PROJ_RESIDUAL as the
            // canonical position and the kernel flips opcode
            // based on layer position.
            let mut out = Vec::with_capacity((bb * ob) as usize);
            for batch_block in 0..bb {
                for out_block in 0..ob {
                    let mut row = TapeRow::new(OP_O_PROJ_RESIDUAL);
                    row.payload[1] = layer as i32;
                    row.payload[2] = batch_block;
                    row.payload[3] = out_block;
                    row.payload[4] = batch_block;
                    row.payload[5] = out_block;
                    out.push(row);
                }
            }
            Some(out)
        }
        "FusedQkvRopeCache" | "FusedQkvRopePrefill" => {
            // Fused qkv + rope_append. Decode vs prefill is a
            // runtime branch in the kernel via
            // `g.num_prefill_tokens`; same row layout for both.
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
            Some(out)
        }
        "FusedGateUpSiluMul" => {
            // Two opcodes per batch×intermediate block —
            // vendor's `gate_silu` then `up_matmul`.
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
            Some(out)
        }
        "AttentionViaCache" | "FlashInferAttentionDecode" => {
            // Decode attention. Pack (seq, kv_head) pairs
            // into chunks; vendor's scheduler uses 8 per row.
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
            Some(out)
        }
        "AttentionPrefillContiguous" | "FlashInferAttentionPrefill" => {
            // Prefill rows depend on per-call seq_info — defer
            // to runtime. Encoder emits an empty Vec; the
            // launcher rebuilds rows from `ctx.cu_seqlens_q`
            // before issuing the megakernel.
            Some(Vec::new())
        }
        "CutlassGemm" | "Gemm" | "CutlassGemv" => {
            // Final lm_head Gemm — vocab-block fanout. Vocab
            // size isn't on `dims` (it's per-canonical); the
            // launcher reads it from g.vocab_size_arg at run
            // time. Here we emit a placeholder row count and
            // patch the per-row col field at run time too.
            let v = 256i32;
            let mut out = Vec::with_capacity((bb * v) as usize);
            for batch_block in 0..bb {
                for vocab_block in 0..v {
                    let mut row = TapeRow::new(OP_LM_HEAD);
                    row.payload[1] = 0;
                    row.payload[2] = batch_block;
                    row.payload[3] = vocab_block;
                    row.payload[4] = batch_block;
                    row.payload[5] = vocab_block;
                    out.push(row);
                }
            }
            Some(out)
        }
        "CutlassFusedAddRmsNormGemm" => {
            // Host-side cutlass impl that fuses [add, rms_norm,
            // gemm] into one kernel. Lives at the lm_head boundary:
            // (final_layer_residual_add) → lm_head_norm → lm_head.
            // In the megakernel the "add" half is structural (the
            // last layer's `tk_cutlass_32x64_s4_add` downproj
            // already wrote residual+output into hidden_states), so
            // this single OpInstance encodes to TWO row groups:
            //   1. OP_LM_HEAD_NORM rows — one per batch position,
            //      same shape as a plain RmsNorm.
            //   2. OP_LM_HEAD rows — vocab-block fanout, same shape
            //      as a plain Gemm at lm_head position.
            let mut out = Vec::with_capacity(batch_size as usize);
            for bidx in 0..batch_size as i32 {
                let mut row = TapeRow::new(OP_LM_HEAD_NORM);
                row.payload[1] = 0;
                row.payload[2] = 1;
                row.payload[3] = bidx;
                out.push(row);
            }
            let v = 256i32;
            for batch_block in 0..bb {
                for vocab_block in 0..v {
                    let mut row = TapeRow::new(OP_LM_HEAD);
                    row.payload[1] = 0;
                    row.payload[2] = batch_block;
                    row.payload[3] = vocab_block;
                    row.payload[4] = batch_block;
                    row.payload[5] = vocab_block;
                    out.push(row);
                }
            }
            Some(out)
        }
        "Embed" | "Reshape" | "Free" | "Alias" => Some(Vec::new()),
        "Loop" => {
            // The caller in `encode_bucket` handles Loop
            // unrolling — reaching this arm is a bug.
            None
        }
        // Variants the megakernel can't run. Caller falls back
        // to host interpreter for any bucket containing one.
        _ => None,
    }
}

/// Encode a `LoweredBucket` into a tape. Walks the bucket's
/// `instances`, unrolling `Loop` bodies into per-iteration row
/// emission (`layer` field bumped each iteration), returning
/// `None` if any op is kvm-ineligible.
pub fn encode_bucket(
    bucket: &LoweredBucket,
    dims: &KvmDims,
    batch_size: u32,
) -> Option<Vec<TapeRow>> {
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
                .unwrap_or(0);
            let body_start = i + 1;
            let body_end = body_start + body_len;
            if body_end > instances.len() {
                return None;
            }
            for layer_iter in 0..dims.num_layers {
                for body_inst in &instances[body_start..body_end] {
                    let rows = encode_op(body_inst, dims, batch_size, layer_iter)?;
                    out.extend(rows);
                }
            }
            i = body_end;
        } else {
            let rows = encode_op(inst, dims, batch_size, 0)?;
            out.extend(rows);
            i += 1;
        }
    }
    Some(out)
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
    use std::path::PathBuf;
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let dir = home.join(".cache").join("cudaforge").join("megakernels");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("tk_megakernel_{canonical_name}.cu"));
    let mut f = std::fs::File::create(&path)?;
    f.write_all(source.as_bytes())?;
    Ok(())
}

/// Per-canonical Rust marshaling wrapper. The host dispatcher
/// invokes this fn for kvm-eligible buckets; it H2Ds the tape,
/// builds the arg pack, and calls the per-canonical extern
/// launcher. Body delegates to the runtime helper
/// `ferrite_forward::interpreter::kvm::launch` which factors
/// out scratch alloc / D2H+H2D / the extern call itself.
///
/// Returns `None` when the lowered IR doesn't carry the
/// accessors the wrapper needs. The dispatcher leaves that
/// canonical's slot empty and the bucket falls through to host.
#[allow(clippy::too_many_arguments)]
pub fn emit_wrapper_fn(
    fn_ident: &Ident,
    canonical_name: &str,
    tape_static_ident: &Ident,
    dims: &KvmDims,
    vocab_size: u32,
    backbone_slot: u32,
    terminal_slot: u32,
    backbone_instances: &[OpInstance],
    lm_head_instances: &[OpInstance],
    shapes: &BTreeMap<String, OpcodeShape>,
) -> Option<TokenStream> {
    let extern_ident = Ident::new(
        &format!("tk_megakernel_{canonical_name}_launch"),
        Span::call_site(),
    );

    // Walk the lowered IR for accessor paths. Field index for
    // weight_fn / cos_sin_fn varies by variant — look it up in
    // the OpcodeShape registry by field NAME, not by position.
    // FusedQkvRopePrefill, for instance, has 5 u32s before
    // weight_fn (in_slot, out_slot, layer, num_q_heads,
    // num_kv_heads), so weight_fn lives at index 5, not 3.
    let attn_norm = nth_field(backbone_instances, shapes, "RmsNorm", "weight_fn", 0)?;
    let mlp_norm = nth_field(backbone_instances, shapes, "RmsNorm", "weight_fn", 1)?;
    let qkv = first_field(
        backbone_instances,
        shapes,
        &["FusedQkvRopeCache", "FusedQkvRopePrefill"],
        "weight_fn",
    )?;
    let cos_sin = first_field(
        backbone_instances,
        shapes,
        &["FusedQkvRopeCache", "FusedQkvRopePrefill"],
        "cos_sin_fn",
    )?;
    let o = nth_field(backbone_instances, shapes, "CutlassGemmAdd", "weight_fn", 0)?;
    let down = nth_field(backbone_instances, shapes, "CutlassGemmAdd", "weight_fn", 1)?;
    let gate_up = first_field(backbone_instances, shapes, &["FusedGateUpSiluMul"], "weight_fn")?;
    let lm_head = first_field(
        lm_head_instances,
        shapes,
        &["CutlassGemm", "Gemm", "CutlassGemv"],
        "weight_fn",
    )?;
    let lm_head_norm = first_field(lm_head_instances, shapes, &["RmsNorm"], "weight_fn")
        .or_else(|| nth_field(backbone_instances, shapes, "RmsNorm", "weight_fn", 2))?;

    let num_layers_lit = proc_macro2::Literal::i32_unsuffixed(dims.num_layers as i32);
    let hidden_dim_lit = proc_macro2::Literal::i32_unsuffixed(dims.hidden_dim as i32);
    let head_dim_lit = proc_macro2::Literal::i32_unsuffixed(dims.head_dim as i32);
    let num_attn_heads_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.num_attention_heads as i32);
    let num_kv_heads_lit = proc_macro2::Literal::i32_unsuffixed(dims.num_kv_heads as i32);
    let intermediate_dim_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.intermediate_dim as i32);
    let num_devices_lit = proc_macro2::Literal::i32_unsuffixed(dims.num_devices as i32);
    let vocab_size_lit = proc_macro2::Literal::i32_unsuffixed(vocab_size as i32);
    let matmul_out_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.matmul_out_block_size as i32);
    let matmul_batch_lit =
        proc_macro2::Literal::i32_unsuffixed(dims.matmul_batch_block_size as i32);
    let kv_page_size_lit = proc_macro2::Literal::i32_unsuffixed(dims.kv_page_size as i32);
    let attn_scale = 1.0f32 / (dims.head_dim as f32).sqrt();
    let attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(attn_scale);
    let backbone_slot_lit = proc_macro2::Literal::u32_unsuffixed(backbone_slot);
    let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(terminal_slot);

    Some(quote! {
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
    })
}

fn field_index_of(shapes: &BTreeMap<String, OpcodeShape>, variant: &str, field: &str) -> Option<usize> {
    let shape = shapes.get(variant)?;
    shape
        .fields
        .iter()
        .position(|(name, _ty)| name.to_string() == field)
}

fn first_field(
    instances: &[OpInstance],
    shapes: &BTreeMap<String, OpcodeShape>,
    names: &[&str],
    field_name: &str,
) -> Option<TokenStream> {
    for inst in instances {
        let n = inst.name.to_string();
        if names.iter().any(|x| *x == n.as_str()) {
            let idx = field_index_of(shapes, &n, field_name)?;
            return inst.field_values.get(idx).cloned();
        }
    }
    None
}

fn nth_field(
    instances: &[OpInstance],
    shapes: &BTreeMap<String, OpcodeShape>,
    name: &str,
    field_name: &str,
    n: usize,
) -> Option<TokenStream> {
    let idx = field_index_of(shapes, name, field_name)?;
    let mut k = 0;
    for inst in instances {
        if inst.name == name {
            if k == n {
                return inst.field_values.get(idx).cloned();
            }
            k += 1;
        }
    }
    None
}
