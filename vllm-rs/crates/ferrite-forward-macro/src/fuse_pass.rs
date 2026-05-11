// SPDX-License-Identifier: Apache-2.0
//
// Compiler-driven megakernel synthesis pass (backend-neutral).
//
// This module sits between `solver` and `codegen` in the macro
// pipeline. It walks the per-bucket SFUF claim assignments, identifies
// fuseable atom subgraphs (subject to dispatch-shape compatibility,
// register/TG-mem-only data flow, atomicity walls), and emits one
// synthesized kernel per group by stitching the atom emit fragments.
//
// MVP scope: a `synthesize_pre_attn_chunk` function that takes the
// already-collected (AddRmsNormAtom, AffineQmvAtom, RopeAppendAtom)
// triple and produces the Metal source for the corresponding fused
// decoder kernel. Once that produces a valid `.metal` source that
// compiles with `xcrun metal`, the rest is plumbing:
//   1. Wire `Implementation::as_atom` so the solver's claim assignments
//      can be mapped to atoms.
//   2. Generalize subgraph identification (don't hardcode the
//      pre-attn-chunk atom sequence).
//   3. Hand the generated source off to the macro for runtime
//      `newLibraryWithSource` loading + `SynthesizedKernel`
//      Instruction emission.

#![allow(dead_code)] // MVP demo; full pipeline wiring in next phase.

use crate::atom::{Atom, AtomConstantValue, AtomCtx};
use crate::atom_lib::{AddRmsNormAtom, AffineQmvAtom, RopeAppendAtom};

/// Backend-neutral synthesis target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthesisBackend {
    Metal,
    Cuda,
}

/// One synthesized kernel ready to hand to a backend's loader. The
/// `source` is shader-language source in the dialect implied by
/// `backend` (MSL for Metal, CUDA / C++ for CUDA).
#[derive(Clone, Debug)]
pub struct SynthesizedKernel {
    pub symbol: String,
    pub source: String,
    pub backend: SynthesisBackend,
}

/// Shape constants the synthesized kernel reads as function constants.
/// Names match the indices used in the body fragments.
#[derive(Clone, Debug)]
pub struct ChunkConstants {
    pub hidden:        u32,
    pub num_q_heads:   u32,
    pub num_kv_heads:  u32,
    pub head_dim:      u32,
    pub rot_dim:       u32,
    pub block_size:    u32,
    pub m:             u32,
    pub group_size:    u32,
    pub rms_norm_eps:  f32,
}

/// Synthesize the pre-attention decoder chunk:
/// `AddRmsNorm → AffineQmv (Q+K+V) → RopeAppend + KvPagedWrite`.
///
/// Output is a complete Metal source file that compiles with
/// `xcrun metal`. The fuse pass owns:
/// - channel-name allocation (kernel-scope variable names threaded
///   from each producer atom's output to the consumer's input);
/// - kernel signature derivation from the group's boundary I/O;
/// - function-constant declarations;
/// - threadgroup-memory scratch allocations;
/// - per-atom body emission with channel-name substitution.
pub fn synthesize_pre_attn_chunk(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
) -> SynthesizedKernel {
    assert_eq!(
        backend,
        SynthesisBackend::Metal,
        "MVP only emits Metal; CUDA emission lands in a later phase",
    );

    // Concrete channel-name allocation. The fuse pass threads the
    // producer atom's output channel name → consumer atom's input
    // channel name as kernel-scope variable names.
    //
    // - "x_norm"   : TG-mem activation produced by AddRmsNorm, consumed
    //                by AffineQmv.
    // - "qmv_smem" : TG-mem dot-product results produced by AffineQmv,
    //                consumed by RopeAppend.
    let x_norm_name   = "__x_norm".to_string();
    let qmv_smem_name = "__qmv_smem".to_string();
    let residual_io   = "__residual_io".to_string();
    let delta_buf     = "__delta".to_string();
    let rms_wt_buf    = "__rms_weight".to_string();
    let qkv_wt_buf    = "__qkv_weight".to_string();
    let qkv_sc_buf    = "__qkv_scales".to_string();
    let qkv_bi_buf    = "__qkv_biases".to_string();
    let cos_sin_buf   = "__cos_sin".to_string();
    let positions_buf = "__positions".to_string();
    let slot_map_buf  = "__slot_mapping".to_string();
    let q_out_buf     = "__q_out".to_string();
    let kv_cache_k    = "__kv_cache_k".to_string();
    let kv_cache_v    = "__kv_cache_v".to_string();

    // Build the per-atom AtomCtx (constants slice is shared across atoms
    // for the MVP; production fuse pass will tighten this).
    let constants_slice: Vec<(&'static str, AtomConstantValue)> = vec![
        ("HIDDEN",      AtomConstantValue::Uint(consts.hidden)),
        ("NUM_Q",       AtomConstantValue::Uint(consts.num_q_heads)),
        ("NUM_KV",      AtomConstantValue::Uint(consts.num_kv_heads)),
        ("HEAD_DIM",    AtomConstantValue::Uint(consts.head_dim)),
        ("ROT_DIM",     AtomConstantValue::Uint(consts.rot_dim)),
        ("BLOCK_SIZE",  AtomConstantValue::Uint(consts.block_size)),
        ("M",           AtomConstantValue::Uint(consts.m)),
        ("EPS",         AtomConstantValue::Float(consts.rms_norm_eps)),
    ];

    let addrms = AddRmsNormAtom;
    let qmv    = AffineQmvAtom { group_size: consts.group_size };
    let rope   = RopeAppendAtom;

    // Bind channel names for each atom.
    let addrms_in  = vec![residual_io.clone(), delta_buf.clone(), rms_wt_buf.clone()];
    let addrms_out = vec![x_norm_name.clone()];
    let addrms_ctx = AtomCtx {
        bound_inputs: &addrms_in,
        bound_outputs: &addrms_out,
        constants: &constants_slice,
        t_act,
        t_scale,
    };

    let qmv_in  = vec![
        x_norm_name.clone(),
        qkv_wt_buf.clone(),
        qkv_sc_buf.clone(),
        qkv_bi_buf.clone(),
    ];
    let qmv_out = vec![qmv_smem_name.clone()];
    let qmv_ctx = AtomCtx {
        bound_inputs: &qmv_in,
        bound_outputs: &qmv_out,
        constants: &constants_slice,
        t_act,
        t_scale,
    };

    let rope_in  = vec![
        qmv_smem_name.clone(),
        cos_sin_buf.clone(),
        positions_buf.clone(),
        slot_map_buf.clone(),
    ];
    let rope_out = vec![q_out_buf.clone(), kv_cache_k.clone(), kv_cache_v.clone()];
    let rope_ctx = AtomCtx {
        bound_inputs: &rope_in,
        bound_outputs: &rope_out,
        constants: &constants_slice,
        t_act,
        t_scale,
    };

    let addrms_body = addrms.emit_metal_body(&addrms_ctx)
        .expect("AddRmsNormAtom Metal emit");
    let qmv_body    = qmv.emit_metal_body(&qmv_ctx)
        .expect("AffineQmvAtom Metal emit");
    let rope_body   = rope.emit_metal_body(&rope_ctx)
        .expect("RopeAppendAtom Metal emit");

    // Symbol name: deterministic hash of the atom sequence + dtype + gs.
    // For the MVP we just use a readable name; production version uses
    // a structural hash.
    let symbol = format!(
        "synth_pre_attn_{}_{}_gs{}",
        t_act,
        t_scale,
        consts.group_size,
    );

    // Assemble the full Metal source.
    let source = format!(
        r#"// SPDX-License-Identifier: Apache-2.0
//
// SYNTHESIZED KERNEL — generated by ferrite-forward-macro::fuse_pass.
// Do not hand-edit. Source-of-truth is the atom sequence in
// `fuse_pass::synthesize_pre_attn_chunk` + the atoms in `atom_lib.rs`.

#include <metal_stdlib>
#include "metal_kittens.h"
using namespace metal;

constant uint  HIDDEN       [[function_constant(0)]];
constant uint  NUM_Q        [[function_constant(1)]];
constant uint  NUM_KV       [[function_constant(2)]];
constant uint  HEAD_DIM     [[function_constant(3)]];
constant uint  ROT_DIM      [[function_constant(4)]];
constant uint  BLOCK_SIZE   [[function_constant(5)]];
constant uint  M            [[function_constant(6)]];
constant float EPS          [[function_constant(7)]];

constant constexpr uint __HIDDEN_MAX  = 8192;
constant constexpr uint __HEAD_DIM_MAX = 256;
constant constexpr uint __SCRATCH_MAX  = __HEAD_DIM_MAX / MK_ROWS_PER_SIMDGROUP;

[[kernel]] void {symbol}(
    device       {t_act}*   {q_out_buf}      [[buffer(0)]],
    device       {t_act}*   {residual_io}    [[buffer(1)]],
    device const {t_act}*   {delta_buf}      [[buffer(2)]],
    device const {t_scale}* {rms_wt_buf}     [[buffer(3)]],
    device const uint32_t*  {qkv_wt_buf}     [[buffer(4)]],
    device const {t_scale}* {qkv_sc_buf}     [[buffer(5)]],
    device const {t_scale}* {qkv_bi_buf}     [[buffer(6)]],
    device const {t_act}*   {cos_sin_buf}    [[buffer(7)]],
    device const uint*      {positions_buf}  [[buffer(8)]],
    device const uint*      {slot_map_buf}   [[buffer(9)]],
    device       {t_act}*   {kv_cache_k}     [[buffer(10)]],
    device       {t_act}*   {kv_cache_v}     [[buffer(11)]],
    uint3 __tg_pos    [[threadgroup_position_in_grid]],
    uint3 __tid_pos   [[thread_position_in_threadgroup]],
    uint  __simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  __simd_lid  [[thread_index_in_simdgroup]])
{{
    const uint __t                = __tg_pos.x;
    const uint __head             = __tg_pos.y;
    const uint __tid              = __tid_pos.x;
    const uint __head_dim         = HEAD_DIM;
    const uint __hidden           = HIDDEN;
    const uint __num_q            = NUM_Q;
    const uint __num_kv           = NUM_KV;
    const uint __rot_dim          = ROT_DIM;
    const uint __half_dim         = __rot_dim / 2;
    const uint __block_sz         = BLOCK_SIZE;
    const uint __num_heads_total  = __num_q + 2u * __num_kv;
    const uint __threads_per_tg   = MK_SIMD_SIZE * __head_dim / MK_ROWS_PER_SIMDGROUP;
    const uint __num_simdgroups   = __head_dim / MK_ROWS_PER_SIMDGROUP;
    const float __eps             = EPS;

    if (__t >= M || __head >= __num_heads_total) return;

    threadgroup {t_act} {x_norm_name}[__HIDDEN_MAX];
    threadgroup float   __scratch  [__SCRATCH_MAX];
    threadgroup float   {qmv_smem_name}[__HEAD_DIM_MAX];

    {addrms_body}

    {qmv_body}

    {rope_body}
}}
"#,
        symbol = symbol,
        t_act = t_act,
        t_scale = t_scale,
        q_out_buf = q_out_buf,
        residual_io = residual_io,
        delta_buf = delta_buf,
        rms_wt_buf = rms_wt_buf,
        qkv_wt_buf = qkv_wt_buf,
        qkv_sc_buf = qkv_sc_buf,
        qkv_bi_buf = qkv_bi_buf,
        cos_sin_buf = cos_sin_buf,
        positions_buf = positions_buf,
        slot_map_buf = slot_map_buf,
        kv_cache_k = kv_cache_k,
        kv_cache_v = kv_cache_v,
        x_norm_name = x_norm_name,
        qmv_smem_name = qmv_smem_name,
        addrms_body = addrms_body,
        qmv_body = qmv_body,
        rope_body = rope_body,
    );

    SynthesizedKernel {
        symbol,
        source,
        backend: SynthesisBackend::Metal,
    }
}

/// Public helper for the synth-kernel dump probe (`bin/dump_synth.rs`).
/// Returns the Metal source string for Llama-3.2-3B-4bit shape so we
/// can run `xcrun metal` on it without standing up the full macro
/// pipeline.
pub fn dump_llama_3_2_3b_4bit_pre_attn() -> SynthesizedKernel {
    let consts = ChunkConstants {
        hidden:        3072,
        num_q_heads:   24,
        num_kv_heads:  8,
        head_dim:      128,
        rot_dim:       128,
        block_size:    16,
        m:             1,
        group_size:    64,
        rms_norm_eps:  1e-5,
    };
    synthesize_pre_attn_chunk(SynthesisBackend::Metal, "bfloat", "half", &consts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_3_2_3b_constants() -> ChunkConstants {
        // Llama-3.2-3B-Instruct-4bit: hidden=3072, q_heads=24, kv_heads=8,
        // head_dim=128, rot_dim=128, group_size=64, block_size=16.
        ChunkConstants {
            hidden:        3072,
            num_q_heads:   24,
            num_kv_heads:  8,
            head_dim:      128,
            rot_dim:       128,
            block_size:    16,
            m:             1,
            group_size:    64,
            rms_norm_eps:  1e-5,
        }
    }

    #[test]
    fn synthesize_pre_attn_chunk_emits_non_empty_metal_source() {
        let kernel = synthesize_pre_attn_chunk(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &llama_3_2_3b_constants(),
        );
        assert!(kernel.source.contains("[[kernel]]"));
        assert!(kernel.source.contains("mk_tg_rmsnorm_scale"));
        assert!(kernel.source.contains("mk_load_vector"));
        assert!(kernel.source.contains("mk_qdot"));
        assert!(kernel.source.contains("mk_rope_pair"));
        assert!(kernel.source.contains(&kernel.symbol));
        assert_eq!(kernel.backend, SynthesisBackend::Metal);
    }
}
