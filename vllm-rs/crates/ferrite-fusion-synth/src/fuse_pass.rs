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
use crate::atom_lib::{AddRmsNormAtom, AffineQmvAtom, RopeAppendAtom, SiluMulAtom};

/// Threads per Apple GPU simdgroup. Defined here in Rust (not just MSL)
/// for synth-time arithmetic on threads_per_tg / simdgroup counts.
/// Matches `MK_SIMD_SIZE` in `metal_kittens.h`.
const MK_SIMD_SIZE: u32 = 32;
/// HEAD_DIM rows owned per simdgroup per atom-loop pass. Matches
/// `MK_ROWS_PER_SIMDGROUP` in `metal_kittens.h`.
const MK_ROWS_PER_SIMDGROUP: u32 = 8;

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
/// Render an f32 as a valid MSL float literal. The caller appends the
/// `f` suffix. Uses Rust's `Debug` formatting (`{:?}`) which preserves
/// enough digits to round-trip and writes small magnitudes in
/// scientific form (e.g. `1e-5`) — both forms MSL accepts.
fn format_msl_float(v: f32) -> String {
    // Debug format on f32 already emits a parseable literal across
    // the full range; bare numerals don't need the `f` suffix to be
    // double-promoted inside an `EPS = <lit>f;` initializer.
    format!("{:?}", v)
}

/// `metal_kittens.h` carries `#pragma once` so it's safe to `#include`
/// from kernel sources. When we inline it into a synthesized main-TU
/// `.metal` file, `xcrun metal` warns `#pragma once in main file
/// [-Wpragma-once-outside-header]`. Strip the pragma at inline time —
/// each synthesized source is its own TU and inlines the header once.
fn inline_header(header: &str) -> String {
    header
        .lines()
        .filter(|l| l.trim() != "#pragma once")
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Clone, Debug)]
pub struct ChunkConstants {
    pub hidden:        u32,
    pub num_q_heads:   u32,
    pub num_kv_heads:  u32,
    pub head_dim:      u32,
    pub rot_dim:       u32,
    pub block_size:    u32,
    /// MLP intermediate size (TP-divided). Used only by
    /// `synthesize_mlp_pre_down_chunk` for the `INTERMEDIATE`
    /// `constant constexpr` bake. Set to 0 for pre-attn chunks.
    pub intermediate:  u32,
    pub m:             u32,
    pub group_size:    u32,
    pub rms_norm_eps:  f32,
    /// When `true`, `synthesize_pre_attn_chunk` emits a `_bias` symbol
    /// variant: the kernel signature gains three extra device-pointer
    /// bindings (`__q_linear_bias`, `__k_linear_bias`,
    /// `__v_linear_bias`) and each `AffineQmvAtom` is instantiated
    /// with `has_linear_bias: true`. Set by the codegen call site to
    /// `BackendCaps::has_bias_add` so a model whose DSL emits
    /// `bias_add(...)` on the QKV projections (Qwen2/Qwen2.5) gets the
    /// biased synth megakernel; bias-free models (Llama) keep the
    /// existing two-suffix-free symbol. Unused by the MLP / gate-up
    /// synthesizers — Qwen2's MLP has no bias.
    pub has_linear_bias: bool,
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
    synthesize_pre_attn_chunk_impl(backend, t_act, t_scale, consts, false)
}

/// Layer-0 variant: emits the same kernel as `synthesize_pre_attn_chunk`
/// but without the `residual += delta` step and without the per-Q-head
/// residual writeback. Symbol name is `synth_pre_attn_init_<t_act>_<t_scale>_gs<gs>`.
/// The lowering arm binds the input slot to BOTH the `residual_io` and
/// `delta` bindings — the kernel never reads `delta` in init mode.
pub fn synthesize_pre_attn_init_chunk(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
) -> SynthesizedKernel {
    synthesize_pre_attn_chunk_impl(backend, t_act, t_scale, consts, true)
}

fn synthesize_pre_attn_chunk_impl(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
    init: bool,
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
    // Per-projection weight triples — the chain emits ONE
    // SynthPreAttn that binds three separate `LinearLayer::AffineQuant`s
    // (Q, K, V). The kernel branches on head index to pick the right
    // triple for its qmv. Avoids the load-time packed-concat
    // infrastructure that would be needed for a single-buffer design.
    let q_wt_buf      = "__q_weight".to_string();
    let q_sc_buf      = "__q_scales".to_string();
    let q_bi_buf      = "__q_biases".to_string();
    let k_wt_buf      = "__k_weight".to_string();
    let k_sc_buf      = "__k_scales".to_string();
    let k_bi_buf      = "__k_biases".to_string();
    let v_wt_buf      = "__v_weight".to_string();
    let v_sc_buf      = "__v_scales".to_string();
    let v_bi_buf      = "__v_biases".to_string();
    // Per-row linear biases (Qwen2 QKV). Bound at buffers 18/19/20
    // only when `consts.has_linear_bias == true`; otherwise the
    // signature stops at buffer 17 (kv_cache_v) like the bias-free
    // variant.
    let q_lb_buf      = "__q_linear_bias".to_string();
    let k_lb_buf      = "__k_linear_bias".to_string();
    let v_lb_buf      = "__v_linear_bias".to_string();
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

    let addrms = AddRmsNormAtom { init };
    let rope   = RopeAppendAtom;
    // One AffineQmvAtom per QKV band — the only thing that varies
    // between bands is the local-head expression (each band's weight
    // buffer is row 0..) and the W/S/B channel triple bound below.
    let qmv_q = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "__head",
        has_linear_bias: consts.has_linear_bias,
    };
    let qmv_k = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "(__head - __num_q)",
        has_linear_bias: consts.has_linear_bias,
    };
    let qmv_v = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "(__head - __num_q - __num_kv)",
        has_linear_bias: consts.has_linear_bias,
    };

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

    // QKV qmv is now atom-driven: three AffineQmvAtom instances, one
    // per band, each with its own local-head expression and W/S/B
    // channel triple. The kernel still branches on head index because
    // the three weight buffers stay separate (avoids load-time concat).
    let _ = qmv_smem_name;

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
    let rope_body   = rope.emit_metal_body(&rope_ctx)
        .expect("RopeAppendAtom Metal emit");

    // Per-band QKV qmv: three AffineQmvAtom calls, each with its own
    // local-head expression and W/S/B channel triple. All three emit
    // into the shared `qmv_smem` so RopeAppend doesn't care which
    // band ran.
    let emit_band_body =
        |atom: &AffineQmvAtom, w: &str, s: &str, b: &str, lb: Option<&str>| -> String {
            let mut band_in = vec![
                x_norm_name.clone(),
                w.to_string(),
                s.to_string(),
                b.to_string(),
            ];
            if let Some(lb_name) = lb {
                band_in.push(lb_name.to_string());
            }
            let band_out = vec![qmv_smem_name.clone()];
            let band_ctx = AtomCtx {
                bound_inputs: &band_in,
                bound_outputs: &band_out,
                constants: &constants_slice,
                t_act,
                t_scale,
            };
            atom.emit_metal_body(&band_ctx)
                .expect("AffineQmvAtom Metal emit")
        };
    let q_lb_opt: Option<&str> = if consts.has_linear_bias {
        Some(q_lb_buf.as_str())
    } else {
        None
    };
    let k_lb_opt: Option<&str> = if consts.has_linear_bias {
        Some(k_lb_buf.as_str())
    } else {
        None
    };
    let v_lb_opt: Option<&str> = if consts.has_linear_bias {
        Some(v_lb_buf.as_str())
    } else {
        None
    };
    let qmv_q_body = emit_band_body(&qmv_q, &q_wt_buf, &q_sc_buf, &q_bi_buf, q_lb_opt);
    let qmv_k_body = emit_band_body(&qmv_k, &k_wt_buf, &k_sc_buf, &k_bi_buf, k_lb_opt);
    let qmv_v_body = emit_band_body(&qmv_v, &v_wt_buf, &v_sc_buf, &v_bi_buf, v_lb_opt);
    let qmv_body = format!(
        r#"
    // --- per-band QKV qmv (3× AffineQmvAtom, band-selected by __head) ---
    if (__head < __num_q) {{
        {qmv_q_body}
    }} else if (__head < __num_q + __num_kv) {{
        {qmv_k_body}
    }} else {{
        {qmv_v_body}
    }}
"#,
        qmv_q_body = qmv_q_body,
        qmv_k_body = qmv_k_body,
        qmv_v_body = qmv_v_body,
    );

    // Symbol name: deterministic hash of the atom sequence + dtype + gs.
    // For the MVP we just use a readable name; production version uses
    // a structural hash.
    // `_bias` suffix distinguishes Qwen2-style biased pre-attn from
    // the bias-free Llama variant. The cost CSV's existing
    // `synth_pre_attn_*` rows match only the bias-free form; the
    // biased variant falls back to the analytical model in
    // `MetalSynthPreAttnImpl::cost_us` until a sweep lands.
    let bias_suffix = if consts.has_linear_bias { "_bias" } else { "" };
    let symbol = if init {
        format!(
            "synth_pre_attn_init_{}_{}_gs{}{}",
            t_act, t_scale, consts.group_size, bias_suffix,
        )
    } else {
        format!(
            "synth_pre_attn_{}_{}_gs{}{}",
            t_act, t_scale, consts.group_size, bias_suffix,
        )
    };

    // Conditional extra kernel params for the biased variant.
    // Indented to align with the surrounding signature.
    let maybe_bias_params = if consts.has_linear_bias {
        format!(
            "    device const {t_act}*   {q_lb_buf}      [[buffer(18)]],\n\
             \x20   device const {t_act}*   {k_lb_buf}      [[buffer(19)]],\n\
             \x20   device const {t_act}*   {v_lb_buf}      [[buffer(20)]],\n",
            t_act = t_act,
            q_lb_buf = q_lb_buf,
            k_lb_buf = k_lb_buf,
            v_lb_buf = v_lb_buf,
        )
    } else {
        String::new()
    };

    let source_tail = format!(
        r#"


// Model-invariant dims baked at synth time. Same source-compile
// optimizations as if they were hand-written literals: loop trip
// counts known, divisions strength-reduced, branches folded.
constant constexpr uint  HIDDEN     = {hidden_lit}u;
constant constexpr uint  NUM_Q      = {num_q_lit}u;
constant constexpr uint  NUM_KV     = {num_kv_lit}u;
constant constexpr uint  HEAD_DIM   = {head_dim_lit}u;
constant constexpr uint  ROT_DIM    = {rot_dim_lit}u;
constant constexpr uint  BLOCK_SIZE = {block_size_lit}u;
constant constexpr float EPS        = {eps_lit}f;
// `M` (active token count up to bucket capacity) stays a
// function constant — varies per dispatch bucket.
constant uint  M [[function_constant(0)]];

constant constexpr uint __HIDDEN_MAX  = 8192;
constant constexpr uint __HEAD_DIM_MAX = 256;
constant constexpr uint __SCRATCH_MAX  = __HEAD_DIM_MAX / MK_ROWS_PER_SIMDGROUP;

[[kernel]] void {symbol}(
    device       {t_act}*   {q_out_buf}      [[buffer(0)]],
    device       {t_act}*   {residual_io}    [[buffer(1)]],
    device const {t_act}*   {delta_buf}      [[buffer(2)]],
    device const {t_scale}* {rms_wt_buf}     [[buffer(3)]],
    device const uint32_t*  {q_wt_buf}       [[buffer(4)]],
    device const {t_scale}* {q_sc_buf}       [[buffer(5)]],
    device const {t_scale}* {q_bi_buf}       [[buffer(6)]],
    device const uint32_t*  {k_wt_buf}       [[buffer(7)]],
    device const {t_scale}* {k_sc_buf}       [[buffer(8)]],
    device const {t_scale}* {k_bi_buf}       [[buffer(9)]],
    device const uint32_t*  {v_wt_buf}       [[buffer(10)]],
    device const {t_scale}* {v_sc_buf}       [[buffer(11)]],
    device const {t_scale}* {v_bi_buf}       [[buffer(12)]],
    device const {t_act}*   {cos_sin_buf}    [[buffer(13)]],
    device const uint*      {positions_buf}  [[buffer(14)]],
    device const uint*      {slot_map_buf}   [[buffer(15)]],
    device       {t_act}*   {kv_cache_k}     [[buffer(16)]],
    device       {t_act}*   {kv_cache_v}     [[buffer(17)]],
{maybe_bias_params}    uint3 __tg_pos    [[threadgroup_position_in_grid]],
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
        q_wt_buf = q_wt_buf,
        q_sc_buf = q_sc_buf,
        q_bi_buf = q_bi_buf,
        k_wt_buf = k_wt_buf,
        k_sc_buf = k_sc_buf,
        k_bi_buf = k_bi_buf,
        v_wt_buf = v_wt_buf,
        v_sc_buf = v_sc_buf,
        v_bi_buf = v_bi_buf,
        cos_sin_buf = cos_sin_buf,
        positions_buf = positions_buf,
        slot_map_buf = slot_map_buf,
        kv_cache_k = kv_cache_k,
        kv_cache_v = kv_cache_v,
        maybe_bias_params = maybe_bias_params,
        x_norm_name = x_norm_name,
        qmv_smem_name = qmv_smem_name,
        addrms_body = addrms_body,
        qmv_body = qmv_body,
        rope_body = rope_body,
        hidden_lit = consts.hidden,
        num_q_lit = consts.num_q_heads,
        num_kv_lit = consts.num_kv_heads,
        head_dim_lit = consts.head_dim,
        rot_dim_lit = consts.rot_dim,
        block_size_lit = consts.block_size,
        eps_lit = format_msl_float(consts.rms_norm_eps),
    );

    // The runtime path (newLibraryWithSource) has no source-tree
    // visibility, so we inline `metal_kittens.h` directly into the
    // synthesized source. The header contains literal `{`/`}` which
    // would crash `format!` — so we concatenate raw (no format!) into
    // the source prefix below.
    let mk_header = inline_header(include_str!(
        "../../ferrite-metal-kernels/shaders/metal_kittens.h"
    ));
    let source = format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         //\n\
         // SYNTHESIZED KERNEL — generated by ferrite-forward-macro::fuse_pass.\n\
         // Do not hand-edit.\n\n\
         #include <metal_stdlib>\n\
         using namespace metal;\n\n\
         // === inlined metal_kittens.h ===\n\
         {mk_header}\n\
         // === end inlined metal_kittens.h ===\n\n\
         {source_tail}",
        mk_header = mk_header,
        source_tail = source_tail,
    );

    SynthesizedKernel {
        symbol,
        source,
        backend: SynthesisBackend::Metal,
    }
}

/// Synthesize the MLP pre-down chunk:
/// `FusedAddRmsNorm → AffineQmv (gate) → AffineQmv (up) → SiluMul`.
///
/// Output is a device buffer of shape `[M, intermediate_size]`. The
/// standalone down-projection AffineQmm follows in the instruction
/// stream and consumes it.
///
/// Mirrors `synthesize_pre_attn_chunk` structurally, with three
/// differences:
///   1. No RoPE / KV-cache write — the epilogue is a SiluMul atom.
///   2. The qmv tile axis spans `intermediate_size / TILE_N` tiles
///      (rather than `num_q + 2*num_kv` heads). TILE_N is set equal
///      to HEAD_DIM so the same `32 * TILE_N / MK_ROWS_PER_SIMDGROUP`
///      thread count and per-simdgroup row layout reuse the existing
///      AffineQmvAtom unchanged.
///   3. Two separate qmv calls write to two TG-mem float scratch
///      buffers (`gate_smem`, `up_smem`); the SiluMul atom reads both.
///
/// The AddRmsNorm atom's residual-writeback predicate
/// `(__head < __num_q)` is reused by setting kernel-scope
/// `__num_q = HIDDEN / TILE_N` — the first `hidden/TILE_N` TGs own
/// the disjoint slices of the HIDDEN-wide residual buffer.
pub fn synthesize_mlp_pre_down_chunk(
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

    let x_norm_name    = "__x_norm".to_string();
    let gate_smem_name = "__gate_smem".to_string();
    let up_smem_name   = "__up_smem".to_string();
    let residual_io    = "__residual_io".to_string();
    let delta_buf      = "__delta".to_string();
    let rms_wt_buf     = "__rms_weight".to_string();
    let gate_wt_buf    = "__gate_weight".to_string();
    let gate_sc_buf    = "__gate_scales".to_string();
    let gate_bi_buf    = "__gate_biases".to_string();
    let up_wt_buf      = "__up_weight".to_string();
    let up_sc_buf      = "__up_scales".to_string();
    let up_bi_buf      = "__up_biases".to_string();
    let silu_mul_out   = "__silu_mul_out".to_string();

    // Per-atom AtomCtx constants. The shape parameters arrive at
    // kernel launch time as function constants; this slice only
    // contributes to the symbol-name hashing for atoms that look at
    // group_size (AffineQmvAtom does).
    let constants_slice: Vec<(&'static str, AtomConstantValue)> = vec![
        ("HIDDEN",       AtomConstantValue::Uint(consts.hidden)),
        ("INTERMEDIATE", AtomConstantValue::Uint(consts.num_q_heads)), // overloaded — unused by atoms
        ("HEAD_DIM",     AtomConstantValue::Uint(consts.head_dim)),
        ("M",            AtomConstantValue::Uint(consts.m)),
        ("EPS",          AtomConstantValue::Float(consts.rms_norm_eps)),
    ];

    let addrms = AddRmsNormAtom::default();
    let silu_mul = SiluMulAtom;
    // Both qmv bands address row 0..intermediate of their own
    // separate weight buffer — local_head_expr is just `__head` (the
    // tile index, since TILE_N = __head_dim).
    let qmv = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "__head",
        // MLP gate/up have no per-row linear bias.
        has_linear_bias: false,
    };

    let addrms_in  = vec![residual_io.clone(), delta_buf.clone(), rms_wt_buf.clone()];
    let addrms_out = vec![x_norm_name.clone()];
    let addrms_ctx = AtomCtx {
        bound_inputs: &addrms_in,
        bound_outputs: &addrms_out,
        constants: &constants_slice,
        t_act,
        t_scale,
    };
    let addrms_body = addrms.emit_metal_body(&addrms_ctx)
        .expect("AddRmsNormAtom Metal emit");

    let emit_qmv_body = |w: &str, s: &str, b: &str, out_smem: &str| -> String {
        let q_in = vec![
            x_norm_name.clone(),
            w.to_string(),
            s.to_string(),
            b.to_string(),
        ];
        let q_out = vec![out_smem.to_string()];
        let q_ctx = AtomCtx {
            bound_inputs: &q_in,
            bound_outputs: &q_out,
            constants: &constants_slice,
            t_act,
            t_scale,
        };
        qmv.emit_metal_body(&q_ctx)
            .expect("AffineQmvAtom Metal emit")
    };
    let gate_qmv_body = emit_qmv_body(&gate_wt_buf, &gate_sc_buf, &gate_bi_buf, &gate_smem_name);
    let up_qmv_body   = emit_qmv_body(&up_wt_buf,   &up_sc_buf,   &up_bi_buf,   &up_smem_name);

    let sm_in  = vec![gate_smem_name.clone(), up_smem_name.clone()];
    let sm_out = vec![silu_mul_out.clone()];
    let sm_ctx = AtomCtx {
        bound_inputs: &sm_in,
        bound_outputs: &sm_out,
        constants: &constants_slice,
        t_act,
        t_scale,
    };
    let silu_mul_body = silu_mul.emit_metal_body(&sm_ctx)
        .expect("SiluMulAtom Metal emit");

    let symbol = format!(
        "synth_mlp_pre_down_{}_{}_gs{}",
        t_act,
        t_scale,
        consts.group_size,
    );

    let source_tail = format!(
        r#"


// Model-invariant dims baked at synth time. See pre-attn chunk for
// the rationale (constant folding, loop-trip-count known at source
// compile, no function-constant indirection at pipeline creation).
constant constexpr uint  HIDDEN_FC       = {hidden_lit}u;
constant constexpr uint  INTERMEDIATE_FC = {intermediate_lit}u;
constant constexpr uint  TILE_N_FC       = {tile_n_lit}u;
constant constexpr float EPS_FC          = {eps_lit}f;
// `M_FC` (active token count up to bucket capacity) stays a
// function constant — varies per dispatch bucket.
constant uint  M_FC [[function_constant(0)]];

constant constexpr uint __HIDDEN_MAX   = 8192;
constant constexpr uint __HEAD_DIM_MAX = 256;
constant constexpr uint __SCRATCH_MAX  = __HEAD_DIM_MAX / MK_ROWS_PER_SIMDGROUP;

[[kernel]] void {symbol}(
    device       {t_act}*   {silu_mul_out}  [[buffer(0)]],
    device       {t_act}*   {residual_io}   [[buffer(1)]],
    device const {t_act}*   {delta_buf}     [[buffer(2)]],
    device const {t_scale}* {rms_wt_buf}    [[buffer(3)]],
    device const uint32_t*  {gate_wt_buf}   [[buffer(4)]],
    device const {t_scale}* {gate_sc_buf}   [[buffer(5)]],
    device const {t_scale}* {gate_bi_buf}   [[buffer(6)]],
    device const uint32_t*  {up_wt_buf}     [[buffer(7)]],
    device const {t_scale}* {up_sc_buf}     [[buffer(8)]],
    device const {t_scale}* {up_bi_buf}     [[buffer(9)]],
    uint3 __tg_pos    [[threadgroup_position_in_grid]],
    uint3 __tid_pos   [[thread_position_in_threadgroup]],
    uint  __simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  __simd_lid  [[thread_index_in_simdgroup]])
{{
    const uint __t                = __tg_pos.x;
    const uint __head             = __tg_pos.y;
    const uint __tid              = __tid_pos.x;
    const uint __head_dim         = TILE_N_FC;
    const uint __hidden           = HIDDEN_FC;
    const uint __intermediate     = INTERMEDIATE_FC;
    // `__num_q` is repurposed in MLP scope as the count of TGs that
    // own a disjoint slice of the HIDDEN-wide residual buffer (one
    // tile per `__head_dim` of hidden). AddRmsNormAtom's
    // `(__head < __num_q)` predicate then gates residual writeback
    // correctly for the first hidden/TILE_N TGs.
    const uint __num_q            = __hidden / __head_dim;
    const uint __num_kv           = 0u;
    (void)__num_kv;
    const uint __num_heads_total  = __intermediate / __head_dim;
    const uint __threads_per_tg   = MK_SIMD_SIZE * __head_dim / MK_ROWS_PER_SIMDGROUP;
    const uint __num_simdgroups   = __head_dim / MK_ROWS_PER_SIMDGROUP;
    const float __eps             = EPS_FC;

    if (__t >= M_FC || __head >= __num_heads_total) return;

    threadgroup {t_act} {x_norm}[__HIDDEN_MAX];
    threadgroup float   __scratch[__SCRATCH_MAX];
    threadgroup float   {gate_smem}[__HEAD_DIM_MAX];
    threadgroup float   {up_smem}[__HEAD_DIM_MAX];

    {addrms_body}

    {gate_qmv_body}

    {up_qmv_body}

    {silu_mul_body}
}}
"#,
        symbol = symbol,
        t_act = t_act,
        t_scale = t_scale,
        silu_mul_out = silu_mul_out,
        residual_io = residual_io,
        delta_buf = delta_buf,
        rms_wt_buf = rms_wt_buf,
        gate_wt_buf = gate_wt_buf,
        gate_sc_buf = gate_sc_buf,
        gate_bi_buf = gate_bi_buf,
        up_wt_buf = up_wt_buf,
        up_sc_buf = up_sc_buf,
        up_bi_buf = up_bi_buf,
        x_norm = x_norm_name,
        gate_smem = gate_smem_name,
        up_smem = up_smem_name,
        addrms_body = addrms_body,
        gate_qmv_body = gate_qmv_body,
        up_qmv_body = up_qmv_body,
        silu_mul_body = silu_mul_body,
        hidden_lit = consts.hidden,
        intermediate_lit = consts.intermediate,
        tile_n_lit = consts.head_dim,
        eps_lit = format_msl_float(consts.rms_norm_eps),
    );

    let mk_header = inline_header(include_str!(
        "../../ferrite-metal-kernels/shaders/metal_kittens.h"
    ));
    let source = format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         //\n\
         // SYNTHESIZED KERNEL — generated by ferrite-forward-macro::fuse_pass.\n\
         // Do not hand-edit.\n\n\
         #include <metal_stdlib>\n\
         using namespace metal;\n\n\
         // === inlined metal_kittens.h ===\n\
         {mk_header}\n\
         // === end inlined metal_kittens.h ===\n\n\
         {source_tail}",
        mk_header = mk_header,
        source_tail = source_tail,
    );

    SynthesizedKernel {
        symbol,
        source,
        backend: SynthesisBackend::Metal,
    }
}

/// Persistent-envelope variant of `synthesize_mlp_pre_down_chunk`.
///
/// Strict superset of the non-persistent signature: same buffers at
/// the same indices through `buffer(9)`, plus one appended
/// `device atomic_uint* __barrier_counter` at buffer(10). The kernel
/// body is byte-identical through the SiluMul atom, with a trailing
/// cross-TG ticket-lock barrier (same pattern as
/// `synthesize_pre_attn_chunk_persistent`). `num_tgs` is derived
/// inline as `M_FC * (INTERMEDIATE_FC / TILE_N_FC)` so the worker
/// doesn't need to bind a scalar uniform.
pub fn synthesize_mlp_pre_down_chunk_persistent(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
) -> SynthesizedKernel {
    assert_eq!(
        backend,
        SynthesisBackend::Metal,
        "MVP only emits Metal",
    );

    let x_norm_name    = "__x_norm".to_string();
    let gate_smem_name = "__gate_smem".to_string();
    let up_smem_name   = "__up_smem".to_string();
    let residual_io    = "__residual_io".to_string();
    let delta_buf      = "__delta".to_string();
    let rms_wt_buf     = "__rms_weight".to_string();
    let gate_wt_buf    = "__gate_weight".to_string();
    let gate_sc_buf    = "__gate_scales".to_string();
    let gate_bi_buf    = "__gate_biases".to_string();
    let up_wt_buf      = "__up_weight".to_string();
    let up_sc_buf      = "__up_scales".to_string();
    let up_bi_buf      = "__up_biases".to_string();
    let silu_mul_out   = "__silu_mul_out".to_string();

    let constants_slice: Vec<(&'static str, AtomConstantValue)> = vec![
        ("HIDDEN",       AtomConstantValue::Uint(consts.hidden)),
        ("INTERMEDIATE", AtomConstantValue::Uint(consts.num_q_heads)),
        ("HEAD_DIM",     AtomConstantValue::Uint(consts.head_dim)),
        ("M",            AtomConstantValue::Uint(consts.m)),
        ("EPS",          AtomConstantValue::Float(consts.rms_norm_eps)),
    ];

    let addrms = AddRmsNormAtom::default();
    let silu_mul = SiluMulAtom;
    let qmv = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "__head",
        has_linear_bias: false,
    };

    let addrms_in  = vec![residual_io.clone(), delta_buf.clone(), rms_wt_buf.clone()];
    let addrms_out = vec![x_norm_name.clone()];
    let addrms_ctx = AtomCtx {
        bound_inputs: &addrms_in,
        bound_outputs: &addrms_out,
        constants: &constants_slice,
        t_act,
        t_scale,
    };
    let addrms_body = addrms.emit_metal_body(&addrms_ctx)
        .expect("AddRmsNormAtom Metal emit");

    let emit_qmv_body = |w: &str, s: &str, b: &str, out_smem: &str| -> String {
        let q_in = vec![
            x_norm_name.clone(),
            w.to_string(),
            s.to_string(),
            b.to_string(),
        ];
        let q_out = vec![out_smem.to_string()];
        let q_ctx = AtomCtx {
            bound_inputs: &q_in,
            bound_outputs: &q_out,
            constants: &constants_slice,
            t_act,
            t_scale,
        };
        qmv.emit_metal_body(&q_ctx)
            .expect("AffineQmvAtom Metal emit")
    };
    let gate_qmv_body = emit_qmv_body(&gate_wt_buf, &gate_sc_buf, &gate_bi_buf, &gate_smem_name);
    let up_qmv_body   = emit_qmv_body(&up_wt_buf,   &up_sc_buf,   &up_bi_buf,   &up_smem_name);

    let sm_in  = vec![gate_smem_name.clone(), up_smem_name.clone()];
    let sm_out = vec![silu_mul_out.clone()];
    let sm_ctx = AtomCtx {
        bound_inputs: &sm_in,
        bound_outputs: &sm_out,
        constants: &constants_slice,
        t_act,
        t_scale,
    };
    let silu_mul_body = silu_mul.emit_metal_body(&sm_ctx)
        .expect("SiluMulAtom Metal emit");

    let symbol = format!(
        "synth_mlp_pre_down_persistent_{}_{}_gs{}",
        t_act,
        t_scale,
        consts.group_size,
    );

    let counter_buf = "__barrier_counter";
    let counter_idx: u32 = 10;

    let mk_header = inline_header(include_str!(
        "../../ferrite-metal-kernels/shaders/metal_kittens.h"
    ));

    let source_tail = format!(
        r#"


// Model-invariant dims baked at synth time, identical to the
// non-persistent variant. M_FC stays a function constant.
constant constexpr uint  HIDDEN_FC       = {hidden_lit}u;
constant constexpr uint  INTERMEDIATE_FC = {intermediate_lit}u;
constant constexpr uint  TILE_N_FC       = {tile_n_lit}u;
constant constexpr float EPS_FC          = {eps_lit}f;
constant uint  M_FC [[function_constant(0)]];

constant constexpr uint __HIDDEN_MAX   = 8192;
constant constexpr uint __HEAD_DIM_MAX = 256;
constant constexpr uint __SCRATCH_MAX  = __HEAD_DIM_MAX / MK_ROWS_PER_SIMDGROUP;

[[kernel]] void {symbol}(
    device       {t_act}*   {silu_mul_out}  [[buffer(0)]],
    device       {t_act}*   {residual_io}   [[buffer(1)]],
    device const {t_act}*   {delta_buf}     [[buffer(2)]],
    device const {t_scale}* {rms_wt_buf}    [[buffer(3)]],
    device const uint32_t*  {gate_wt_buf}   [[buffer(4)]],
    device const {t_scale}* {gate_sc_buf}   [[buffer(5)]],
    device const {t_scale}* {gate_bi_buf}   [[buffer(6)]],
    device const uint32_t*  {up_wt_buf}     [[buffer(7)]],
    device const {t_scale}* {up_sc_buf}     [[buffer(8)]],
    device const {t_scale}* {up_bi_buf}     [[buffer(9)]],
    device atomic_uint*     {counter_buf}   [[buffer({counter_idx})]],
    uint3 __tg_pos    [[threadgroup_position_in_grid]],
    uint3 __tgs_per_grid [[threadgroups_per_grid]],
    uint3 __tid_pos   [[thread_position_in_threadgroup]],
    uint  __simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  __simd_lid  [[thread_index_in_simdgroup]])
{{
    const uint __t                = __tg_pos.x;
    const uint __head             = __tg_pos.y;
    const uint __tid              = __tid_pos.x;
    const uint __head_dim         = TILE_N_FC;
    const uint __hidden           = HIDDEN_FC;
    const uint __intermediate     = INTERMEDIATE_FC;
    const uint __num_q            = __hidden / __head_dim;
    const uint __num_kv           = 0u;
    (void)__num_kv;
    const uint __num_heads_total  = __intermediate / __head_dim;
    const uint __threads_per_tg   = MK_SIMD_SIZE * __head_dim / MK_ROWS_PER_SIMDGROUP;
    const uint __num_simdgroups   = __head_dim / MK_ROWS_PER_SIMDGROUP;
    const float __eps             = EPS_FC;

    // Read the runtime-dispatched grid count via the builtin — see
    // pre_attn persistent for the rationale (M_FC is the bucket cap;
    // m_scaling shrinks the grid to num_tokens at runtime).
    const uint __persistent_num_tgs = __tgs_per_grid.x * __tgs_per_grid.y;

    if (__t >= M_FC || __head >= __num_heads_total) return;

    threadgroup {t_act} {x_norm}[__HIDDEN_MAX];
    threadgroup float   __scratch[__SCRATCH_MAX];
    threadgroup float   {gate_smem}[__HEAD_DIM_MAX];
    threadgroup float   {up_smem}[__HEAD_DIM_MAX];

    {addrms_body}

    {gate_qmv_body}

    {up_qmv_body}

    {silu_mul_body}

    // Single-phase envelope: no trailing cross-TG barrier required.
    // Metal's implicit kernel-completion barrier already publishes
    // every TG's device writes by the time the dispatch returns.
    // The persistent counter binding (`{counter_buf}`) is reserved
    // for Phase 2c (multi-phase fusion) where the kernel chains
    // additional work after a cross-TG sync; until then, leaving
    // a ticket-lock here would force every TG to remain concurrently
    // resident — which exceeds M4 occupancy at INTERMEDIATE / TILE_N
    // dispatches (128 TGs on Llama-1B) and deadlocks.
    (void){counter_buf};
    (void)__persistent_num_tgs;
}}
"#,
        symbol = symbol,
        t_act = t_act,
        t_scale = t_scale,
        silu_mul_out = silu_mul_out,
        residual_io = residual_io,
        delta_buf = delta_buf,
        rms_wt_buf = rms_wt_buf,
        gate_wt_buf = gate_wt_buf,
        gate_sc_buf = gate_sc_buf,
        gate_bi_buf = gate_bi_buf,
        up_wt_buf = up_wt_buf,
        up_sc_buf = up_sc_buf,
        up_bi_buf = up_bi_buf,
        counter_buf = counter_buf,
        counter_idx = counter_idx,
        x_norm = x_norm_name,
        gate_smem = gate_smem_name,
        up_smem = up_smem_name,
        addrms_body = addrms_body,
        gate_qmv_body = gate_qmv_body,
        up_qmv_body = up_qmv_body,
        silu_mul_body = silu_mul_body,
        hidden_lit = consts.hidden,
        intermediate_lit = consts.intermediate,
        tile_n_lit = consts.head_dim,
        eps_lit = format_msl_float(consts.rms_norm_eps),
    );

    let source = format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         //\n\
         // SYNTHESIZED PERSISTENT MLP-PRE-DOWN KERNEL — generated by ferrite-fusion-synth::fuse_pass.\n\
         // Do not hand-edit.\n\n\
         #include <metal_stdlib>\n\
         #include <metal_atomic>\n\
         using namespace metal;\n\n\
         // === inlined metal_kittens.h ===\n\
         {mk_header}\n\
         // === end inlined metal_kittens.h ===\n\n\
         {source_tail}",
        mk_header = mk_header,
        source_tail = source_tail,
    );

    SynthesizedKernel {
        symbol,
        source,
        backend: SynthesisBackend::Metal,
    }
}

/// Synthesize the fused gate+up+SiluMul large-M prefill kernel.
///
/// Claims gate_Gemm + up_Gemm + Silu + Mul (4 tiles). Uses simdgroup_matrix
/// 8x8 GEMM tiles: BM=BN=BK=32, WM=WN=2, TM=TN=2, 128 threads per TG.
/// Gate and up projections share a single __Ws buffer (sequential passes).
/// Epilogue uses simdgroup_store to per-simdgroup scratch for SiluMul.
/// Compiled at -O3 (simdgroup_store epilogue is correct at -O3).
///
/// NOTE: this kernel includes <metal_simdgroup_matrix> directly in its
/// source. It does NOT add new code to metal_kittens.h — preserving
/// the hashes of synth_pre_attn / synth_mlp_pre_down so their cached
/// metallibs are never invalidated by changes to this kernel.
pub fn synthesize_gate_up_silu_mul_large_chunk(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
) -> SynthesizedKernel {
    assert_eq!(backend, SynthesisBackend::Metal, "Metal only");

    let symbol = format!(
        "synth_gate_up_silu_mul_large_{}_{}_gs{}",
        t_act, t_scale, consts.group_size,
    );

    let gs = consts.group_size;
    let silu_mul_out = "__silu_mul_out";
    let x_norm       = "__x_norm";
    let gate_wt      = "__gate_weight";
    let gate_sc      = "__gate_scales";
    let gate_bi      = "__gate_biases";
    let up_wt        = "__up_weight";
    let up_sc        = "__up_scales";
    let up_bi        = "__up_biases";

    let bk_pad = 40u32;  // BK(32) + 8 (bfloat bank-conflict pad)
    let group_steps = consts.group_size / 32; // gs/BK; gs=64 → 2

    let source_tail = format!(r#"
// BM=BN=BK=32, WM=WN=2, TM=TN=2, TGP=128. Mirrors MLX affine_qmm_t.
constant constexpr int BM_SG   = 32;
constant constexpr int BN_SG   = 32;
constant constexpr int BK_SG   = 32;
constant constexpr int BK_PAD  = {bk_pad};
constant constexpr int TGP_SG  = 128;
constant constexpr int WM_SG   = 2;
constant constexpr int WN_SG   = 2;
constant constexpr int TM_SG   = 2;
constant constexpr int TN_SG   = 2;
constant constexpr int KFR_SG  = 4;   // BK / 8
constant constexpr int GS_SG   = {gs};
constant constexpr int GS_STEPS = {group_steps};

constant constexpr uint  HIDDEN_LG       = {hidden}u;
constant constexpr uint  INTERMEDIATE_LG = {intermediate}u;
constant uint  M_LG [[function_constant(0)]];

[[kernel, max_total_threads_per_threadgroup(TGP_SG)]]
void {symbol}(
    device       {t_act}*   {silu_mul_out}  [[buffer(0)]],
    device const {t_act}*   {x_norm}        [[buffer(1)]],
    device const uint32_t*  {gate_wt}       [[buffer(2)]],
    device const {t_scale}* {gate_sc}       [[buffer(3)]],
    device const {t_scale}* {gate_bi}       [[buffer(4)]],
    device const uint32_t*  {up_wt}         [[buffer(5)]],
    device const {t_scale}* {up_sc}         [[buffer(6)]],
    device const {t_scale}* {up_bi}         [[buffer(7)]],
    uint3 __tgid [[threadgroup_position_in_grid]],
    uint  __simd_gid [[simdgroup_index_in_threadgroup]],
    uint  __simd_lid [[thread_index_in_simdgroup]])
{{
    const int __c_col = (int)__tgid.x * BN_SG;
    const int __c_row = (int)__tgid.y * BM_SG;
    if (__c_row >= (int)M_LG || __c_col >= (int)INTERMEDIATE_LG) return;

    const int __K = (int)HIDDEN_LG;
    const int __N = (int)INTERMEDIATE_LG;
    const int __M = (int)M_LG;
    const int __thread_idx = (int)(__simd_gid * 32u + __simd_lid);

    const int __m_tile = min(BM_SG, __M - __c_row);
    const int __n_tile = min(BN_SG, __N - __c_col);
    const bool __m_full = (__m_tile == BM_SG);
    const bool __n_full = (__n_tile == BN_SG);

    const int __bi_x = __thread_idx / 4;
    const int __bj_x = 8 * (__thread_idx % 4);
    const int __bi_w = (4 * __thread_idx) / 16;
    const int __bj_w = (4 * __thread_idx) % 16;

    threadgroup {t_act} __Xs   [BM_SG * BK_PAD];
    threadgroup {t_act} __Ws   [BN_SG * BK_PAD];
    threadgroup float   __ep_g [WM_SG * WN_SG * 64];
    threadgroup float   __ep_u [WM_SG * WN_SG * 64];

    const int __K_w = __K / 2;
    const int __K_g = __K / GS_SG;

    const device {t_act}*   __x_base  = {x_norm}  + (int64_t)__c_row * __K;
    const device uint8_t*   __wg_base = (const device uint8_t*){gate_wt} + (int64_t)__c_col * __K_w;
    const device {t_scale}* __sg_base = {gate_sc} + (int64_t)__c_col * __K_g;
    const device {t_scale}* __bg_base = {gate_bi} + (int64_t)__c_col * __K_g;
    const device uint8_t*   __wu_base = (const device uint8_t*){up_wt}   + (int64_t)__c_col * __K_w;
    const device {t_scale}* __su_base = {up_sc}   + (int64_t)__c_col * __K_g;
    const device {t_scale}* __bu_base = {up_bi}   + (int64_t)__c_col * __K_g;

    const device {t_act}*   __X_src  = __x_base  + __bi_x * __K + __bj_x;
    const device uint8_t*   __Wg_src = __wg_base + __bi_w * __K_w + __bj_w;
    const device {t_scale}* __Sg_row = __sg_base + __bi_w * __K_g;
    const device {t_scale}* __Bg_row = __bg_base + __bi_w * __K_g;
    const device uint8_t*   __Wu_src = __wu_base + __bi_w * __K_w + __bj_w;
    const device {t_scale}* __Su_row = __su_base + __bi_w * __K_g;
    const device {t_scale}* __Bu_row = __bu_base + __bi_w * __K_g;

    simdgroup_float8x8 __gate_acc[TM_SG][TN_SG];
    simdgroup_float8x8 __up_acc  [TM_SG][TN_SG];
    for (int __i = 0; __i < TM_SG; ++__i)
        for (int __j = 0; __j < TN_SG; ++__j) {{
            __gate_acc[__i][__j] = simdgroup_float8x8(0.0f);
            __up_acc  [__i][__j] = simdgroup_float8x8(0.0f);
        }}

    const int __sgM = (int)__simd_gid / WN_SG;
    const int __sgN = (int)__simd_gid % WN_SG;
    int __gs_cnt_g = 0, __gs_cnt_u = 0;

    for (int __k = 0; __k < __K; __k += BK_SG) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (__m_full || __bi_x < __m_tile) {{
            *((threadgroup vec<{t_act}, 4>*)(__Xs + __bi_x * BK_PAD + __bj_x))     = *((const device vec<{t_act}, 4>*)(__X_src));
            *((threadgroup vec<{t_act}, 4>*)(__Xs + __bi_x * BK_PAD + __bj_x + 4)) = *((const device vec<{t_act}, 4>*)(__X_src + 4));
        }} else {{
            *((threadgroup vec<{t_act}, 4>*)(__Xs + __bi_x * BK_PAD + __bj_x))     = vec<{t_act}, 4>(0);
            *((threadgroup vec<{t_act}, 4>*)(__Xs + __bi_x * BK_PAD + __bj_x + 4)) = vec<{t_act}, 4>(0);
        }}

        {{
            {t_act} __s0g = {t_act}(*__Sg_row), __b0g = {t_act}(*__Bg_row);
            {t_act} __s1g = __s0g / {t_act}(16.0f);
            if (__n_full || __bi_w < __n_tile) {{
                for (int __i = 0; __i < 4; ++__i) {{
                    uint8_t __b = __Wg_src[__i];
                    __Ws[__bi_w * BK_PAD + __bj_w * 2 + __i * 2 + 0] = __s0g * {t_act}(__b & 0x0f) + __b0g;
                    __Ws[__bi_w * BK_PAD + __bj_w * 2 + __i * 2 + 1] = __s1g * {t_act}(__b & 0xf0) + __b0g;
                }}
            }} else {{
                for (int __i = 0; __i < 8; ++__i) __Ws[__bi_w * BK_PAD + __bj_w * 2 + __i] = {t_act}(0);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int __kf = 0; __kf < KFR_SG; ++__kf) {{
            simdgroup_matrix<{t_act}, 8, 8> __A_frag[TM_SG];
            for (int __i = 0; __i < TM_SG; ++__i)
                simdgroup_load(__A_frag[__i], __Xs + (__sgM*16 + __i*8)*BK_PAD + __kf*8, BK_PAD);
            simdgroup_matrix<{t_act}, 8, 8> __Bg[TN_SG];
            for (int __j = 0; __j < TN_SG; ++__j)
                simdgroup_load(__Bg[__j], __Ws + (__sgN*16 + __j*8)*BK_PAD + __kf*8, BK_PAD, ulong2(0,0), true);
            for (int __i = 0; __i < TM_SG; ++__i)
                for (int __j = 0; __j < TN_SG; ++__j)
                    simdgroup_multiply_accumulate(__gate_acc[__i][__j], __A_frag[__i], __Bg[__j], __gate_acc[__i][__j]);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {{
            {t_act} __s0u = {t_act}(*__Su_row), __b0u = {t_act}(*__Bu_row);
            {t_act} __s1u = __s0u / {t_act}(16.0f);
            if (__n_full || __bi_w < __n_tile) {{
                for (int __i = 0; __i < 4; ++__i) {{
                    uint8_t __b = __Wu_src[__i];
                    __Ws[__bi_w * BK_PAD + __bj_w * 2 + __i * 2 + 0] = __s0u * {t_act}(__b & 0x0f) + __b0u;
                    __Ws[__bi_w * BK_PAD + __bj_w * 2 + __i * 2 + 1] = __s1u * {t_act}(__b & 0xf0) + __b0u;
                }}
            }} else {{
                for (int __i = 0; __i < 8; ++__i) __Ws[__bi_w * BK_PAD + __bj_w * 2 + __i] = {t_act}(0);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int __kf = 0; __kf < KFR_SG; ++__kf) {{
            simdgroup_matrix<{t_act}, 8, 8> __A_frag[TM_SG];
            for (int __i = 0; __i < TM_SG; ++__i)
                simdgroup_load(__A_frag[__i], __Xs + (__sgM*16 + __i*8)*BK_PAD + __kf*8, BK_PAD);
            simdgroup_matrix<{t_act}, 8, 8> __Bu[TN_SG];
            for (int __j = 0; __j < TN_SG; ++__j)
                simdgroup_load(__Bu[__j], __Ws + (__sgN*16 + __j*8)*BK_PAD + __kf*8, BK_PAD, ulong2(0,0), true);
            for (int __i = 0; __i < TM_SG; ++__i)
                for (int __j = 0; __j < TN_SG; ++__j)
                    simdgroup_multiply_accumulate(__up_acc[__i][__j], __A_frag[__i], __Bu[__j], __up_acc[__i][__j]);
        }}

        __X_src  += BK_SG;
        __Wg_src += BK_SG / 2;  __Wu_src += BK_SG / 2;
        if (++__gs_cnt_g == GS_STEPS) {{ __gs_cnt_g = 0; ++__Sg_row; ++__Bg_row; }}
        if (++__gs_cnt_u == GS_STEPS) {{ __gs_cnt_u = 0; ++__Su_row; ++__Bu_row; }}
    }}

    const int __ep_row = (int)__simd_lid / 4;
    const int __ep_c0  = ((int)__simd_lid % 4) * 2;
    const int __sg_off = (int)__simd_gid * 64;
    for (int __i = 0; __i < TM_SG; ++__i) {{
        for (int __j = 0; __j < TN_SG; ++__j) {{
            simdgroup_store(__gate_acc[__i][__j], __ep_g + __sg_off, 8);
            simdgroup_store(__up_acc  [__i][__j], __ep_u + __sg_off, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const int __m  = __c_row + __sgM*16 + __i*8 + __ep_row;
            const int __n0 = __c_col + __sgN*16 + __j*8 + __ep_c0;
            const int __n1 = __n0 + 1;
            if (__m < __M) {{
                if (__n0 < __N) {{
                    float __g0 = __ep_g[__sg_off + __ep_row * 8 + __ep_c0];
                    float __u0 = __ep_u[__sg_off + __ep_row * 8 + __ep_c0];
                    float __sg0 = __g0 / (1.0f + exp(-__g0));
                    {silu_mul_out}[(int64_t)__m * __N + __n0] = {t_act}(__sg0 * __u0);
                }}
                if (__n1 < __N) {{
                    float __g1 = __ep_g[__sg_off + __ep_row * 8 + __ep_c0 + 1];
                    float __u1 = __ep_u[__sg_off + __ep_row * 8 + __ep_c0 + 1];
                    float __sg1 = __g1 / (1.0f + exp(-__g1));
                    {silu_mul_out}[(int64_t)__m * __N + __n1] = {t_act}(__sg1 * __u1);
                }}
            }}
        }}
    }}
}}
"#,
        symbol = symbol, t_act = t_act, t_scale = t_scale, gs = gs,
        bk_pad = bk_pad, group_steps = group_steps,
        silu_mul_out = silu_mul_out, x_norm = x_norm,
        gate_wt = gate_wt, gate_sc = gate_sc, gate_bi = gate_bi,
        up_wt = up_wt, up_sc = up_sc, up_bi = up_bi,
        hidden = consts.hidden, intermediate = consts.intermediate,
    );

    // metal_kittens.h is included unchanged — do NOT add simdgroup_matrix
    // code to that header or all synth kernels will be recompiled.
    // This kernel includes <metal_simdgroup_matrix> directly in its header.
    let mk_header = inline_header(include_str!("../../ferrite-metal-kernels/shaders/metal_kittens.h"));
    let source = format!(
        "// SPDX-License-Identifier: Apache-2.0\n// SYNTHESIZED KERNEL — do not hand-edit.\n\n\
         #include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n\n\
         // === inlined metal_kittens.h ===\n{mk_header}\n// === end inlined metal_kittens.h ===\n\n{source_tail}",
        mk_header = mk_header, source_tail = source_tail,
    );

    SynthesizedKernel { symbol, source, backend: SynthesisBackend::Metal }
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
        intermediate:  8192,
        m:             1,
        group_size:    64,
        rms_norm_eps:  1e-5,
        has_linear_bias: false,
    };
    synthesize_pre_attn_chunk(SynthesisBackend::Metal, "bfloat", "half", &consts)
}

/// Persistent-envelope variant of `synthesize_pre_attn_chunk`.
///
/// Emits a kernel whose body is the same AddRmsNorm + 3×AffineQmv + RopeAppend
/// atom chain, but wrapped in the persistent envelope (counter @ buffer 0,
/// num_tgs @ buffer 1, all other buffer indices shifted by +2). The body is
/// emitted as a single `PersistentPhase` since this kernel has only one phase
/// (Phase 2b will introduce multi-phase variants that fuse across chunks).
///
/// The trailing `cross_tg_barrier_msl` after the single phase is a no-op for
/// downstream correctness when this kernel is dispatched standalone, but
/// costs ~5 µs per call. It exists because the envelope template emits one
/// barrier per phase including the last (uniform post-condition).
///
/// Symbol name: `synth_pre_attn_persistent_<t_act>_<t_scale>_gs<gs>[_bias]`
/// — distinguishes from the non-persistent variant so both can coexist in
/// the AOT-compiled metallib registry.
pub fn synthesize_pre_attn_chunk_persistent(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
) -> SynthesizedKernel {
    synthesize_pre_attn_chunk_persistent_impl(backend, t_act, t_scale, consts, false)
}

/// Layer-0 (init) variant of `synthesize_pre_attn_chunk_persistent`.
pub fn synthesize_pre_attn_init_chunk_persistent(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
) -> SynthesizedKernel {
    synthesize_pre_attn_chunk_persistent_impl(backend, t_act, t_scale, consts, true)
}

/// Append-style persistent variant: signature is a strict superset of
/// the non-persistent `synth_pre_attn_*` kernel (same buffers at the
/// same indices), plus one extra `device atomic_uint* __barrier_counter`
/// appended at the next free index. `num_tgs` is derived inline as
/// `M * (NUM_Q + 2u * NUM_KV)` so the worker doesn't need to bind a
/// scalar uniform. This keeps the interpreter arm a one-line patch
/// over the existing SynthPreAttn arm.
fn synthesize_pre_attn_chunk_persistent_impl(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
    init: bool,
) -> SynthesizedKernel {
    assert_eq!(
        backend,
        SynthesisBackend::Metal,
        "MVP only emits Metal",
    );

    // Channel-name allocation — same as `synthesize_pre_attn_chunk_impl`
    // (kept verbatim so atom-body emissions textually match).
    let x_norm_name   = "__x_norm".to_string();
    let qmv_smem_name = "__qmv_smem".to_string();
    let residual_io   = "__residual_io".to_string();
    let delta_buf     = "__delta".to_string();
    let rms_wt_buf    = "__rms_weight".to_string();
    let q_wt_buf = "__q_weight".to_string();
    let q_sc_buf = "__q_scales".to_string();
    let q_bi_buf = "__q_biases".to_string();
    let k_wt_buf = "__k_weight".to_string();
    let k_sc_buf = "__k_scales".to_string();
    let k_bi_buf = "__k_biases".to_string();
    let v_wt_buf = "__v_weight".to_string();
    let v_sc_buf = "__v_scales".to_string();
    let v_bi_buf = "__v_biases".to_string();
    let q_lb_buf = "__q_linear_bias".to_string();
    let k_lb_buf = "__k_linear_bias".to_string();
    let v_lb_buf = "__v_linear_bias".to_string();
    let cos_sin_buf   = "__cos_sin".to_string();
    let positions_buf = "__positions".to_string();
    let slot_map_buf  = "__slot_mapping".to_string();
    let q_out_buf     = "__q_out".to_string();
    let kv_cache_k    = "__kv_cache_k".to_string();
    let kv_cache_v    = "__kv_cache_v".to_string();

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

    let addrms = AddRmsNormAtom { init };
    let rope   = RopeAppendAtom;
    let qmv_q = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "__head",
        has_linear_bias: consts.has_linear_bias,
    };
    let qmv_k = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "(__head - __num_q)",
        has_linear_bias: consts.has_linear_bias,
    };
    let qmv_v = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "(__head - __num_q - __num_kv)",
        has_linear_bias: consts.has_linear_bias,
    };

    let addrms_in  = vec![residual_io.clone(), delta_buf.clone(), rms_wt_buf.clone()];
    let addrms_out = vec![x_norm_name.clone()];
    let addrms_ctx = AtomCtx {
        bound_inputs: &addrms_in,
        bound_outputs: &addrms_out,
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
    let rope_body   = rope.emit_metal_body(&rope_ctx)
        .expect("RopeAppendAtom Metal emit");

    let emit_band_body =
        |atom: &AffineQmvAtom, w: &str, s: &str, b: &str, lb: Option<&str>| -> String {
            let mut band_in = vec![
                x_norm_name.clone(), w.to_string(), s.to_string(), b.to_string(),
            ];
            if let Some(lb_name) = lb {
                band_in.push(lb_name.to_string());
            }
            let band_out = vec![qmv_smem_name.clone()];
            let band_ctx = AtomCtx {
                bound_inputs: &band_in,
                bound_outputs: &band_out,
                constants: &constants_slice,
                t_act,
                t_scale,
            };
            atom.emit_metal_body(&band_ctx)
                .expect("AffineQmvAtom Metal emit")
        };
    let q_lb_opt: Option<&str> = consts.has_linear_bias.then_some(q_lb_buf.as_str());
    let k_lb_opt: Option<&str> = consts.has_linear_bias.then_some(k_lb_buf.as_str());
    let v_lb_opt: Option<&str> = consts.has_linear_bias.then_some(v_lb_buf.as_str());
    let qmv_q_body = emit_band_body(&qmv_q, &q_wt_buf, &q_sc_buf, &q_bi_buf, q_lb_opt);
    let qmv_k_body = emit_band_body(&qmv_k, &k_wt_buf, &k_sc_buf, &k_bi_buf, k_lb_opt);
    let qmv_v_body = emit_band_body(&qmv_v, &v_wt_buf, &v_sc_buf, &v_bi_buf, v_lb_opt);
    let qmv_body = format!(
        r#"
    // --- per-band QKV qmv (3× AffineQmvAtom, band-selected by __head) ---
    if (__head < __num_q) {{
        {qmv_q_body}
    }} else if (__head < __num_q + __num_kv) {{
        {qmv_k_body}
    }} else {{
        {qmv_v_body}
    }}
"#,
    );

    let bias_suffix = if consts.has_linear_bias { "_bias" } else { "" };
    let symbol = if init {
        format!(
            "synth_pre_attn_init_persistent_{}_{}_gs{}{}",
            t_act, t_scale, consts.group_size, bias_suffix,
        )
    } else {
        format!(
            "synth_pre_attn_persistent_{}_{}_gs{}{}",
            t_act, t_scale, consts.group_size, bias_suffix,
        )
    };

    // Buffer indices: same as the non-persistent kernel through
    // buffer(17) (kv_cache_v); the persistent variant only ADDS the
    // counter at the next free index.
    //   counter index = 18 (no bias) or 21 (with bias)
    // num_tgs = M * (NUM_Q + 2 * NUM_KV) is derived inline so the
    // worker doesn't need to bind a uniform scalar.
    let counter_buf = "__barrier_counter";
    let counter_idx: u32 = if consts.has_linear_bias { 21 } else { 18 };

    let maybe_bias_params = if consts.has_linear_bias {
        format!(
            "    device const {t_act}*   {q_lb_buf}      [[buffer(18)]],\n\
             \x20   device const {t_act}*   {k_lb_buf}      [[buffer(19)]],\n\
             \x20   device const {t_act}*   {v_lb_buf}      [[buffer(20)]],\n",
            t_act = t_act,
            q_lb_buf = q_lb_buf, k_lb_buf = k_lb_buf, v_lb_buf = v_lb_buf,
        )
    } else {
        String::new()
    };

    let mk_header = inline_header(include_str!(
        "../../ferrite-metal-kernels/shaders/metal_kittens.h"
    ));

    let source_tail = format!(
        r#"

// Model-invariant dims baked at synth time, identical to the
// non-persistent variant. M stays a function constant.
constant constexpr uint  HIDDEN     = {hidden_lit}u;
constant constexpr uint  NUM_Q      = {num_q_lit}u;
constant constexpr uint  NUM_KV     = {num_kv_lit}u;
constant constexpr uint  HEAD_DIM   = {head_dim_lit}u;
constant constexpr uint  ROT_DIM    = {rot_dim_lit}u;
constant constexpr uint  BLOCK_SIZE = {block_size_lit}u;
constant constexpr float EPS        = {eps_lit}f;
constant uint  M [[function_constant(0)]];

constant constexpr uint __HIDDEN_MAX   = HIDDEN;
constant constexpr uint __HEAD_DIM_MAX = HEAD_DIM;
constant constexpr uint __SCRATCH_MAX  = __HEAD_DIM_MAX / MK_ROWS_PER_SIMDGROUP;

[[kernel, max_total_threads_per_threadgroup({threads_per_tg}u)]]
void {symbol}(
    device       {t_act}*   {q_out_buf}      [[buffer(0)]],
    device       {t_act}*   {residual_io}    [[buffer(1)]],
    device const {t_act}*   {delta_buf}      [[buffer(2)]],
    device const {t_scale}* {rms_wt_buf}     [[buffer(3)]],
    device const uint32_t*  {q_wt_buf}       [[buffer(4)]],
    device const {t_scale}* {q_sc_buf}       [[buffer(5)]],
    device const {t_scale}* {q_bi_buf}       [[buffer(6)]],
    device const uint32_t*  {k_wt_buf}       [[buffer(7)]],
    device const {t_scale}* {k_sc_buf}       [[buffer(8)]],
    device const {t_scale}* {k_bi_buf}       [[buffer(9)]],
    device const uint32_t*  {v_wt_buf}       [[buffer(10)]],
    device const {t_scale}* {v_sc_buf}       [[buffer(11)]],
    device const {t_scale}* {v_bi_buf}       [[buffer(12)]],
    device const {t_act}*   {cos_sin_buf}    [[buffer(13)]],
    device const uint*      {positions_buf}  [[buffer(14)]],
    device const uint*      {slot_map_buf}   [[buffer(15)]],
    device       {t_act}*   {kv_cache_k}     [[buffer(16)]],
    device       {t_act}*   {kv_cache_v}     [[buffer(17)]],
{maybe_bias_params}    device atomic_uint* {counter_buf} [[buffer({counter_idx})]],
    uint3 __tg_pos    [[threadgroup_position_in_grid]],
    uint3 __tgs_per_grid [[threadgroups_per_grid]],
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

    // Persistent-envelope barrier-state. The runtime scales the X-axis
    // grid by m_scaling to `num_tokens` (≤ M=bucket_m cap), so the
    // actual dispatched grid is `num_tokens × num_heads_total`. Read
    // it via the `[[threadgroups_per_grid]]` builtin — relying on
    // `M * num_heads_total` would over-count when num_tokens < M
    // (bucket cap) and the counter spins forever.
    const uint __persistent_num_tgs = __tgs_per_grid.x * __tgs_per_grid.y;

    if (__t >= M || __head >= __num_heads_total) return;

    threadgroup {t_act} {x_norm_name}[__HIDDEN_MAX];
    threadgroup float   __scratch  [__SCRATCH_MAX];
    threadgroup float   {qmv_smem_name}[__HEAD_DIM_MAX];

    {addrms_body}

    {qmv_body}

    {rope_body}

    // Single-phase envelope: no trailing cross-TG barrier required.
    // See `synthesize_mlp_pre_down_chunk_persistent` for the
    // rationale — the persistent counter stays bound (Phase 2c
    // multi-phase fusion will need it) but is otherwise unused so a
    // grid larger than M4 concurrent-residency doesn't deadlock.
    (void){counter_buf};
    (void)__persistent_num_tgs;
}}
"#,
        symbol = symbol, t_act = t_act, t_scale = t_scale,
        q_out_buf = q_out_buf, residual_io = residual_io,
        delta_buf = delta_buf, rms_wt_buf = rms_wt_buf,
        q_wt_buf = q_wt_buf, q_sc_buf = q_sc_buf, q_bi_buf = q_bi_buf,
        k_wt_buf = k_wt_buf, k_sc_buf = k_sc_buf, k_bi_buf = k_bi_buf,
        v_wt_buf = v_wt_buf, v_sc_buf = v_sc_buf, v_bi_buf = v_bi_buf,
        cos_sin_buf = cos_sin_buf, positions_buf = positions_buf,
        slot_map_buf = slot_map_buf,
        kv_cache_k = kv_cache_k, kv_cache_v = kv_cache_v,
        maybe_bias_params = maybe_bias_params,
        counter_buf = counter_buf, counter_idx = counter_idx,
        x_norm_name = x_norm_name, qmv_smem_name = qmv_smem_name,
        addrms_body = addrms_body,
        qmv_body = qmv_body,
        rope_body = rope_body,
        threads_per_tg = 8 * consts.head_dim,
        hidden_lit = consts.hidden,
        num_q_lit = consts.num_q_heads,
        num_kv_lit = consts.num_kv_heads,
        head_dim_lit = consts.head_dim,
        rot_dim_lit = consts.rot_dim,
        block_size_lit = consts.block_size,
        eps_lit = format_msl_float(consts.rms_norm_eps),
    );

    let source = format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         //\n\
         // SYNTHESIZED PERSISTENT PRE-ATTN KERNEL — generated by ferrite-fusion-synth::fuse_pass.\n\
         // Do not hand-edit.\n\n\
         #include <metal_stdlib>\n\
         #include <metal_atomic>\n\
         using namespace metal;\n\n\
         // === inlined metal_kittens.h ===\n\
         {mk_header}\n\
         // === end inlined metal_kittens.h ===\n\n\
         {source_tail}",
        mk_header = mk_header,
        source_tail = source_tail,
    );

    SynthesizedKernel {
        symbol,
        source,
        backend: SynthesisBackend::Metal,
    }
}

// ───────────────────────────────────────────────────────────────────
// Phase 2c: pre-attn + in-kernel BN=8 attention persistent megakernel
// ───────────────────────────────────────────────────────────────────
//
// Two-phase persistent kernel collapsing the existing per-chunk pair
// (synth_pre_attn_persistent_* dispatch + AttentionViaCache dispatch)
// into one dispatch per layer. Eliminates ~16 host dispatches/decode
// token on Llama-3.2-1B (one attention dispatch per layer × 16 layers).
//
// Phase 0: pre-attn (AddRmsNorm → QKV qmv → RopeAppend), same atom
// bodies as `synth_pre_attn_persistent_*`.
//
// Cross-TG barrier (ticket-lock; `__barrier_counter < num_tgs * 1u`)
// to ensure all TGs have flushed Q + KV writes to L2 before attention
// reads them.
//
// Phase 1: BN=8 paged-cache attention via `bn8_attention_body_msl`,
// dispatched at the same grid as pre-attn `(M × num_heads_total)`. The
// attention body guards `if (__head < NUM_Q)` so heads dispatched for
// K/V projections in pre-attn sit idle during attention while still
// bumping the surrounding barrier counter (no deadlock).
//
// Dispatch sizing: grid stays `(num_tokens × num_heads_total)` →
// `1 × 48` for Llama-3.2-1B (32 Q + 8 K + 8 V). On M4-base with
// `threads_per_tg = 32 * HEAD_DIM / MK_ROWS_PER_SIMDGROUP` = 256 (for
// HEAD_DIM=64), `safe_max_concurrent_tgs(256) = 118` so 48 TGs fit
// well under the residency cap — no deadlock risk.
//
// 1B-only today: HEAD_DIM=64 implies threads_per_tg=256 which matches
// what the BN=8 attention atom needs (BN×32 = 256). For HEAD_DIM=128
// (3B), pre-attn uses 512 threads_per_tg and the BN=8 attention atom
// would only use 256 of them — needs either a pre-attn-at-256 redesign
// or a BN=16 attention variant. Tracked for Step 2 (whole-forward).
pub fn synthesize_pre_attn_attn_chunk_persistent(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
    threads_per_tg: u32,
) -> SynthesizedKernel {
    synthesize_pre_attn_attn_chunk_persistent_impl(backend, t_act, t_scale, consts, threads_per_tg, false)
}

/// Layer-0 (init) variant of `synthesize_pre_attn_attn_chunk_persistent`.
pub fn synthesize_pre_attn_attn_init_chunk_persistent(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
    threads_per_tg: u32,
) -> SynthesizedKernel {
    synthesize_pre_attn_attn_chunk_persistent_impl(backend, t_act, t_scale, consts, threads_per_tg, true)
}

fn synthesize_pre_attn_attn_chunk_persistent_impl(
    backend: SynthesisBackend,
    t_act: &'static str,
    t_scale: &'static str,
    consts: &ChunkConstants,
    threads_per_tg: u32,
    init: bool,
) -> SynthesizedKernel {
    assert_eq!(
        backend,
        SynthesisBackend::Metal,
        "MVP only emits Metal",
    );
    assert!(
        consts.head_dim % 32 == 0,
        "Phase 2c attention body requires HEAD_DIM divisible by 32 \
         (qk_per_thread = HEAD_DIM/32). HEAD_DIM={}",
        consts.head_dim,
    );
    // threads_per_tg shape contract: must be a multiple of MK_SIMD_SIZE
    // (32) so it decomposes cleanly into simdgroups. The chosen value
    // is chip-driven (the persistent megakernel needs ONE fixed value
    // across all phases, picked from `MetalTargetProfile::
    // safe_max_concurrent_tgs` given the dispatched-TG count). For
    // M4-base targeting Llama-3.2 today, 256 (= 8 simdgroups × 32
    // lanes) is the choice — fits the 48-TG (num_q+2*num_kv on 1B)
    // dispatch comfortably under the safe cap of 118, and pairs with
    // the BN=8 attention atom (BN×32 = 256).
    assert!(
        threads_per_tg % MK_SIMD_SIZE == 0,
        "threads_per_tg must be a multiple of MK_SIMD_SIZE (32); got {threads_per_tg}",
    );
    let num_simdgroups = threads_per_tg / MK_SIMD_SIZE;
    assert!(
        num_simdgroups == 8,
        "Phase 2c currently pairs with the BN=8 attention atom, requiring \
         threads_per_tg = 256 (8 simdgroups × 32 lanes). Got threads_per_tg={threads_per_tg} \
         (would yield {num_simdgroups} simdgroups). Lift this when bn_attention_body_msl \
         is parameterized on BN.",
    );
    // HEAD_DIM must divide evenly into the atom-loop's rows-per-pass
    // (num_simdgroups × MK_ROWS_PER_SIMDGROUP = 8 × 8 = 64 for the
    // BN=8 / 256-threads-per-tg case). HEAD_DIM=64 → 1 pass, HEAD_DIM=128
    // → 2 passes, HEAD_DIM=192 → 3 passes, HEAD_DIM=256 → 4 passes.
    let rows_per_pass = num_simdgroups * MK_ROWS_PER_SIMDGROUP;
    assert!(
        consts.head_dim % rows_per_pass == 0,
        "HEAD_DIM ({}) must be a multiple of num_simdgroups*MK_ROWS_PER_SIMDGROUP ({rows_per_pass})",
        consts.head_dim,
    );

    // Channel-name allocation — identical to
    // `synthesize_pre_attn_chunk_persistent_impl` for the pre-attn
    // bindings, plus three new bindings for the attention phase
    // (`__attn_out`, `__seq_used_k`, `__block_table`).
    let x_norm_name   = "__x_norm".to_string();
    let qmv_smem_name = "__qmv_smem".to_string();
    let residual_io   = "__residual_io".to_string();
    let delta_buf     = "__delta".to_string();
    let rms_wt_buf    = "__rms_weight".to_string();
    let q_wt_buf      = "__q_weight".to_string();
    let q_sc_buf      = "__q_scales".to_string();
    let q_bi_buf      = "__q_biases".to_string();
    let k_wt_buf      = "__k_weight".to_string();
    let k_sc_buf      = "__k_scales".to_string();
    let k_bi_buf      = "__k_biases".to_string();
    let v_wt_buf      = "__v_weight".to_string();
    let v_sc_buf      = "__v_scales".to_string();
    let v_bi_buf      = "__v_biases".to_string();
    let q_lb_buf      = "__q_linear_bias".to_string();
    let k_lb_buf      = "__k_linear_bias".to_string();
    let v_lb_buf      = "__v_linear_bias".to_string();
    let cos_sin_buf   = "__cos_sin".to_string();
    let positions_buf = "__positions".to_string();
    let slot_map_buf  = "__slot_mapping".to_string();
    let q_out_buf     = "__q_out".to_string();
    let kv_cache_k    = "__kv_cache_k".to_string();
    let kv_cache_v    = "__kv_cache_v".to_string();
    // New for Phase 2c.
    let attn_out_buf    = "__attn_out".to_string();
    let seq_used_k_buf  = "__seq_used_k".to_string();
    let block_table_buf = "__block_table".to_string();

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

    // Atom set + bodies — identical to the pre-attn-only persistent
    // variant. Replicated rather than refactored-out to keep this
    // function self-contained until Phase 2c step B lands and the
    // duplication can be unified through an atom-DAG walker.
    let addrms = AddRmsNormAtom { init };
    let rope   = RopeAppendAtom;
    let qmv_q = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "__head",
        has_linear_bias: consts.has_linear_bias,
    };
    let qmv_k = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "(__head - __num_q)",
        has_linear_bias: consts.has_linear_bias,
    };
    let qmv_v = AffineQmvAtom {
        group_size: consts.group_size,
        local_head_expr: "(__head - __num_q - __num_kv)",
        has_linear_bias: consts.has_linear_bias,
    };

    let addrms_in  = vec![residual_io.clone(), delta_buf.clone(), rms_wt_buf.clone()];
    let addrms_out = vec![x_norm_name.clone()];
    let addrms_ctx = AtomCtx {
        bound_inputs: &addrms_in, bound_outputs: &addrms_out,
        constants: &constants_slice, t_act, t_scale,
    };
    let rope_in  = vec![
        qmv_smem_name.clone(), cos_sin_buf.clone(),
        positions_buf.clone(), slot_map_buf.clone(),
    ];
    let rope_out = vec![q_out_buf.clone(), kv_cache_k.clone(), kv_cache_v.clone()];
    let rope_ctx = AtomCtx {
        bound_inputs: &rope_in, bound_outputs: &rope_out,
        constants: &constants_slice, t_act, t_scale,
    };

    let addrms_body = addrms.emit_metal_body(&addrms_ctx).expect("AddRmsNormAtom Metal emit");
    let rope_body   = rope.emit_metal_body(&rope_ctx).expect("RopeAppendAtom Metal emit");

    let emit_band_body =
        |atom: &AffineQmvAtom, w: &str, s: &str, b: &str, lb: Option<&str>| -> String {
            let mut band_in = vec![
                x_norm_name.clone(), w.to_string(), s.to_string(), b.to_string(),
            ];
            if let Some(lb_name) = lb {
                band_in.push(lb_name.to_string());
            }
            let band_out = vec![qmv_smem_name.clone()];
            let band_ctx = AtomCtx {
                bound_inputs: &band_in, bound_outputs: &band_out,
                constants: &constants_slice, t_act, t_scale,
            };
            atom.emit_metal_body(&band_ctx).expect("AffineQmvAtom Metal emit")
        };
    let q_lb_opt: Option<&str> = consts.has_linear_bias.then_some(q_lb_buf.as_str());
    let k_lb_opt: Option<&str> = consts.has_linear_bias.then_some(k_lb_buf.as_str());
    let v_lb_opt: Option<&str> = consts.has_linear_bias.then_some(v_lb_buf.as_str());
    let qmv_q_body = emit_band_body(&qmv_q, &q_wt_buf, &q_sc_buf, &q_bi_buf, q_lb_opt);
    let qmv_k_body = emit_band_body(&qmv_k, &k_wt_buf, &k_sc_buf, &k_bi_buf, k_lb_opt);
    let qmv_v_body = emit_band_body(&qmv_v, &v_wt_buf, &v_sc_buf, &v_bi_buf, v_lb_opt);
    let qmv_body = format!(
        r#"
    // --- per-band QKV qmv (3× AffineQmvAtom, band-selected by __head) ---
    if (__head < __num_q) {{
        {qmv_q_body}
    }} else if (__head < __num_q + __num_kv) {{
        {qmv_k_body}
    }} else {{
        {qmv_v_body}
    }}
"#,
    );

    let attn_body = bn8_attention_body_msl(
        t_act,
        &q_out_buf, &attn_out_buf,
        &seq_used_k_buf, &block_table_buf,
        &kv_cache_k, &kv_cache_v,
    );
    let barrier_body = cross_tg_barrier_msl(0, "__barrier_counter");

    // Symbol encodes HEAD_DIM + threads_per_tg so distinct
    // (model, chip-choice) pairs don't collide in the
    // SpecializedPipelineCache when multiple model variants are loaded
    // by the same worker.
    let bias_suffix = if consts.has_linear_bias { "_bias" } else { "" };
    let symbol = if init {
        format!(
            "synth_pre_attn_attn_init_persistent_{}_{}_gs{}_hd{}_t{}{}",
            t_act, t_scale, consts.group_size, consts.head_dim, threads_per_tg, bias_suffix,
        )
    } else {
        format!(
            "synth_pre_attn_attn_persistent_{}_{}_gs{}_hd{}_t{}{}",
            t_act, t_scale, consts.group_size, consts.head_dim, threads_per_tg, bias_suffix,
        )
    };

    // Buffer indices: pre-attn 0..17 (same as synth_pre_attn_persistent),
    // optional bias 18..20, then attention's new bindings:
    //   attn_out     at next free index
    //   seq_used_k   at +1
    //   block_table  at +2
    //   barrier_counter at +3
    let pre_attn_end: u32 = if consts.has_linear_bias { 21 } else { 18 };
    let attn_out_idx:    u32 = pre_attn_end;
    let seq_used_k_idx:  u32 = pre_attn_end + 1;
    let block_table_idx: u32 = pre_attn_end + 2;
    let counter_idx:     u32 = pre_attn_end + 3;

    let maybe_bias_params = if consts.has_linear_bias {
        format!(
            "    device const {t_act}*   {q_lb_buf}      [[buffer(18)]],\n\
             \x20   device const {t_act}*   {k_lb_buf}      [[buffer(19)]],\n\
             \x20   device const {t_act}*   {v_lb_buf}      [[buffer(20)]],\n",
            t_act = t_act,
            q_lb_buf = q_lb_buf, k_lb_buf = k_lb_buf, v_lb_buf = v_lb_buf,
        )
    } else {
        String::new()
    };

    let mk_header = inline_header(include_str!(
        "../../ferrite-metal-kernels/shaders/metal_kittens.h"
    ));

    // ATTN_SCALE_BAKED = 1/sqrt(HEAD_DIM), computed at synth time so
    // the attention body can use a constexpr float literal (no
    // function-constant indirection, no runtime sqrt).
    let attn_scale = 1.0_f64 / (consts.head_dim as f64).sqrt();
    let source_tail = format!(
        r#"

// Model-invariant dims baked at synth time, identical to the pre-attn
// persistent variant. M stays a function constant.
constant constexpr uint  HIDDEN     = {hidden_lit}u;
constant constexpr uint  NUM_Q      = {num_q_lit}u;
constant constexpr uint  NUM_KV     = {num_kv_lit}u;
constant constexpr uint  HEAD_DIM   = {head_dim_lit}u;
constant constexpr uint  ROT_DIM    = {rot_dim_lit}u;
constant constexpr uint  BLOCK_SIZE = {block_size_lit}u;
constant constexpr float EPS        = {eps_lit}f;
constant constexpr float ATTN_SCALE_BAKED = {attn_scale_lit}f;
constant uint  M                  [[function_constant(0)]];
constant uint  MAX_BLOCKS_PER_SEQ [[function_constant(1)]];

constant constexpr uint __HIDDEN_MAX   = HIDDEN;
constant constexpr uint __HEAD_DIM_MAX = HEAD_DIM;
constant constexpr uint __SCRATCH_MAX  = __HEAD_DIM_MAX / MK_ROWS_PER_SIMDGROUP;

[[kernel, max_total_threads_per_threadgroup({threads_per_tg_lit}u)]]
void {symbol}(
    device       {t_act}*   {q_out_buf}      [[buffer(0)]],
    device       {t_act}*   {residual_io}    [[buffer(1)]],
    device const {t_act}*   {delta_buf}      [[buffer(2)]],
    device const {t_scale}* {rms_wt_buf}     [[buffer(3)]],
    device const uint32_t*  {q_wt_buf}       [[buffer(4)]],
    device const {t_scale}* {q_sc_buf}       [[buffer(5)]],
    device const {t_scale}* {q_bi_buf}       [[buffer(6)]],
    device const uint32_t*  {k_wt_buf}       [[buffer(7)]],
    device const {t_scale}* {k_sc_buf}       [[buffer(8)]],
    device const {t_scale}* {k_bi_buf}       [[buffer(9)]],
    device const uint32_t*  {v_wt_buf}       [[buffer(10)]],
    device const {t_scale}* {v_sc_buf}       [[buffer(11)]],
    device const {t_scale}* {v_bi_buf}       [[buffer(12)]],
    device const {t_act}*   {cos_sin_buf}    [[buffer(13)]],
    device const uint*      {positions_buf}  [[buffer(14)]],
    device const uint*      {slot_map_buf}   [[buffer(15)]],
    device       {t_act}*   {kv_cache_k}     [[buffer(16)]],
    device       {t_act}*   {kv_cache_v}     [[buffer(17)]],
{maybe_bias_params}    device       {t_act}*   {attn_out_buf}    [[buffer({attn_out_idx})]],
    device const uint*      {seq_used_k_buf} [[buffer({seq_used_k_idx})]],
    device const uint*      {block_table_buf} [[buffer({block_table_idx})]],
    device atomic_uint* __barrier_counter [[buffer({counter_idx})]],
    uint3 __tg_pos    [[threadgroup_position_in_grid]],
    uint3 __tgs_per_grid [[threadgroups_per_grid]],
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
    // threads_per_tg + num_simdgroups are SHAPE-DECOUPLED from HEAD_DIM
    // — baked from the synth-time `threads_per_tg` parameter, not
    // derived from `HEAD_DIM`. The atom bodies' HEAD_DIM-axis pass
    // loops use these to cover HEAD_DIM in `head_dim / (num_simdgroups
    // * MK_ROWS_PER_SIMDGROUP)` passes (1 for HEAD_DIM=64, 2 for
    // HEAD_DIM=128, etc.). The whole persistent megakernel runs at one
    // fixed `(threads_per_tg, dispatched_tgs)` regardless of model
    // HEAD_DIM (see project_persistent_decode_handoff).
    const uint __threads_per_tg   = {threads_per_tg_lit}u;
    const uint __num_simdgroups   = {num_simdgroups_lit}u;
    const float __eps             = EPS;

    // Persistent envelope: total TGs derived from `[[threadgroups_per_grid]]`
    // (the runtime m_scaling shrinks the X-axis to num_tokens). Required
    // by the cross-TG ticket-lock barrier.
    const uint num_tgs = __tgs_per_grid.x * __tgs_per_grid.y;

    // NOTE: no early-return guard. The dispatch grid is exactly
    // `(num_tokens × __num_heads_total)`, so every dispatched TG has
    // valid __t / __head and proceeds through both phases + the
    // intervening barrier. An early-return here would skip the barrier
    // and deadlock the spin in TGs that did reach it.

    threadgroup {t_act} {x_norm_name}[__HIDDEN_MAX];
    threadgroup float   __scratch  [__SCRATCH_MAX];
    threadgroup float   {qmv_smem_name}[__HEAD_DIM_MAX];

    // ── phase 0: pre-attn (AddRmsNorm → QKV qmv → RopeAppend) ──
    {addrms_body}

    {qmv_body}

    {rope_body}

    // ── cross-TG barrier: pre-attn → attention ──
    // Every dispatched TG must reach this barrier (no early return
    // above). Heads with __head >= NUM_Q sit idle inside the attention
    // body but still execute this barrier on their way to it.
    // The `cross_tg_barrier_msl` helper emits `if (tid == 0u)`; atom
    // bodies use `__tid` (pre-attn convention), so we alias the two.
    const uint tid = __tid;
    {barrier_body}

    // ── phase 1: BN=8 paged-cache decode attention ──
    {attn_body}

    // No trailing barrier — the downstream chunk (o_proj + MLP) is
    // dispatched as a separate kernel which provides the L2-flush via
    // the queue-level fence between kernel boundaries.
}}
"#,
        symbol = symbol, t_act = t_act, t_scale = t_scale,
        q_out_buf = q_out_buf, residual_io = residual_io,
        delta_buf = delta_buf, rms_wt_buf = rms_wt_buf,
        q_wt_buf = q_wt_buf, q_sc_buf = q_sc_buf, q_bi_buf = q_bi_buf,
        k_wt_buf = k_wt_buf, k_sc_buf = k_sc_buf, k_bi_buf = k_bi_buf,
        v_wt_buf = v_wt_buf, v_sc_buf = v_sc_buf, v_bi_buf = v_bi_buf,
        cos_sin_buf = cos_sin_buf, positions_buf = positions_buf,
        slot_map_buf = slot_map_buf,
        kv_cache_k = kv_cache_k, kv_cache_v = kv_cache_v,
        maybe_bias_params = maybe_bias_params,
        attn_out_buf = attn_out_buf,
        seq_used_k_buf = seq_used_k_buf,
        block_table_buf = block_table_buf,
        attn_out_idx = attn_out_idx,
        seq_used_k_idx = seq_used_k_idx,
        block_table_idx = block_table_idx,
        counter_idx = counter_idx,
        x_norm_name = x_norm_name, qmv_smem_name = qmv_smem_name,
        addrms_body = addrms_body,
        qmv_body = qmv_body,
        rope_body = rope_body,
        barrier_body = barrier_body,
        attn_body = attn_body,
        threads_per_tg_lit = threads_per_tg,
        num_simdgroups_lit = num_simdgroups,
        hidden_lit = consts.hidden,
        num_q_lit = consts.num_q_heads,
        num_kv_lit = consts.num_kv_heads,
        head_dim_lit = consts.head_dim,
        rot_dim_lit = consts.rot_dim,
        block_size_lit = consts.block_size,
        eps_lit = format_msl_float(consts.rms_norm_eps),
        attn_scale_lit = format_msl_float(attn_scale as f32),
    );

    let source = format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         //\n\
         // SYNTHESIZED PHASE-2c PERSISTENT KERNEL — generated by\n\
         // ferrite-fusion-synth::synthesize_pre_attn_attn_chunk_persistent.\n\
         // Pre-attn (AddRmsNorm + QKV qmv + RopeAppend) + in-kernel BN=8\n\
         // decode attention, separated by a cross-TG ticket-lock barrier.\n\
         // Do not hand-edit.\n\n\
         #include <metal_stdlib>\n\
         #include <metal_atomic>\n\
         using namespace metal;\n\n\
         // === inlined metal_kittens.h ===\n\
         {mk_header}\n\
         // === end inlined metal_kittens.h ===\n\n\
         {source_tail}",
        mk_header = mk_header,
        source_tail = source_tail,
    );

    SynthesizedKernel {
        symbol,
        source,
        backend: SynthesisBackend::Metal,
    }
}

// ───────────────────────────────────────────────────────────────────
// Persistent megakernel synthesis (multi-phase, cross-TG barrier).
// ───────────────────────────────────────────────────────────────────
//
// Emits the MSL body fragment of `attention_via_cache_v2_<dtype>_bn8_*`
// from `ferrite-metal-kernels/shaders/attention.metal`, adapted to
// reference kernel-scope variables produced by `pre_attn_persistent_*`
// instead of taking them as kernel arguments. Spliceable into a
// persistent kernel's attention phase between cross-TG barriers.
//
// Kernel-scope variables this body expects (all declared by the
// caller's persistent-kernel prologue / pre-attn phase):
//   __t            : token index (uint, =tg_pos.x; for decode also =seq_idx)
//   __head         : head index (uint, =tg_pos.y; covers Q+K+V band)
//   __simd_gid     : simdgroup_index_in_threadgroup (uint, =[0..BN))
//   __simd_lid     : thread_index_in_simdgroup       (uint, =[0..32))
//
// Compile-time constants the caller must declare (`constant constexpr`
// at file scope or `#define` above the kernel):
//   HEAD_DIM, NUM_Q, NUM_KV, BLOCK_SIZE  (already in pre-attn preamble)
//   ATTN_SCALE_BAKED                     (NEW: 1/sqrt(HEAD_DIM))
//   MAX_BLOCKS_PER_SEQ                   (NEW: function_constant(1) at
//                                         pipeline build; runtime
//                                         config-derived)
//
// Buffer names this body references (caller's signature must bind):
//   `q_buf`       — Q tensor [tokens, NUM_Q, HEAD_DIM]   (READ)
//   `out_buf`     — attention output [tokens, NUM_Q, HEAD_DIM] (WRITE)
//   `seq_used_k_buf` — uint[batch], total cached K
//   `block_table_buf` — uint[batch, MAX_BLOCKS_PER_SEQ]
//   `k_cache_buf`, `v_cache_buf` — paged caches [num_blocks, NUM_KV,
//                                  BLOCK_SIZE, HEAD_DIM]
//
// Threadgroup arrays this body declares inline (kept local to the body
// so the same caller can splice multiple attention phases without name
// collisions): `__attn_partials`, `__attn_max`, `__attn_sum`.
//
// The body wraps its work in `if (__head < NUM_Q) { ... }` so heads
// dispatched for pre-attn's K/V projections sit idle during attention
// (they still bump the surrounding barrier counter because the guard
// is *inside* the phase, not around it).
//
// Constraint: HEAD_DIM must be a multiple of 32 (qk_per_thread =
// HEAD_DIM/32). Llama-3.2/Qwen/Mistral/Phi all satisfy.
pub fn bn8_attention_body_msl(
    t_act: &str,
    q_buf: &str,
    out_buf: &str,
    seq_used_k_buf: &str,
    block_table_buf: &str,
    k_cache_buf: &str,
    v_cache_buf: &str,
) -> String {
    // ATTN_BN8_MAX_HEAD_DIM matches `attention.metal`'s define; sized to
    // the largest HEAD_DIM the body supports (256 → 8KiB tg_partials).
    const MAX_HEAD_DIM: u32 = 256;
    format!(
        r#"
    // ── atom: BN=8 paged-cache decode attention (Phase 2c) ──
    // BN-agnostic combine; same algorithm as attention_via_cache_v2_*_bn8_specialized.
    // TG memory: BN*MAX_HD*4 + BN*4*2 = 8*256*4 + 64 ≈ 8.3 KiB.
    if (__head < NUM_Q) {{
        constexpr int __ATTN_BN     = 8;
        constexpr int __ATTN_BD     = 32;
        constexpr uint __ATTN_MAX_HEAD_DIM = {MAX_HEAD_DIM}u;
        typedef float __ATTN_U;

        const uint __attn_seq_idx     = __t;
        const uint __attn_q_head_idx  = __head;
        const uint __attn_qk_per_thread = HEAD_DIM / uint(__ATTN_BD);
        const uint __attn_group_ratio = NUM_Q / NUM_KV;
        const uint __attn_kv_head_idx = __attn_q_head_idx / __attn_group_ratio;
        const uint __attn_kv_len      = {seq_used_k}[__attn_seq_idx];

        const uint __attn_kv_blk_stride  = NUM_KV * BLOCK_SIZE * HEAD_DIM;
        const uint __attn_kv_head_stride = BLOCK_SIZE * HEAD_DIM;
        const uint __attn_kv_tok_stride  = HEAD_DIM;

        thread __ATTN_U __attn_q_reg[8];
        thread __ATTN_U __attn_o_reg[8];

        threadgroup __ATTN_U __attn_partials[__ATTN_BN * __ATTN_MAX_HEAD_DIM];
        threadgroup __ATTN_U __attn_max[__ATTN_BN];
        threadgroup __ATTN_U __attn_sum[__ATTN_BN];

        device const {t_act}* __attn_q_row = {q_buf}
            + (__attn_seq_idx * NUM_Q + __attn_q_head_idx) * HEAD_DIM;
        device       {t_act}* __attn_o_row = {out_buf}
            + (__attn_seq_idx * NUM_Q + __attn_q_head_idx) * HEAD_DIM;
        device const uint*   __attn_row_block_table = {block_table}
            + __attn_seq_idx * MAX_BLOCKS_PER_SEQ;

        for (uint __i = 0u; __i < __attn_qk_per_thread; ++__i) {{
            __attn_q_reg[__i] = __ATTN_U(ATTN_SCALE_BAKED)
                              * __ATTN_U(__attn_q_row[__simd_lid * __attn_qk_per_thread + __i]);
            __attn_o_reg[__i] = 0;
        }}

        __ATTN_U __attn_max_score     = -FLT_MAX;
        __ATTN_U __attn_sum_exp_score = 0;

        for (uint __i = __simd_gid; __i < __attn_kv_len; __i += uint(__ATTN_BN)) {{
            const uint __attn_logical_block  = __i / BLOCK_SIZE;
            const uint __attn_physical_block = __attn_row_block_table[__attn_logical_block];
            const uint __attn_token_in_block = __i - __attn_logical_block * BLOCK_SIZE;
            device const {t_act}* __attn_k_ptr =
                {k_cache}
                + __attn_physical_block * __attn_kv_blk_stride
                + __attn_kv_head_idx    * __attn_kv_head_stride
                + __attn_token_in_block * __attn_kv_tok_stride
                + __simd_lid * __attn_qk_per_thread;
            device const {t_act}* __attn_v_ptr =
                {v_cache}
                + __attn_physical_block * __attn_kv_blk_stride
                + __attn_kv_head_idx    * __attn_kv_head_stride
                + __attn_token_in_block * __attn_kv_tok_stride
                + __simd_lid * __attn_qk_per_thread;

            __ATTN_U __attn_score = 0;
            for (uint __j = 0u; __j < __attn_qk_per_thread; ++__j) {{
                __attn_score += __attn_q_reg[__j] * __ATTN_U(__attn_k_ptr[__j]);
            }}
            __attn_score = simd_sum(__attn_score);

            __ATTN_U __attn_new_max  = max(__attn_max_score, __attn_score);
            __ATTN_U __attn_factor   = metal::fast::exp(__attn_max_score - __attn_new_max);
            __ATTN_U __attn_exp_score = metal::fast::exp(__attn_score - __attn_new_max);
            __attn_max_score      = __attn_new_max;
            __attn_sum_exp_score  = __attn_sum_exp_score * __attn_factor + __attn_exp_score;

            for (uint __j = 0u; __j < __attn_qk_per_thread; ++__j) {{
                __attn_o_reg[__j] = __attn_o_reg[__j] * __attn_factor
                                  + __attn_exp_score * __ATTN_U(__attn_v_ptr[__j]);
            }}
        }}

        for (uint __j = 0u; __j < __attn_qk_per_thread; ++__j) {{
            __attn_partials[__simd_gid * __ATTN_MAX_HEAD_DIM
                            + __simd_lid * __attn_qk_per_thread + __j] = __attn_o_reg[__j];
        }}
        if (__simd_lid == 0u) {{
            __attn_max[__simd_gid] = __attn_max_score;
            __attn_sum[__simd_gid] = __attn_sum_exp_score;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        __ATTN_U __attn_global_max = -FLT_MAX;
        for (uint __g = 0u; __g < uint(__ATTN_BN); ++__g) {{
            __attn_global_max = max(__attn_global_max, __attn_max[__g]);
        }}
        __ATTN_U __attn_global_sum = 0;
        __ATTN_U __attn_factors[__ATTN_BN];
        for (uint __g = 0u; __g < uint(__ATTN_BN); ++__g) {{
            __attn_factors[__g] = metal::fast::exp(__attn_max[__g] - __attn_global_max);
            __attn_global_sum  += __attn_sum[__g] * __attn_factors[__g];
        }}

        thread __ATTN_U __attn_final_o[8];
        for (uint __j = 0u; __j < __attn_qk_per_thread; ++__j) __attn_final_o[__j] = 0;
        for (uint __g = 0u; __g < uint(__ATTN_BN); ++__g) {{
            const __ATTN_U __attn_f = __attn_factors[__g];
            for (uint __j = 0u; __j < __attn_qk_per_thread; ++__j) {{
                __attn_final_o[__j] += __attn_partials[__g * __ATTN_MAX_HEAD_DIM
                                                       + __simd_lid * __attn_qk_per_thread + __j]
                                        * __attn_f;
            }}
        }}
        if (__attn_global_sum != 0) {{
            for (uint __j = 0u; __j < __attn_qk_per_thread; ++__j) __attn_final_o[__j] /= __attn_global_sum;
        }}

        if (__simd_gid == 0u) {{
            device {t_act}* __attn_o_ptr = __attn_o_row + __simd_lid * __attn_qk_per_thread;
            for (uint __j = 0u; __j < __attn_qk_per_thread; ++__j) {{
                __attn_o_ptr[__j] = {t_act}(__attn_final_o[__j]);
            }}
        }}
    }}
"#,
        t_act = t_act,
        q_buf = q_buf,
        out_buf = out_buf,
        seq_used_k = seq_used_k_buf,
        block_table = block_table_buf,
        k_cache = k_cache_buf,
        v_cache = v_cache_buf,
        MAX_HEAD_DIM = MAX_HEAD_DIM,
    )
}

// ───────────────────────────────────────────────────────────────────
// Persistent megakernel synthesis (multi-phase, cross-TG barrier).
//
// Background: the synth chunks above (synthesize_pre_attn_chunk etc.)
// each emit ONE kernel containing ONE fused phase region. Today's
// worker dispatches each of these chunks as a separate Metal kernel
// boundary — N dispatches per token, with each dispatch boundary
// costing ~1 ms of per-dispatch GPU command-processor overhead on M4
// (measured in `ferrite-metal-cost-sweep::persistent_vs_dispatched`).
//
// The persistent megakernel collapses all phase chunks into ONE
// kernel dispatch per token, with cross-TG ticket-lock barriers
// between phases instead of kernel boundaries. Measured win: 11×
// faster on M4 at production resource pressure (1024 threads/TG,
// 16 TGs, heterogeneous phase bodies — see
// `project_metal_persistent_megakernel_feasibility.md`).
//
// This MVP scaffolding emits a persistent kernel from a sequence of
// `PersistentPhase` descriptors. Each phase is currently just an MSL
// body fragment; future work threads atom DAG → phase body emission
// + automatic phase-boundary detection where TG decomposition has
// to change. The runtime atomic-counter target uses a strictly-
// increasing schedule so a single device counter serves all phases.
// ───────────────────────────────────────────────────────────────────

/// One phase of a persistent megakernel. Body is an MSL fragment
/// emitted into the kernel scope between cross-TG barriers. Callers
/// must use only kernel-scope variable names (no shared local
/// state across phase fragment boundaries beyond what's declared at
/// kernel scope above the per-phase bodies).
#[derive(Clone, Debug)]
pub struct PersistentPhase {
    /// Diagnostic name — appears in a comment in the emitted MSL,
    /// and the symbol name when phase count is small.
    pub name: String,
    /// MSL body fragment. Has access to:
    /// - `uint tg_id`            — `[[threadgroup_position_in_grid]]`
    /// - `uint tid`              — `[[thread_position_in_threadgroup]]`
    /// - `constant uint num_tgs` — TG count (passed via function_constant)
    /// - any buffer parameters declared in the kernel signature
    pub body: String,
}

impl PersistentPhase {
    /// Build a phase from a raw MSL body string. Use when the body
    /// is hand-written or pre-computed.
    pub fn from_body(name: impl Into<String>, body: impl Into<String>) -> Self {
        Self { name: name.into(), body: body.into() }
    }

    /// Build a phase by calling an `Atom`'s `emit_metal_body` on the
    /// supplied `AtomCtx`. The caller is responsible for providing
    /// kernel-scope variable declarations (via the `prologue` of
    /// `synthesize_persistent_chunk`) that the atom body references —
    /// e.g. `__t`, `__hidden`, `__head`, `__head_dim`, and any TG-
    /// memory arrays the atom's output channels expect.
    ///
    /// Returns `None` if the atom doesn't implement Metal emission.
    pub fn from_atom_metal(
        name: impl Into<String>,
        atom: &dyn crate::atom::Atom,
        ctx: &crate::atom::AtomCtx,
    ) -> Option<Self> {
        atom.emit_metal_body(ctx).map(|body| Self { name: name.into(), body })
    }
}

/// Standard pre-attention kernel-scope prologue. Emits the local
/// variable declarations atom bodies in `atom_lib` expect to find at
/// kernel scope: thread/TG position derivatives (`__t`, `__head`,
/// `__tid`), per-shape constants (`__hidden`, `__head_dim`, etc.),
/// the early-exit guard, and the standard TG-memory allocations
/// (`x_norm`, `__scratch`, `qmv_smem`).
///
/// Inputs are MSL constant names (typically baked as `constexpr` in
/// the kernel source above the prologue). Constants must be visible
/// at the point the prologue is spliced in.
///
/// Used by the persistent-megakernel codegen path so atoms that work
/// in the existing `synthesize_pre_attn_chunk` body can also run in
/// a persistent kernel — same prologue, different envelope.
pub fn pre_attn_kernel_scope_prologue(
    t_act: &str,
    hidden_const: &str,
    head_dim_const: &str,
    num_q_const: &str,
    num_kv_const: &str,
    rot_dim_const: &str,
    block_size_const: &str,
    eps_const: &str,
    m_const: &str,
    x_norm_name: &str,
    qmv_smem_name: &str,
    hidden_max: u32,
    head_dim_max: u32,
) -> String {
    format!(
        r#"
    const uint __t                = __tg_pos.x;
    const uint __head             = __tg_pos.y;
    const uint __tid              = __tid_pos.x;
    const uint __head_dim         = {head_dim};
    const uint __hidden           = {hidden};
    const uint __num_q            = {num_q};
    const uint __num_kv           = {num_kv};
    const uint __rot_dim          = {rot_dim};
    const uint __half_dim         = __rot_dim / 2;
    const uint __block_sz         = {block_size};
    const uint __num_heads_total  = __num_q + 2u * __num_kv;
    const uint __threads_per_tg   = MK_SIMD_SIZE * __head_dim / MK_ROWS_PER_SIMDGROUP;
    const uint __num_simdgroups   = __head_dim / MK_ROWS_PER_SIMDGROUP;
    const float __eps             = {eps};

    if (__t >= {m} || __head >= __num_heads_total) return;

    // `constant` address-space qualifier is illegal on function-scope
    // automatic variables — drop it and use plain `constexpr`. The
    // values are still compile-time constants (the array sizes below
    // resolve at MSL-compile time).
    constexpr uint __HIDDEN_MAX   = {hidden_max}u;
    constexpr uint __HEAD_DIM_MAX = {head_dim_max}u;
    constexpr uint __SCRATCH_MAX  = __HEAD_DIM_MAX / MK_ROWS_PER_SIMDGROUP;

    threadgroup {t_act} {x_norm}[__HIDDEN_MAX];
    threadgroup float   __scratch  [__SCRATCH_MAX];
    threadgroup float   {qmv_smem}[__HEAD_DIM_MAX];
"#,
        t_act = t_act,
        hidden = hidden_const,
        head_dim = head_dim_const,
        num_q = num_q_const,
        num_kv = num_kv_const,
        rot_dim = rot_dim_const,
        block_size = block_size_const,
        eps = eps_const,
        m = m_const,
        hidden_max = hidden_max,
        head_dim_max = head_dim_max,
        x_norm = x_norm_name,
        qmv_smem = qmv_smem_name,
    )
}

/// MSL helper fragment that performs a cross-TG ticket-lock barrier
/// at `phase_idx`. TG-leader threads (`tid == 0`) atomically bump
/// the device counter and spin until it reaches the target value
/// `num_tgs * (phase_idx + 1)`, after which a TG-level barrier
/// propagates the device-memory acquire to non-leader threads.
///
/// Apple Metal supports only `memory_order_relaxed` on atomics;
/// the trailing `threadgroup_barrier(mem_flags::mem_device)` is
/// what provides the actual cross-phase visibility ordering.
fn cross_tg_barrier_msl(phase_idx: usize, counter_name: &str) -> String {
    // Apple Metal only supports memory_order_relaxed atomics, so we
    // rely on `threadgroup_barrier(mem_flags::mem_device)` for cross-
    // phase visibility instead of atomic ordering. The PRE-barrier
    // flushes this TG's phase-N writes to L2 before signalling
    // arrival; the POST-barrier ensures phase-(N+1) reads happen
    // after all TGs have signalled.
    //
    // Without the pre-barrier the atomic increment can be observed
    // by another TG before that TG's phase-N device writes are
    // visible, and downstream reads of those writes return stale
    // (often zero) data — verified by the
    // `synth_persistent_e2e_test` end-to-end check.
    format!(
        r#"
    // ── cross-TG barrier (phase {phase_idx}) ──
    // Flush this TG's phase-{phase_idx} device writes to L2 before
    // signalling arrival.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {{
        atomic_fetch_add_explicit({counter_name}, 1u, memory_order_relaxed);
        uint __target = num_tgs * {target}u;
        while (atomic_load_explicit({counter_name}, memory_order_relaxed) < __target) {{
            // spin
        }}
    }}
    // Acquire-side: ensure phase-{next_phase} reads see all other
    // TGs' phase-{phase_idx} device writes.
    threadgroup_barrier(mem_flags::mem_device);
"#,
        phase_idx = phase_idx,
        next_phase = phase_idx + 1,
        counter_name = counter_name,
        target = phase_idx + 1,
    )
}

/// Synthesize a persistent megakernel containing multiple phase
/// bodies separated by cross-TG ticket-lock barriers.
///
/// The emitted kernel signature is (MSL):
/// ```text
/// [[kernel, max_total_threads_per_threadgroup(N)]]
/// void <symbol>(
///     device atomic_uint* __barrier_counter [[buffer(0)]],
///     constant uint& num_tgs                [[buffer(1)]],
///     /* per-phase buffer params at indices 2..N (caller provides
///        the kernel-scope variable declarations via the prologue) */
///     uint tg_id [[threadgroup_position_in_grid]],
///     uint tid   [[thread_position_in_threadgroup]])
/// { ... }
/// ```
///
/// `extra_signature` lets callers append per-phase buffer parameters
/// to the signature (one comma-prefixed `,\n    <type> <name> [[buffer(K)]]`
/// per parameter). `prologue` is emitted at the top of the kernel
/// body (before phase 0) — typically `threadgroup` allocations and
/// derived constants.
///
/// This is the MVP scaffolding; future work will derive the signature,
/// prologue, and phase bodies from an atom DAG walk rather than
/// requiring the caller to pre-format them.
pub fn synthesize_persistent_chunk(
    backend: SynthesisBackend,
    symbol: &str,
    threads_per_tg: u32,
    extra_signature: &str,
    prologue: &str,
    phases: &[PersistentPhase],
) -> SynthesizedKernel {
    synthesize_persistent_chunk_with_preamble(
        backend, symbol, threads_per_tg, extra_signature, "", prologue, phases,
    )
}

/// As `synthesize_persistent_chunk`, but accepts a `preamble` that's
/// emitted between the `#include`s and the kernel function. Use to
/// bake model-shape constants as `constexpr` at file scope so they
/// resolve to literal values inside the kernel prologue + atom bodies.
pub fn synthesize_persistent_chunk_with_preamble(
    backend: SynthesisBackend,
    symbol: &str,
    threads_per_tg: u32,
    extra_signature: &str,
    preamble: &str,
    prologue: &str,
    phases: &[PersistentPhase],
) -> SynthesizedKernel {
    assert_eq!(
        backend,
        SynthesisBackend::Metal,
        "MVP only emits Metal",
    );
    assert!(!phases.is_empty(), "synthesize_persistent_chunk: empty phase list");

    let counter_name = "__barrier_counter";
    let mut body = String::new();
    body.push_str(prologue);
    body.push('\n');

    for (i, phase) in phases.iter().enumerate() {
        body.push_str(&format!("\n    // ── phase {}: {} ──\n", i, phase.name));
        body.push_str(&phase.body);
        // Cross-TG barrier between phases. The barrier after the
        // last phase is also emitted so a downstream phase / kernel
        // observes the final phase's writes; it costs ~5 µs and
        // makes the post-condition uniform.
        body.push_str(&cross_tg_barrier_msl(i, counter_name));
    }

    let source = format!(
        r#"// SPDX-License-Identifier: Apache-2.0
// SYNTHESIZED PERSISTENT MEGAKERNEL — do not hand-edit.
// Generated by ferrite-fusion-synth::synthesize_persistent_chunk.
//
// Phases ({n_phases}):
{phase_comments}

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

{preamble}

[[kernel, max_total_threads_per_threadgroup({threads_per_tg})]]
void {symbol}(
    device atomic_uint* {counter}   [[buffer(0)]],
    constant uint&      num_tgs     [[buffer(1)]]{extra_signature},
    uint3 __tg_pos    [[threadgroup_position_in_grid]],
    uint3 __tid_pos   [[thread_position_in_threadgroup]],
    uint  __simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  __simd_lid  [[thread_index_in_simdgroup]])
{{
    // Convenience scalars for phase bodies that don't need the full
    // 3D thread/TG position. Atoms in `atom_lib` reference the 3D
    // versions directly; simple bodies can use these.
    const uint tg_id = __tg_pos.x;
    const uint tid   = __tid_pos.x;
{body}
}}
"#,
        n_phases = phases.len(),
        phase_comments = phases
            .iter()
            .enumerate()
            .map(|(i, p)| format!("//   [{}] {}", i, p.name))
            .collect::<Vec<_>>()
            .join("\n"),
        threads_per_tg = threads_per_tg,
        symbol = symbol,
        counter = counter_name,
        extra_signature = extra_signature,
        preamble = preamble,
        body = body,
    );

    SynthesizedKernel {
        symbol: symbol.to_string(),
        source,
        backend: SynthesisBackend::Metal,
    }
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
            intermediate:  8192,
            m:             1,
            group_size:    64,
            rms_norm_eps:  1e-5,
            has_linear_bias: false,
        }
    }

    #[test]
    fn synthesize_persistent_chunk_emits_well_formed_msl() {
        // Three trivial phases: each writes a marker to a per-phase
        // output buffer. Cross-TG barriers separate them. The test
        // validates the structural shape of the emitted MSL — that
        // the kernel signature, phase comments, barriers, and bodies
        // all appear at the expected places.
        let phases = vec![
            PersistentPhase {
                name: "phase_a".to_string(),
                body: "    if (tid == 0u) out_a[tg_id] = 1.0f;\n".to_string(),
            },
            PersistentPhase {
                name: "phase_b".to_string(),
                body: "    if (tid == 0u) out_b[tg_id] = 2.0f;\n".to_string(),
            },
            PersistentPhase {
                name: "phase_c".to_string(),
                body: "    if (tid == 0u) out_c[tg_id] = 3.0f;\n".to_string(),
            },
        ];
        let extra_sig = ",\n    device float* out_a [[buffer(2)]],\n    device float* out_b [[buffer(3)]],\n    device float* out_c [[buffer(4)]]";
        let kernel = synthesize_persistent_chunk(
            SynthesisBackend::Metal,
            "test_persistent_three_phases",
            256,
            extra_sig,
            "",
            &phases,
        );
        assert_eq!(kernel.symbol, "test_persistent_three_phases");
        assert!(kernel.source.contains("[[kernel, max_total_threads_per_threadgroup(256)]]"));
        assert!(kernel.source.contains("test_persistent_three_phases"));
        // All three phase names appear (in both header comment block
        // and per-phase markers).
        assert!(kernel.source.matches("phase_a").count() >= 2);
        assert!(kernel.source.matches("phase_b").count() >= 2);
        assert!(kernel.source.matches("phase_c").count() >= 2);
        // Three cross-TG barriers (one after each phase).
        assert_eq!(kernel.source.matches("cross-TG barrier (phase ").count(), 3);
        // Counter targets are strictly increasing (1, 2, 3).
        assert!(kernel.source.contains("num_tgs * 1u"));
        assert!(kernel.source.contains("num_tgs * 2u"));
        assert!(kernel.source.contains("num_tgs * 3u"));
        // Atomic spin pattern is intact.
        assert!(kernel.source.contains("atomic_fetch_add_explicit"));
        assert!(kernel.source.contains("memory_order_relaxed"));
        assert!(kernel.source.contains("threadgroup_barrier(mem_flags::mem_device)"));
        // Phase bodies preserved verbatim.
        assert!(kernel.source.contains("out_a[tg_id] = 1.0f"));
        assert!(kernel.source.contains("out_b[tg_id] = 2.0f"));
        assert!(kernel.source.contains("out_c[tg_id] = 3.0f"));
    }

    #[test]
    fn persistent_phase_from_atom_metal_threads_through() {
        // Wire AddRmsNormAtom through PersistentPhase::from_atom_metal
        // and synthesize_persistent_chunk. Validates that the atom
        // composition path emits MSL containing the atom's body
        // verbatim with channel-name substitutions applied.
        let atom = AddRmsNormAtom::default();
        let in_names = vec![
            "__residual_io".to_string(),
            "__delta".to_string(),
            "__rms_weight".to_string(),
        ];
        let out_names = vec!["__x_norm".to_string()];
        let consts: Vec<(&'static str, AtomConstantValue)> = vec![
            ("HIDDEN", AtomConstantValue::Uint(2048)),
            ("NUM_Q", AtomConstantValue::Uint(32)),
            ("NUM_KV", AtomConstantValue::Uint(8)),
            ("HEAD_DIM", AtomConstantValue::Uint(64)),
            ("EPS", AtomConstantValue::Float(1e-5)),
        ];
        let ctx = AtomCtx {
            bound_inputs: &in_names,
            bound_outputs: &out_names,
            constants: &consts,
            t_act: "bfloat",
            t_scale: "half",
        };
        let phase = PersistentPhase::from_atom_metal("add_rms_norm", &atom, &ctx)
            .expect("AddRmsNormAtom should emit Metal body");
        assert!(phase.body.contains("AddRmsNorm"),
            "atom body should contain its own comment marker");
        assert!(phase.body.contains("__residual_io"),
            "channel substitution should write the bound name into the body");

        // Wire through the persistent-chunk synthesizer. Symbol naming +
        // structural shape should match what the standalone tests check.
        let kernel = synthesize_persistent_chunk(
            SynthesisBackend::Metal,
            "test_atom_persistent_chunk",
            256,
            ",\n    device bfloat* __residual_io [[buffer(2)]]",
            "",
            std::slice::from_ref(&phase),
        );
        assert!(kernel.source.contains("add_rms_norm"));
        assert!(kernel.source.contains("test_atom_persistent_chunk"));
        // One cross-TG barrier emitted after the single atom phase.
        assert_eq!(kernel.source.matches("cross-TG barrier (phase ").count(), 1);
        // Atom body's marker still present after composition.
        assert!(kernel.source.contains("AddRmsNorm"));
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

    #[test]
    fn synthesize_pre_attn_chunk_persistent_structural() {
        // Persistent-envelope variant emits the same atom-body content
        // plus the persistent-kernel scaffolding (counter binding,
        // num_tgs, atomic spin barrier). Validate symbol naming,
        // envelope markers, and that the atom bodies survive.
        let consts = llama_3_2_3b_constants();
        let kernel = synthesize_pre_attn_chunk_persistent(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &consts,
        );
        assert_eq!(kernel.backend, SynthesisBackend::Metal);
        assert!(kernel.symbol.starts_with("synth_pre_attn_persistent_"));
        assert!(kernel.symbol.contains("bfloat"));
        assert!(kernel.symbol.contains("gs64"));
        // Persistent envelope markers — counter binding is plumbed
        // through the signature even though the single-phase variant
        // doesn't currently use it (Phase 2b dropped the trailing
        // barrier; the binding stays for forward-compat with
        // multi-phase fusion like `synthesize_pre_attn_attn_*`).
        assert!(kernel.source.contains("__barrier_counter"));
        assert!(kernel.source.contains("max_total_threads_per_threadgroup"));
        // Atom bodies survive.
        assert!(kernel.source.contains("mk_tg_rmsnorm_scale"));
        assert!(kernel.source.contains("mk_qdot"));
        assert!(kernel.source.contains("mk_rope_pair"));
        // Buffer indices match the non-persistent kernel (strict
        // signature superset). q_out at 0, kv_cache_v at 17, counter
        // appended at 18.
        assert!(kernel.source.contains("__q_out      [[buffer(0)]]"));
        assert!(kernel.source.contains("__kv_cache_v     [[buffer(17)]]"));
        assert!(kernel.source.contains("__barrier_counter [[buffer(18)]]"));
        // M function constant still wired.
        assert!(kernel.source.contains("constant uint  M [[function_constant(0)]]"));
        // num_tgs derived from `[[threadgroups_per_grid]]`, not from
        // `M * num_heads_total` — the runtime m_scaling shrinks the
        // X-axis to num_tokens ≤ M, so `M * num_heads_total` would
        // over-count and a multi-phase ticket-lock would spin forever.
        assert!(kernel.source.contains("__persistent_num_tgs = __tgs_per_grid.x * __tgs_per_grid.y"));
    }

    fn llama_3_2_1b_constants() -> ChunkConstants {
        // Llama-3.2-1B-Instruct-4bit: hidden=2048, q_heads=32, kv_heads=8,
        // head_dim=64, rot_dim=64, group_size=64, block_size=16.
        ChunkConstants {
            hidden:        2048,
            num_q_heads:   32,
            num_kv_heads:  8,
            head_dim:      64,
            rot_dim:       64,
            block_size:    16,
            intermediate:  8192,
            m:             1,
            group_size:    64,
            rms_norm_eps:  1e-5,
            has_linear_bias: false,
        }
    }

    #[test]
    fn synthesize_pre_attn_attn_chunk_persistent_structural() {
        // Phase 2c structural test: pre-attn + cross-TG barrier +
        // in-kernel BN=8 attention all appear in one synthesized kernel
        // with the correct buffer signature and barrier scaffolding.
        let consts = llama_3_2_1b_constants();
        let kernel = synthesize_pre_attn_attn_chunk_persistent(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &consts,
            256,
        );
        assert_eq!(kernel.backend, SynthesisBackend::Metal);
        assert!(kernel.symbol.starts_with("synth_pre_attn_attn_persistent_"));
        assert!(kernel.symbol.contains("bfloat"));
        assert!(kernel.symbol.contains("gs64"));

        // Pre-attn atom bodies survive verbatim.
        assert!(kernel.source.contains("mk_tg_rmsnorm_scale"));
        assert!(kernel.source.contains("mk_qdot"));
        assert!(kernel.source.contains("mk_rope_pair"));

        // Phase comments delimit the two phases + the barrier.
        assert!(kernel.source.contains("phase 0: pre-attn"));
        assert!(kernel.source.contains("phase 1: BN=8 paged-cache decode attention"));
        assert!(kernel.source.contains("cross-TG barrier: pre-attn"));

        // Cross-TG ticket-lock barrier scaffolding (phase index 0,
        // target = num_tgs * 1u).
        assert!(kernel.source.contains("atomic_fetch_add_explicit"));
        assert!(kernel.source.contains("num_tgs * 1u"));
        assert!(kernel.source.contains("threadgroup_barrier(mem_flags::mem_device)"));

        // Buffer signature: pre-attn 0..17, attention bindings 18..20,
        // barrier counter 21 (no-bias variant). The strict-superset
        // shape matches `synth_pre_attn_persistent_*` for 0..17.
        assert!(kernel.source.contains("__q_out      [[buffer(0)]]"));
        assert!(kernel.source.contains("__kv_cache_v     [[buffer(17)]]"));
        assert!(kernel.source.contains("__attn_out    [[buffer(18)]]"));
        assert!(kernel.source.contains("__seq_used_k [[buffer(19)]]"));
        assert!(kernel.source.contains("__block_table [[buffer(20)]]"));
        assert!(kernel.source.contains("__barrier_counter [[buffer(21)]]"));

        // Function constants: M (existing) + MAX_BLOCKS_PER_SEQ (NEW).
        assert!(kernel.source.contains("constant uint  M                  [[function_constant(0)]]"));
        assert!(kernel.source.contains(
            "constant uint  MAX_BLOCKS_PER_SEQ [[function_constant(1)]]"
        ));

        // Baked attention scale (1/sqrt(HEAD_DIM=64) = 0.125).
        assert!(kernel.source.contains("ATTN_SCALE_BAKED"));

        // Attention body uses the bn8 markers; head guard so K/V
        // heads sit idle during attention.
        assert!(kernel.source.contains("if (__head < NUM_Q)"));
        assert!(kernel.source.contains("__attn_partials"));
        assert!(kernel.source.contains("__attn_max"));
        assert!(kernel.source.contains("__attn_sum"));

        // Reads Q from the pre-attn output buffer, writes attn_out.
        assert!(kernel.source.contains("__attn_q_row = __q_out"));
        assert!(kernel.source.contains("__attn_o_row = __attn_out"));

        // num_tgs derived from `[[threadgroups_per_grid]]` (not from
        // M * num_heads_total which would over-count when m_scaling
        // shrinks the X-axis to num_tokens < M).
        assert!(kernel.source.contains("num_tgs = __tgs_per_grid.x * __tgs_per_grid.y"));

        // Critically: no early-return guard before the barrier. The
        // grid is exactly (num_tokens × num_heads_total) so every
        // dispatched TG is valid; an early-return would deadlock the
        // surviving TGs spinning on the ticket-lock.
        assert!(
            !kernel.source.contains("if (__t >= M || __head >= __num_heads_total) return;"),
            "Phase 2c must NOT emit the defensive early-return — it bypasses the cross-TG barrier"
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn synthesize_pre_attn_attn_chunk_persistent_compiles_with_metal() {
        // Verify the Phase 2c synth source survives `xcrun metal -c`.
        // Skipped on non-macOS (no `xcrun`). The non-bias / bias / init
        // variants all share the same shape; testing the non-bias path
        // catches the common case + the new attention body shape.
        let consts = llama_3_2_1b_constants();
        let kernel = synthesize_pre_attn_attn_chunk_persistent(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &consts,
            256,
        );
        let bytes = crate::aot::aot_compile_metallib(&kernel.symbol, &kernel.source);
        assert!(
            !bytes.is_empty(),
            "Phase 2c metallib compile returned empty bytes — `xcrun metal` likely failed (set FERRITE_SYNTH_DUMP=1 + check /tmp/ferrite-synth-dump)",
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn synthesize_pre_attn_attn_chunk_persistent_compiles_3b_at_256_t_per_tg() {
        // Shape-decoupling guarantee: same `synth_pre_attn_attn_persistent_*`
        // family compiles at 256 threads/TG for HEAD_DIM=128 (Llama-3.2-3B)
        // via the atom-body pass loop (NUM_PASSES=2). Pre-decoupling this
        // would have required threads_per_tg=512 (which exceeds the
        // safe-residency cap on M4 when grid >= 60 TGs).
        let consts = llama_3_2_3b_constants();
        let kernel = synthesize_pre_attn_attn_chunk_persistent(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &consts,
            256,
        );
        // Threads-per-TG attribute is the chosen 256, NOT 4*HEAD_DIM=512.
        assert!(kernel.source.contains("[[kernel, max_total_threads_per_threadgroup(256u)]]"));
        // num_simdgroups baked at the chip-driven value, NOT HEAD_DIM/8.
        assert!(kernel.source.contains("__num_simdgroups   = 8u"));
        let bytes = crate::aot::aot_compile_metallib(&kernel.symbol, &kernel.source);
        assert!(
            !bytes.is_empty(),
            "3B-at-256 Phase 2c metallib compile failed — set FERRITE_SYNTH_DUMP=1 to dump the source",
        );
    }

    #[test]
    fn synthesize_pre_attn_attn_chunk_persistent_init_variant() {
        let consts = llama_3_2_1b_constants();
        let kernel = synthesize_pre_attn_attn_init_chunk_persistent(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &consts,
            256,
        );
        assert!(kernel.symbol.starts_with("synth_pre_attn_attn_init_persistent_"));
        // Init body has the "no residual add" comment marker from
        // AddRmsNormAtom { init: true }.
        assert!(kernel.source.contains("layer-0 / no residual add"));
    }

    #[test]
    fn synthesize_pre_attn_attn_chunk_persistent_with_bias_shifts_counter() {
        // Bias variant (Qwen2/2.5 QKV linear bias): extra 18/19/20
        // bindings push the new attention bindings to 21/22/23 and
        // the counter to 24.
        let consts = ChunkConstants { has_linear_bias: true, ..llama_3_2_1b_constants() };
        let kernel = synthesize_pre_attn_attn_chunk_persistent(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &consts,
            256,
        );
        assert!(kernel.symbol.ends_with("_bias"));
        assert!(kernel.source.contains("__q_linear_bias      [[buffer(18)]]"));
        assert!(kernel.source.contains("__v_linear_bias      [[buffer(20)]]"));
        assert!(kernel.source.contains("__attn_out    [[buffer(21)]]"));
        assert!(kernel.source.contains("__barrier_counter [[buffer(24)]]"));
    }

    #[test]
    fn synthesize_pre_attn_init_chunk_persistent_uses_init_symbol() {
        let consts = llama_3_2_3b_constants();
        let kernel = synthesize_pre_attn_init_chunk_persistent(
            SynthesisBackend::Metal,
            "bfloat",
            "half",
            &consts,
        );
        assert!(kernel.symbol.starts_with("synth_pre_attn_init_persistent_"));
        // Init body has the "no residual add" comment marker.
        assert!(kernel.source.contains("layer-0 / no residual add"));
    }
}
