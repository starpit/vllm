// SPDX-License-Identifier: Apache-2.0
//! Askama template context structs.
//!
//! Each `#[derive(Template)]` struct corresponds to one template file under
//! `templates/fused/`. The fields become the variables the template can
//! reference (`{{ field_name }}`).

use askama::Template;

use super::units::{Bytes, Count, Dim, Iters, Tiles};

// ── Preamble (split for polyalgorithm) ──────────────────────────────────

#[derive(Template)]
#[template(path = "fused/preamble_header.cu", escape = "none")]
pub struct PreambleHeaderCtx<'a> {
    pub mode_label: &'a str,
    pub nl: Count,
    pub hd: Dim,
    pub id: Dim,
    pub hdm: Dim,
    pub nah: Count,
    pub nkh: Count,
}

#[derive(Template)]
#[template(path = "fused/preamble_constants.cu", escape = "none")]
pub struct PreambleConstantsCtx {
    pub cta_rows: Dim,
    pub num_warps: Count,
    pub gqa_ratio: Count,
    pub kv_page_size: Dim,
    pub iters_per_page: Iters,
    pub total_shmem: Bytes,
    pub kv_tile_bytes: Bytes,
    pub k_dim: Dim,
    pub out_block: Dim,
    pub rdpw: Count,
    pub hdm: Dim,
}

// ── Preamble (combined, used by non-polyalgorithm paths) ────────────────

#[derive(Template)]
#[template(path = "fused/preamble.cu", escape = "none")]
pub struct PreambleCtx<'a> {
    pub mode_label: &'a str,
    pub cta_rows: Dim,
    pub nl: Count,
    pub hd: Dim,
    pub id: Dim,
    pub hdm: Dim,
    pub nah: Count,
    pub nkh: Count,
    pub num_warps: Count,
    pub gqa_ratio: Count,
    pub kv_page_size: Dim,
    pub iters_per_page: Iters,
    pub total_shmem: Bytes,
    pub kv_tile_bytes: Bytes,
    pub k_dim: Dim,
    pub out_block: Dim,
    pub rdpw: Count,
}

// ── RMSNorm ─────────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/rmsnorm.cu", escape = "none")]
pub struct RmsNormCtx<'a> {
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub output_global: &'a str,
    pub wgt_offset: Bytes,
    pub scratch_offset: Bytes,
}

// ── GEMM phase ──────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/gemm.cu", escape = "none")]
pub struct GemmCtx<'a> {
    pub phase_comment: &'a str,
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub num_k_iters: Iters,
    pub num_col_tiles: Tiles,
    pub a_size: Bytes,
    pub b_size: Bytes,
    pub stage_size: Bytes,
    pub b_offset: Bytes,
    pub epilogue: String,
    pub cooperative: bool,
    pub col_batch: Tiles,
}

// ── Epilogue variants ───────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/epilogue_store.cu", escape = "none")]
pub struct EpilogueStoreCtx<'a> {
    pub output: &'a str,
    pub row_var: &'a str,
}

#[derive(Template)]
#[template(path = "fused/epilogue_residual.cu", escape = "none")]
pub struct EpilogueResidualCtx<'a> {
    pub residual: &'a str,
    pub row_var: &'a str,
}

#[derive(Template)]
#[template(path = "fused/epilogue_silu.cu", escape = "none")]
pub struct EpilogueSiluCtx<'a> {
    pub output: &'a str,
    pub row_var: &'a str,
}

#[derive(Template)]
#[template(path = "fused/epilogue_mulgate.cu", escape = "none")]
pub struct EpilogueMulGateCtx<'a> {
    pub gate_output: &'a str,
    pub row_var: &'a str,
}

// ── RoPE + KV append ────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/rope_kv_append.cu", escape = "none")]
pub struct RopeKvAppendCtx {
    pub hdm: Dim,
    pub q_end: Dim,
    pub k_start: Dim,
    pub v_start: Dim,
    pub kv_elems: Dim,
}

// ── Attention ───────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/attention.cu", escape = "none")]
pub struct AttentionCtx {
    pub attn_passes: Iters,
    pub nkh: Count,
    pub nah: Count,
    pub stage_sz: Bytes,
}

// ── Launch wrapper ──────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/launch_wrapper.cu", escape = "none")]
pub struct LaunchWrapperCtx<'a> {
    pub tensor_arg_helper: &'a str,
    pub launch_params: &'a str,
    pub globals_construction: &'a str,
    pub num_threads: Count,
}

// ── Top-level kernel ────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/kernel.cu", escape = "none")]
pub struct KernelCtx<'a> {
    pub preamble: &'a str,
    pub phases: &'a [String],
    pub launch_wrapper: &'a str,
    pub num_threads: Count,
    pub phase_names_str: &'a str,
    pub num_phases: usize,
}

// ── Multi-CTA templates ────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/gemm_mcta.cu", escape = "none")]
pub struct GemmMctaCtx<'a> {
    pub phase_comment: &'a str,
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub num_k_iters: Iters,
    pub num_col_tiles: Tiles,
    pub a_size: Bytes,
    pub b_size: Bytes,
    pub stage_size: Bytes,
    pub b_offset: Bytes,
    pub epilogue: String,
    pub cooperative: bool,
    pub col_batch: Tiles,
    pub num_stages: Count,
    pub per_warp_b: bool,
}

#[derive(Template)]
#[template(path = "fused/gemm_warpspec_mcta.cu", escape = "none")]
pub struct GemmWarpspecMctaCtx<'a> {
    pub phase_comment: &'a str,
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub num_k_iters: Iters,
    pub num_col_tiles: Tiles,
    pub a_size: Bytes,
    pub b_size: Bytes,
    pub stage_size: Bytes,
    pub b_offset: Bytes,
    pub epilogue: String,
    pub num_stages: Count,
}

#[derive(Template)]
#[template(path = "fused/rmsnorm_mcta.cu", escape = "none")]
pub struct RmsNormMctaCtx<'a> {
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub output_global: &'a str,
    pub wgt_offset: Bytes,
    pub scratch_offset: Bytes,
}

#[derive(Template)]
#[template(path = "fused/rope_kv_append_mcta.cu", escape = "none")]
pub struct RopeKvAppendMctaCtx {
    pub hdm: Dim,
    pub q_end: Dim,
    pub k_start: Dim,
    pub v_start: Dim,
    pub kv_elems: Dim,
}

#[derive(Template)]
#[template(path = "fused/attention_mcta.cu", escape = "none")]
pub struct AttentionMctaCtx {
    pub nkh: Count,
    pub nah: Count,
    pub stage_sz: Bytes,
}

#[derive(Template)]
#[template(path = "fused/gemm_gate_up_mcta.cu", escape = "none")]
pub struct GemmGateUpMctaCtx<'a> {
    pub phase_comment: &'a str,
    pub input_global: &'a str,
    pub gate_weight_global: &'a str,
    pub up_weight_global: &'a str,
    pub output_global: &'a str,
    pub num_k_iters: Iters,
    pub num_col_tiles: Tiles,
    pub a_size: Bytes,
    pub b_size: Bytes,
    pub stage_size: Bytes,
    pub b_offset: Bytes,
    pub cooperative: bool,
    pub num_stages: Count,
    pub per_warp_b: bool,
}

#[derive(Template)]
#[template(path = "fused/launch_wrapper_mcta.cu", escape = "none")]
pub struct LaunchWrapperMctaCtx<'a> {
    pub tensor_arg_helper: &'a str,
    pub launch_params: &'a str,
    pub globals_construction: &'a str,
    pub num_threads: Count,
    pub grid_size: Count,
    pub id_col_tiles: Tiles,
    pub kernel_suffix: &'a str,
}

#[derive(Template)]
#[template(path = "fused/launch_inner_mcta.cu", escape = "none")]
pub struct LaunchInnerMctaCtx<'a> {
    pub num_threads: Count,
    pub id_col_tiles: Tiles,
    pub kernel_suffix: &'a str,
}

#[derive(Template)]
#[template(path = "fused/polyalgorithm_dispatch.cu", escape = "none")]
pub struct PolyalgorithmDispatchCtx<'a> {
    pub tensor_arg_helper: &'a str,
    pub launch_params: &'a str,
    pub globals_construction: &'a str,
}

#[derive(Template)]
#[template(path = "fused/kernel_multi_cta.cu", escape = "none")]
pub struct KernelMctaCtx<'a> {
    pub preamble: &'a str,
    pub phases: &'a [String],
    pub launch_wrapper: &'a str,
    pub num_threads: Count,
    pub phase_names_str: &'a str,
    pub num_phases: usize,
    pub grid_size: Count,
    pub kernel_suffix: &'a str,
}
