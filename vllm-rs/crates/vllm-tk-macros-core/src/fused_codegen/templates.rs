// SPDX-License-Identifier: Apache-2.0
//! Askama template context structs.
//!
//! Each `#[derive(Template)]` struct corresponds to one template file under
//! `templates/fused/`. The fields become the variables the template can
//! reference (`{{ field_name }}`).

use askama::Template;

// ── Preamble ────────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/preamble.cu", escape = "none")]
pub struct PreambleCtx<'a> {
    pub mode_label: &'a str,
    pub cta_rows: usize,
    pub nl: usize,
    pub hd: usize,
    pub id: usize,
    pub hdm: usize,
    pub nah: usize,
    pub nkh: usize,
    pub num_warps: usize,
    pub gqa_ratio: usize,
    pub kv_page_size: usize,
    pub iters_per_page: usize,
    pub total_shmem: usize,
    pub kv_tile_bytes: usize,
    pub k_dim: usize,
    pub out_block: usize,
    pub rdpw: usize,
}

// ── RMSNorm ─────────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/rmsnorm.cu", escape = "none")]
pub struct RmsNormCtx<'a> {
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub output_global: &'a str,
    pub wgt_offset: usize,
    pub scratch_offset: usize,
}

// ── GEMM phase ──────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/gemm.cu", escape = "none")]
pub struct GemmCtx<'a> {
    pub phase_comment: &'a str,
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub num_k_iters: usize,
    pub num_col_tiles: usize,
    pub a_size: usize,
    pub b_size: usize,
    pub stage_size: usize,
    pub b_offset: usize,
    pub epilogue: String,
    pub cooperative: bool,
    pub col_batch: usize,
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
    pub hdm: usize,
    pub q_end: usize,
    pub k_start: usize,
    pub v_start: usize,
    pub kv_elems: usize,
}

// ── Attention ───────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/attention.cu", escape = "none")]
pub struct AttentionCtx {
    pub attn_passes: usize,
    pub nkh: usize,
    pub nah: usize,
    pub stage_sz: usize,
}

// ── Launch wrapper ──────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/launch_wrapper.cu", escape = "none")]
pub struct LaunchWrapperCtx<'a> {
    pub tensor_arg_helper: &'a str,
    pub launch_params: &'a str,
    pub globals_construction: &'a str,
    pub num_threads: usize,
}

// ── Top-level kernel ────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/kernel.cu", escape = "none")]
pub struct KernelCtx<'a> {
    pub preamble: &'a str,
    pub phases: &'a [String],
    pub launch_wrapper: &'a str,
    pub num_threads: usize,
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
    pub num_k_iters: usize,
    pub num_col_tiles: usize,
    pub a_size: usize,
    pub b_size: usize,
    pub stage_size: usize,
    pub b_offset: usize,
    pub epilogue: String,
    pub cooperative: bool,
    pub col_batch: usize,
    pub num_stages: usize,
}

#[derive(Template)]
#[template(path = "fused/rmsnorm_mcta.cu", escape = "none")]
pub struct RmsNormMctaCtx<'a> {
    pub input_global: &'a str,
    pub weight_global: &'a str,
    pub output_global: &'a str,
    pub wgt_offset: usize,
    pub scratch_offset: usize,
}

#[derive(Template)]
#[template(path = "fused/rope_kv_append_mcta.cu", escape = "none")]
pub struct RopeKvAppendMctaCtx {
    pub hdm: usize,
    pub q_end: usize,
    pub k_start: usize,
    pub v_start: usize,
    pub kv_elems: usize,
}

#[derive(Template)]
#[template(path = "fused/attention_mcta.cu", escape = "none")]
pub struct AttentionMctaCtx {
    pub nkh: usize,
    pub nah: usize,
    pub stage_sz: usize,
}

#[derive(Template)]
#[template(path = "fused/gemm_gate_up_mcta.cu", escape = "none")]
pub struct GemmGateUpMctaCtx<'a> {
    pub phase_comment: &'a str,
    pub input_global: &'a str,
    pub gate_weight_global: &'a str,
    pub up_weight_global: &'a str,
    pub output_global: &'a str,
    pub num_k_iters: usize,
    pub num_col_tiles: usize,
    pub a_size: usize,
    pub b_size: usize,
    pub stage_size: usize,
    pub b_offset: usize,
    pub cooperative: bool,
    pub num_stages: usize,
}

#[derive(Template)]
#[template(path = "fused/launch_wrapper_mcta.cu", escape = "none")]
pub struct LaunchWrapperMctaCtx<'a> {
    pub tensor_arg_helper: &'a str,
    pub launch_params: &'a str,
    pub globals_construction: &'a str,
    pub num_threads: usize,
    pub grid_size: usize,
    pub id_col_tiles: usize,
}

#[derive(Template)]
#[template(path = "fused/kernel_multi_cta.cu", escape = "none")]
pub struct KernelMctaCtx<'a> {
    pub preamble: &'a str,
    pub phases: &'a [String],
    pub launch_wrapper: &'a str,
    pub num_threads: usize,
    pub phase_names_str: &'a str,
    pub num_phases: usize,
    pub grid_size: usize,
}

// ── Decode templates ──────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "fused/decode_preamble.cu", escape = "none")]
pub struct DecodePreambleCtx {
    pub cta_rows: usize,
    pub padded_cta_rows: usize,
    pub nl: usize,
    pub hd: usize,
    pub id: usize,
    pub hdm: usize,
    pub nah: usize,
    pub nkh: usize,
    pub num_warps: usize,
    pub gqa_ratio: usize,
    pub kv_page_size: usize,
    pub iters_per_page: usize,
    pub peak_shmem: usize,
    pub kv_tile_bytes: usize,
    pub k_dim: usize,
    pub out_block: usize,
    pub hidden_shmem: usize,
    pub meta_shmem: usize,
    pub num_stages: usize,
}

#[derive(Template)]
#[template(path = "fused/decode_rmsnorm.cu", escape = "none")]
pub struct DecodeRmsNormCtx<'a> {
    pub weight_global: &'a str,
    pub input_offset: usize,
    pub output_offset: usize,
    pub wgt_shmem_offset: usize,
    pub scratch_offset: usize,
    pub rdpw: usize,
}

#[derive(Template)]
#[template(path = "fused/decode_kernel.cu", escape = "none")]
pub struct DecodeKernelCtx<'a> {
    pub preamble: &'a str,
    pub phases: &'a [String],
    pub launch_wrapper: &'a str,
    pub num_threads: usize,
    pub phase_names_str: &'a str,
    pub num_phases: usize,
    pub cta_rows: usize,
    pub padded_cta_rows: usize,
    pub hidden_shmem: usize,
    pub meta_shmem: usize,
}

#[derive(Template)]
#[template(path = "fused/decode_launch_wrapper.cu", escape = "none")]
pub struct DecodeLaunchWrapperCtx<'a> {
    pub tensor_arg_helper: &'a str,
    pub launch_params: &'a str,
    pub globals_construction: &'a str,
    pub num_threads: usize,
}
