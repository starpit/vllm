// SPDX-License-Identifier: Apache-2.0
//! Askama template context for the scheduled BSP megakernel.
//!
//! Currently a single template — `templates/scheduled/megakernel.cu` —
//! renders the entire `.cu` file: prelude (data tables + model dim
//! constants) followed by tile bodies + the megakernel + `extern "C"`
//! launchers. The body is still one big chunk of literal CUDA; the
//! next refactor step will split it into per-tile templates included
//! via `{% include %}` so each tile body can be parametrized on
//! `(M_TILE, KB, STAGES, gemm_variant)` independently.
//!
//! The huge u32 data tables (`WAVE_OPS`, `NODE_ID_FOR_OP`,
//! `WAVE_CTA_OFFSETS`) are pre-rendered in Rust as opaque strings and
//! handed to askama as `{{ wave_ops_table }}` etc. Iterating those in
//! askama would work but would balloon render time on the seq=1024
//! variant (126,976 entries).

use askama::Template;

#[derive(Template)]
#[template(path = "scheduled/megakernel.cu", escape = "none")]
pub struct MegakernelCtx<'a> {
    pub name: &'a str,

    // Schedule shape
    pub num_nodes: u32,
    pub num_waves: u32,
    pub num_ctas: u32,

    // Pre-rendered data tables (opaque to askama).
    pub wave_ops_table: String,
    pub node_id_for_op_table: String,
    pub wave_cta_offsets_table: String,

    // Model dims
    pub model_num_layers: u32,
    pub model_hidden_dim: u32,
    pub model_intermediate: u32,
    pub model_num_attn_h: u32,
    pub model_num_kv_h: u32,
    pub model_head_dim: u32,
    pub model_seq_len: u32,
    pub model_row_tile: u32,
    pub model_qkv_col_tile: u32,
    pub model_o_col_tile: u32,
    pub model_gate_up_col_tile: u32,
    pub model_down_col_tile: u32,
    pub model_kv_page_size: u32,
    pub model_pages_per_layer: u32,
}
