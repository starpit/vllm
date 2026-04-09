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
    /// Total number of (potentially-replicated) op stream entries — sums
    /// over `cta_nodes[c].len()` across every wave and CTA. With
    /// wave-cooperative bindings (e.g. FlashInferAttentionLayer) the
    /// same NodeId can appear in every CTA's stream, so `num_ops`
    /// can exceed `num_nodes`. WAVE_OPS and NODE_ID_FOR_OP are sized
    /// by num_ops; per-node validation arrays (rt.flags) stay sized
    /// by num_nodes.
    pub num_ops: u32,
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

    // ── Per-target hardware profile (see target_profile::TargetProfile) ──
    /// SM count of the target device.
    pub target_num_sm: u32,
    /// CTAs per SM under cooperative-launch residency for THIS megakernel.
    /// Multiplied with `target_num_sm` gives the cooperative grid size,
    /// which is what FlashInfer's planner must produce work_indptr for.
    pub target_cooperative_blocks_per_sm: u32,
    /// = target_num_sm * target_cooperative_blocks_per_sm. Equals
    /// `NUM_CTAS` in the rendered megakernel and the FlashInfer
    /// planner's `num_blks_y`.
    pub target_num_clusters: u32,
    /// Per-block dynamic shmem ceiling, in bytes.
    pub target_max_dynamic_shmem_bytes: u32,

    // ── Kernel choices (rendered as plain string tags so the
    //    template can `{% if target_gemm_kernel == "..." %}` branch
    //    on them). Each tag corresponds to a `*KernelChoice` enum
    //    variant in `crate::target_profile`.
    pub target_gemm_kernel: &'static str,
    pub target_attention_kernel: &'static str,
    pub target_norm_kernel: &'static str,
    pub target_rope_kernel: &'static str,

    // ── GEMM tile shape parameters (only meaningful when
    //    target_gemm_kernel is one of the CUTLASS variants).
    //    Rendered unconditionally so the template can substitute
    //    them into the cute boilerplate. For HandWrittenWmma the
    //    template branch ignores them.
    pub target_gemm_tile_m: u32,
    pub target_gemm_tile_n: u32,
    pub target_gemm_tile_k: u32,
    pub target_gemm_pipeline_stages: u32,

    // ── CP3: per-kind lowering data ──────────────────────────────────
    /// Pre-rendered host-side `WAVE_KIND_HOST[NUM_WAVES]` table
    /// (`{ 14, 15, 11, 16, 13, 14, 15, ... }`) listing each wave's
    /// `BoundKernel::kernel_tag()`. The CP3 launcher reads this to
    /// dispatch each wave to the matching per-kind `__global__`
    /// instantiation. The wave's kind is monomorphic (the BSP
    /// scheduler enforces it), so we take the first non-empty CTA
    /// stream's first op's tag and use that as the wave's kind.
    pub wave_kind_host_table: String,
    /// Distinct kernel_tag values present in the schedule, sorted.
    /// The codegen uses this to emit one explicit template
    /// instantiation per kind (avoiding the cost of compiling 17
    /// per-variant template instantiations when only ~5 are used).
    /// Iterated by askama via `{% for k in distinct_kinds %}`.
    pub distinct_kinds: Vec<u32>,
}
