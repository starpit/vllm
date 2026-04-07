// SPDX-License-Identifier: Apache-2.0
//! Template-driven codegen for the fused prefill kernel.
//!
//! Replaces the monolithic `generate_fused_prefill_layer_kernel` (1400 lines
//! of `writeln!` calls) with askama templates that look like the actual CUDA
//! they generate. The Rust side builds context structs from the DAG and
//! `FusedPrefillConfig`; askama renders them.
//!
//! Goals:
//! 1. Match v1 functionally for the 16-row, redundant-GEMM config.
//! 2. Support 128-row, cooperative-GEMM as a config switch.
//! 3. Be readable: templates show exactly what CUDA gets emitted.

pub mod config;
mod derived;
mod templates;

use askama::Template;

use crate::cuda_codegen::{
    LAUNCH_PARAMS, emit_globals_construction, emit_tensor_arg_and_globals_helper,
};
use crate::dag::ModelDag;
use crate::fused_codegen::config::{FusedPrefillConfig, GemmMode};
use crate::fused_codegen::derived::FusedDerived;
use crate::fused_codegen::templates::*;

/// Generate a fused prefill layer kernel using the template pipeline.
pub fn generate_fused_prefill_v2(dag: &ModelDag, cfg: &FusedPrefillConfig) -> String {
    let d = FusedDerived::new(dag, cfg);
    let cooperative = matches!(cfg.gemm_mode, GemmMode::Cooperative);

    // ── Preamble ──
    let mode_label = if cooperative {
        "cooperative"
    } else {
        "redundant-compute"
    };
    let preamble = PreambleCtx {
        mode_label,
        cta_rows: cfg.cta_rows,
        nl: d.nl,
        hd: d.hd,
        id: d.id,
        hdm: d.hdm,
        nah: d.nah,
        nkh: d.nkh,
        num_warps: cfg.num_warps,
        gqa_ratio: d.gqa_ratio,
        kv_page_size: cfg.kv_page_size,
        iters_per_page: d.iters_per_page,
        total_shmem: d.total_shmem,
        kv_tile_bytes: d.kv_tile_bytes,
        k_dim: cfg.k_dim,
        out_block: cfg.out_block,
        rdpw: d.rdpw,
    }
    .render()
    .expect("preamble template render");

    // ── Phases (in execution order within the layer loop) ──
    let phases: Vec<String> = vec![
        // Phase 1: attn_norm
        render_rmsnorm(
            &d,
            "g.hidden_states",
            "g.attn_norm_weights",
            "g.rms_rope_intermediates",
        ),
        // Phase 2: QKV GEMM
        render_gemm(
            &d,
            cfg,
            "Phase 2: QKV GEMM (rms_rope × qkv_weights → silu_out)",
            "g.rms_rope_intermediates",
            "g.qkv_weights",
            d.hd_k_iters,
            d.qkv_col_tiles,
            EpilogueKind::Store("g.silu_out"),
        ),
        phase_fence(),
        // Phase 2b: RoPE + KV append
        render_rope_kv_append(&d),
        // Phase 3: Attention
        render_attention(&d, cfg),
        // Phase 4: o_proj + residual
        render_gemm(
            &d,
            cfg,
            "Phase 4: o_proj + residual (attn_out × o_weights + hidden → hidden)",
            "g.attn_out",
            "g.o_weights",
            d.hd_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
        phase_fence(),
        // Phase 5: mlp_norm
        render_rmsnorm(
            &d,
            "g.hidden_states",
            "g.mlp_norm_weights",
            "g.rms_gate_intermediates",
        ),
        // Phase 6: gate + SiLU
        render_gemm(
            &d,
            cfg,
            "Phase 6: gate GEMM + SiLU",
            "g.rms_gate_intermediates",
            "g.gate_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::SiLU("g.silu_out"),
        ),
        phase_fence(),
        // Phase 7: up × gate
        render_gemm(
            &d,
            cfg,
            "Phase 7: up GEMM × gate",
            "g.rms_gate_intermediates",
            "g.up_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::MulGate("g.silu_out"),
        ),
        phase_fence(),
        // Phase 8: down + residual (no trailing fence — matches v1)
        render_gemm(
            &d,
            cfg,
            "Phase 8: down_proj + residual",
            "g.silu_out",
            "g.down_weights",
            d.id_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
    ];

    // ── Launch wrapper ──
    let mut tah = String::new();
    emit_tensor_arg_and_globals_helper(&mut tah);
    let mut gc = String::new();
    emit_globals_construction(&mut gc, "    ");

    let launch_wrapper = LaunchWrapperCtx {
        tensor_arg_helper: &tah,
        launch_params: LAUNCH_PARAMS,
        globals_construction: &gc,
        num_threads: d.num_threads,
    }
    .render()
    .expect("launch_wrapper template render");

    // ── Phase names for timing printout ──
    let phase_name_list = [
        "attn_norm",
        "QKV_GEMM",
        "fence1",
        "RoPE_KV",
        "attention",
        "o_proj",
        "fence2",
        "mlp_norm",
        "gate_silu",
        "fence3",
        "up_mulgate",
        "fence4",
        "down_residual",
    ];
    let phase_names_str = phase_name_list[..phases.len()]
        .iter()
        .map(|n| format!("\"{}\"", n))
        .collect::<Vec<_>>()
        .join(", ");

    // ── Compose into final kernel ──
    KernelCtx {
        preamble: &preamble,
        phases: &phases,
        launch_wrapper: &launch_wrapper,
        num_threads: d.num_threads,
        phase_names_str: &phase_names_str,
        num_phases: phases.len(),
    }
    .render()
    .expect("kernel template render")
}

// ── Phase rendering helpers ──────────────────────────────────────────────

fn render_rmsnorm(
    d: &FusedDerived,
    input_global: &str,
    weight_global: &str,
    output_global: &str,
) -> String {
    RmsNormCtx {
        input_global,
        weight_global,
        output_global,
        wgt_offset: d.hd * 2,
        scratch_offset: d.hd * 4,
    }
    .render()
    .expect("rmsnorm template render")
}

#[allow(clippy::too_many_arguments)]
fn render_gemm(
    d: &FusedDerived,
    cfg: &FusedPrefillConfig,
    phase_comment: &str,
    input_global: &str,
    weight_global: &str,
    num_k_iters: usize,
    num_col_tiles: usize,
    epilogue: EpilogueKind<'_>,
) -> String {
    let cooperative = matches!(cfg.gemm_mode, GemmMode::Cooperative);
    let row_var = if cooperative { "my_row_bid" } else { "bid" };

    let epilogue_str = match epilogue {
        EpilogueKind::Store(output) => EpilogueStoreCtx { output, row_var }
            .render()
            .expect("epilogue store render"),
        EpilogueKind::ResidualAdd(residual) => EpilogueResidualCtx { residual, row_var }
            .render()
            .expect("epilogue residual render"),
        EpilogueKind::SiLU(output) => EpilogueSiluCtx { output, row_var }
            .render()
            .expect("epilogue silu render"),
        EpilogueKind::MulGate(gate_output) => EpilogueMulGateCtx {
            gate_output,
            row_var,
        }
        .render()
        .expect("epilogue mulgate render"),
    };

    GemmCtx {
        phase_comment,
        input_global,
        weight_global,
        num_k_iters,
        num_col_tiles,
        a_size: d.a_size,
        b_size: d.b_size,
        stage_size: d.stage_size,
        b_offset: d.b_offset,
        epilogue: epilogue_str,
        cooperative,
        col_batch: d.col_batch,
    }
    .render()
    .expect("gemm template render")
}

fn render_rope_kv_append(d: &FusedDerived) -> String {
    let q_end = d.nah * d.hdm;
    let k_start = q_end;
    let k_end = q_end + d.nkh * d.hdm;
    let v_start = k_end;
    let kv_elems = d.nkh * d.hdm;
    RopeKvAppendCtx {
        hdm: d.hdm,
        q_end,
        k_start,
        v_start,
        kv_elems,
    }
    .render()
    .expect("rope_kv_append template render")
}

fn render_attention(d: &FusedDerived, cfg: &FusedPrefillConfig) -> String {
    AttentionCtx {
        attn_passes: cfg.attn_passes(),
        nkh: d.nkh,
        nah: d.nah,
        stage_sz: d.kv_tile_bytes * 2,
    }
    .render()
    .expect("attention template render")
}

fn phase_fence() -> String {
    "    __threadfence(); __syncthreads();\n".to_string()
}

enum EpilogueKind<'a> {
    Store(&'a str),
    ResidualAdd(&'a str),
    SiLU(&'a str),
    MulGate(&'a str),
}

// ── Multi-CTA kernel generation ─────────────────────────────────────────

/// Generate a multi-CTA fused prefill kernel where all CTAs collaborate on each phase.
pub fn generate_fused_prefill_mcta(
    dag: &ModelDag,
    cfg: &FusedPrefillConfig,
    grid_size: usize,
) -> String {
    let d = FusedDerived::new(dag, cfg);

    // ── Preamble (reuse same preamble) ──
    let preamble = PreambleCtx {
        mode_label: "multi-CTA",
        cta_rows: cfg.cta_rows,
        nl: d.nl,
        hd: d.hd,
        id: d.id,
        hdm: d.hdm,
        nah: d.nah,
        nkh: d.nkh,
        num_warps: cfg.num_warps,
        gqa_ratio: d.gqa_ratio,
        kv_page_size: cfg.kv_page_size,
        iters_per_page: d.iters_per_page,
        total_shmem: d.total_shmem,
        kv_tile_bytes: d.kv_tile_bytes,
        k_dim: cfg.k_dim,
        out_block: cfg.out_block,
        rdpw: d.rdpw,
    }
    .render()
    .expect("preamble template render");

    // ── Phases ──
    let phases: Vec<String> = vec![
        // Phase 1: attn_norm
        render_rmsnorm_mcta(
            &d,
            "g.hidden_states",
            "g.attn_norm_weights",
            "g.rms_rope_intermediates",
        ),
        // Phase 2: QKV GEMM
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 2: QKV GEMM (multi-CTA)",
            "g.rms_rope_intermediates",
            "g.qkv_weights",
            d.hd_k_iters,
            d.qkv_col_tiles,
            EpilogueKind::Store("g.silu_out"),
        ),
        // Phase 3: RoPE + KV append
        render_rope_kv_append_mcta(&d),
        // Phase 4: Attention
        render_attention_mcta(&d),
        // Phase 5: o_proj + residual
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 5: o_proj + residual (multi-CTA)",
            "g.attn_out",
            "g.o_weights",
            d.hd_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
        // Phase 6: mlp_norm
        render_rmsnorm_mcta(
            &d,
            "g.hidden_states",
            "g.mlp_norm_weights",
            "g.rms_gate_intermediates",
        ),
        // Phase 7: gate + SiLU
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 7: gate GEMM + SiLU (multi-CTA)",
            "g.rms_gate_intermediates",
            "g.gate_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::SiLU("g.silu_out"),
        ),
        // Phase 8: up × gate
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 8: up GEMM × gate (multi-CTA)",
            "g.rms_gate_intermediates",
            "g.up_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::MulGate("g.silu_out"),
        ),
        // Phase 9: down + residual
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 9: down_proj + residual (multi-CTA)",
            "g.silu_out",
            "g.down_weights",
            d.id_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
    ];

    // ── Launch wrapper ──
    let mut tah = String::new();
    emit_tensor_arg_and_globals_helper(&mut tah);
    let mut gc = String::new();
    emit_globals_construction(&mut gc, "    ");

    let launch_wrapper = LaunchWrapperMctaCtx {
        tensor_arg_helper: &tah,
        launch_params: LAUNCH_PARAMS,
        globals_construction: &gc,
        num_threads: d.num_threads,
        grid_size,
        id_col_tiles: d.id_col_tiles,
    }
    .render()
    .expect("launch_wrapper_mcta template render");

    // ── Phase names ──
    let phase_name_list = [
        "attn_norm",
        "QKV_GEMM",
        "RoPE_KV",
        "attention",
        "o_proj",
        "mlp_norm",
        "gate_silu",
        "up_mulgate",
        "down_residual",
    ];
    let phase_names_str = phase_name_list[..phases.len()]
        .iter()
        .map(|n| format!("\"{}\"", n))
        .collect::<Vec<_>>()
        .join(", ");

    // ── Compose ──
    KernelMctaCtx {
        preamble: &preamble,
        phases: &phases,
        launch_wrapper: &launch_wrapper,
        num_threads: d.num_threads,
        phase_names_str: &phase_names_str,
        num_phases: phases.len(),
        grid_size,
    }
    .render()
    .expect("kernel_mcta template render")
}

/// Generate a multi-CTA fused prefill kernel with fused gate+up (8 phases instead of 9).
pub fn generate_fused_prefill_mcta_fused_gateup(
    dag: &ModelDag,
    cfg: &FusedPrefillConfig,
    grid_size: usize,
) -> String {
    let d = FusedDerived::new(dag, cfg);

    let preamble = PreambleCtx {
        mode_label: "multi-CTA-fused-gateup",
        cta_rows: cfg.cta_rows,
        nl: d.nl,
        hd: d.hd,
        id: d.id,
        hdm: d.hdm,
        nah: d.nah,
        nkh: d.nkh,
        num_warps: cfg.num_warps,
        gqa_ratio: d.gqa_ratio,
        kv_page_size: cfg.kv_page_size,
        iters_per_page: d.iters_per_page,
        total_shmem: d.total_shmem,
        kv_tile_bytes: d.kv_tile_bytes,
        k_dim: cfg.k_dim,
        out_block: cfg.out_block,
        rdpw: d.rdpw,
    }
    .render()
    .expect("preamble template render");

    // 8 phases: attn_norm, QKV, RoPE, attention, o_proj, mlp_norm, fused_gate_up, down
    let phases: Vec<String> = vec![
        render_rmsnorm_mcta(
            &d,
            "g.hidden_states",
            "g.attn_norm_weights",
            "g.rms_rope_intermediates",
        ),
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 2: QKV GEMM (multi-CTA)",
            "g.rms_rope_intermediates",
            "g.qkv_weights",
            d.hd_k_iters,
            d.qkv_col_tiles,
            EpilogueKind::Store("g.silu_out"),
        ),
        render_rope_kv_append_mcta(&d),
        render_attention_mcta(&d),
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 5: o_proj + residual (multi-CTA)",
            "g.attn_out",
            "g.o_weights",
            d.hd_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
        render_rmsnorm_mcta(
            &d,
            "g.hidden_states",
            "g.mlp_norm_weights",
            "g.rms_gate_intermediates",
        ),
        // Fused gate+up: one phase, no barrier between gate and up
        render_gemm_gate_up_mcta(
            &d,
            cfg,
            "Phase 7: fused gate+up (multi-CTA)",
            "g.rms_gate_intermediates",
            "g.gate_weights",
            "g.up_weights",
            "g.silu_out",
            d.hd_k_iters,
            d.id_col_tiles,
        ),
        render_gemm_mcta(
            &d,
            cfg,
            "Phase 8: down_proj + residual (multi-CTA)",
            "g.silu_out",
            "g.down_weights",
            d.id_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
    ];

    let mut tah = String::new();
    emit_tensor_arg_and_globals_helper(&mut tah);
    let mut gc = String::new();
    emit_globals_construction(&mut gc, "    ");

    let launch_wrapper = LaunchWrapperMctaCtx {
        tensor_arg_helper: &tah,
        launch_params: LAUNCH_PARAMS,
        globals_construction: &gc,
        num_threads: d.num_threads,
        grid_size,
        id_col_tiles: d.id_col_tiles,
    }
    .render()
    .expect("launch_wrapper_mcta template render");

    let phase_name_list = [
        "attn_norm",
        "QKV_GEMM",
        "RoPE_KV",
        "attention",
        "o_proj",
        "mlp_norm",
        "fused_gate_up",
        "down_residual",
    ];
    let phase_names_str = phase_name_list[..phases.len()]
        .iter()
        .map(|n| format!("\"{}\"", n))
        .collect::<Vec<_>>()
        .join(", ");

    KernelMctaCtx {
        preamble: &preamble,
        phases: &phases,
        launch_wrapper: &launch_wrapper,
        num_threads: d.num_threads,
        phase_names_str: &phase_names_str,
        num_phases: phases.len(),
        grid_size,
    }
    .render()
    .expect("kernel_mcta template render")
}

// ── Multi-CTA phase rendering helpers ───────────────────────────────────

fn render_rmsnorm_mcta(
    d: &FusedDerived,
    input_global: &str,
    weight_global: &str,
    output_global: &str,
) -> String {
    RmsNormMctaCtx {
        input_global,
        weight_global,
        output_global,
        wgt_offset: d.hd * 2,
        scratch_offset: d.hd * 4,
    }
    .render()
    .expect("rmsnorm_mcta template render")
}

#[allow(clippy::too_many_arguments)]
fn render_gemm_mcta(
    d: &FusedDerived,
    cfg: &FusedPrefillConfig,
    phase_comment: &str,
    input_global: &str,
    weight_global: &str,
    num_k_iters: usize,
    num_col_tiles: usize,
    epilogue: EpilogueKind<'_>,
) -> String {
    let cooperative = matches!(cfg.gemm_mode, GemmMode::Cooperative);
    // In cooperative mode, each warp stores its own row_tile; template sets it as local var
    let row_var = "row_tile";

    let epilogue_str = match epilogue {
        EpilogueKind::Store(output) => EpilogueStoreCtx { output, row_var }
            .render()
            .expect("epilogue store render"),
        EpilogueKind::ResidualAdd(residual) => EpilogueResidualCtx { residual, row_var }
            .render()
            .expect("epilogue residual render"),
        EpilogueKind::SiLU(output) => EpilogueSiluCtx { output, row_var }
            .render()
            .expect("epilogue silu render"),
        EpilogueKind::MulGate(gate_output) => EpilogueMulGateCtx {
            gate_output,
            row_var,
        }
        .render()
        .expect("epilogue mulgate render"),
    };

    GemmMctaCtx {
        phase_comment,
        input_global,
        weight_global,
        num_k_iters,
        num_col_tiles,
        a_size: d.a_size,
        b_size: d.b_size,
        stage_size: d.stage_size,
        b_offset: d.b_offset,
        epilogue: epilogue_str,
        cooperative,
        col_batch: cfg.col_batch,
        num_stages: cfg.num_stages,
    }
    .render()
    .expect("gemm_mcta template render")
}

fn render_rope_kv_append_mcta(d: &FusedDerived) -> String {
    let q_end = d.nah * d.hdm;
    let k_start = q_end;
    let k_end = q_end + d.nkh * d.hdm;
    let v_start = k_end;
    let kv_elems = d.nkh * d.hdm;
    RopeKvAppendMctaCtx {
        hdm: d.hdm,
        q_end,
        k_start,
        v_start,
        kv_elems,
    }
    .render()
    .expect("rope_kv_append_mcta template render")
}

#[allow(clippy::too_many_arguments)]
fn render_gemm_gate_up_mcta(
    d: &FusedDerived,
    cfg: &FusedPrefillConfig,
    phase_comment: &str,
    input_global: &str,
    gate_weight_global: &str,
    up_weight_global: &str,
    output_global: &str,
    num_k_iters: usize,
    num_col_tiles: usize,
) -> String {
    let cooperative = matches!(cfg.gemm_mode, GemmMode::Cooperative);
    GemmGateUpMctaCtx {
        phase_comment,
        input_global,
        gate_weight_global,
        up_weight_global,
        output_global,
        num_k_iters,
        num_col_tiles,
        a_size: d.a_size,
        b_size: d.b_size,
        stage_size: d.stage_size,
        b_offset: d.b_offset,
        cooperative,
        num_stages: cfg.num_stages,
    }
    .render()
    .expect("gemm_gate_up_mcta template render")
}

fn render_attention_mcta(d: &FusedDerived) -> String {
    AttentionMctaCtx {
        nkh: d.nkh,
        nah: d.nah,
        stage_sz: d.kv_tile_bytes * 2,
    }
    .render()
    .expect("attention_mcta template render")
}

// ── Decode kernel generation ────────────────────────────────────────────

use crate::fused_codegen::config::{
    ByteSize, FusedDecodeConfig, KIters, ShmemOffset, StorageStrategy, TileCount,
};
use crate::fused_codegen::derived::FusedDecodeDerived;

/// Generate a fused decode layer kernel using the template pipeline.
///
/// Row-fused architecture: each CTA owns `cta_rows` sequences and runs them
/// through ALL phases of ALL layers. No cross-CTA barriers.
pub fn generate_fused_decode(dag: &ModelDag, cfg: &FusedDecodeConfig) -> String {
    let d = FusedDecodeDerived::new(dag, cfg);

    // ── Preamble ──
    let preamble = DecodePreambleCtx {
        cta_rows: cfg.cta_rows,
        padded_cta_rows: d.padded_cta_rows,
        nl: d.nl,
        hd: d.hd,
        id: d.id,
        hdm: d.hdm,
        nah: d.nah,
        nkh: d.nkh,
        num_warps: cfg.num_warps,
        gqa_ratio: d.gqa_ratio,
        kv_page_size: cfg.kv_page_size,
        iters_per_page: d.iters_per_page,
        peak_shmem: d.peak_shmem.0,
        kv_tile_bytes: d.kv_tile_bytes.0,
        k_dim: cfg.k_dim,
        out_block: cfg.out_block,
        hidden_shmem: d.hidden_shmem.0,
        meta_shmem: d.meta_shmem.0,
        num_stages: cfg.num_stages,
    }
    .render()
    .expect("decode preamble render");

    // ── Phases ──
    let rdpw = d.hd / cfg.num_warps;

    // Shmem layout offsets within phase_shm:
    //   [0 .. hidden_shmem): hidden_states slab [padded_rows, HD] BF16
    //   The rest is time-shared by different phases.
    // For attention, we reuse the hidden slab area for Q and attn_out.
    let q_stride = d.nah * d.hdm; // Q elements per row
    let attn_out_stride = d.nah * d.hdm; // same as Q

    let hidden_offset = ShmemOffset(0);
    let wgt_offset = ShmemOffset(d.hidden_shmem.0);
    let scratch_offset = ShmemOffset(d.hidden_shmem.0 + d.hd * 2);

    let phases: Vec<String> = vec![
        // Phase 1: attn_norm (hidden_shmem → hidden_shmem, in-place)
        render_decode_rmsnorm(
            &d,
            cfg,
            "g.attn_norm_weights",
            hidden_offset,
            hidden_offset,
            wgt_offset,
            scratch_offset,
            rdpw,
        ),
        // Phase 2: QKV GEMM (shmem activations × qkv_weights → global qkv output)
        // Output goes to global because K/V need to be written to paged cache.
        render_decode_gemm(
            &d,
            cfg,
            "Phase 2: QKV GEMM (shmem × qkv_weights → silu_out)",
            hidden_offset,
            "g.qkv_weights",
            d.hd_k_iters,
            d.qkv_col_tiles,
            DecodeEpilogueKind::StoreGlobal("g.silu_out"),
        ),
        // Phase 3: RoPE + KV cache append
        // Reads QKV from global silu_out, applies RoPE, writes K/V to cache, Q to shmem
        render_decode_rope_kv_append(&d, hidden_offset, q_stride),
        // Phase 4: Decode attention (Q from shmem, KV from cache, output to shmem)
        render_decode_attention(&d, hidden_offset, q_stride, hidden_offset, attn_out_stride),
        // Phase 5: o_proj + residual (attn_out from shmem × o_weights + hidden → hidden)
        // After attention, attn_out is in shmem at offset 0.
        // hidden_states were written to global at start; we need to re-read.
        // Actually, hidden_states are maintained in the hidden slab.
        // But attention overwrote shmem... Let me reconsider.
        //
        // Design: hidden_states live at shmem offset 0 between attn_norm and the
        // first GEMM. After QKV GEMM, the hidden slab is no longer needed for attn_norm
        // output. We need hidden_states for the residual in o_proj.
        //
        // Solution: write hidden_states to global BEFORE the QKV GEMM,
        // then read back for residual. OR: keep hidden in a separate slab.
        //
        // For now: read residual from global hidden_states.
        render_decode_gemm(
            &d,
            cfg,
            "Phase 5: o_proj + residual (attn_out × o_weights + hidden → hidden_shmem)",
            hidden_offset,
            "g.o_weights",
            d.hd_k_iters,
            d.hd_col_tiles,
            DecodeEpilogueKind::ResidualShmem {
                residual_offset: hidden_offset,
                output_offset: hidden_offset,
            },
        ),
        // Phase 6: mlp_norm (hidden_shmem → hidden_shmem, in-place)
        render_decode_rmsnorm(
            &d,
            cfg,
            "g.mlp_norm_weights",
            hidden_offset,
            hidden_offset,
            wgt_offset,
            scratch_offset,
            rdpw,
        ),
        // Phase 7: Fused MLP (streaming gate+up+SiLU+down over ID tiles)
        render_decode_fused_mlp(&d, cfg, hidden_offset, hidden_offset),
    ];

    // ── Launch wrapper ──
    let mut tah = String::new();
    emit_tensor_arg_and_globals_helper(&mut tah);
    let mut gc = String::new();
    emit_globals_construction(&mut gc, "    ");

    let launch_wrapper = DecodeLaunchWrapperCtx {
        tensor_arg_helper: &tah,
        launch_params: LAUNCH_PARAMS,
        globals_construction: &gc,
        num_threads: d.num_threads,
    }
    .render()
    .expect("decode launch_wrapper render");

    // ── Phase names ──
    let phase_name_list = [
        "attn_norm",
        "QKV_GEMM",
        "RoPE_KV",
        "attention",
        "o_proj",
        "mlp_norm",
        "fused_MLP",
    ];
    let phase_names_str = phase_name_list[..phases.len()]
        .iter()
        .map(|n| format!("\"{}\"", n))
        .collect::<Vec<_>>()
        .join(", ");

    // ── Compose ──
    DecodeKernelCtx {
        preamble: &preamble,
        phases: &phases,
        launch_wrapper: &launch_wrapper,
        num_threads: d.num_threads,
        phase_names_str: &phase_names_str,
        num_phases: phases.len(),
        cta_rows: cfg.cta_rows,
        padded_cta_rows: d.padded_cta_rows,
        hidden_shmem: d.hidden_shmem.0,
        meta_shmem: d.meta_shmem.0,
    }
    .render()
    .expect("decode kernel render")
}

// ── DAG-driven storage assignment ──────────────────────────────────────

use crate::dag::{BufferId, BufferKind};
use std::collections::HashMap;

/// Assign a `StorageStrategy` to each activation buffer in the DAG, based on
/// the shmem budget, buffer sizes, and the polyalgorithmic rules.
///
/// Weights, KV cache, and metadata always get `Global`. The interesting choices
/// are for activations:
/// - `hidden_states` → Shmem(0), persistent across phases
/// - Norm outputs → FusedIntoConsumer (when shmem budget doesn't allow a second slab)
/// - QKV output → Global (too large for shmem alongside hidden)
/// - Q after RoPE → Shmem (reuses hidden slab area since hidden is preserved elsewhere)
/// - Attention output → Shmem (same region as Q)
/// - MLP intermediates (gate, up, silu*up) → Register (streaming, never materialized)
/// - hidden_states after residual → Shmem(0), overwrites previous value
#[allow(dead_code)]
pub fn assign_decode_storage(
    dag: &ModelDag,
    d: &FusedDecodeDerived,
    _cfg: &FusedDecodeConfig,
) -> HashMap<BufferId, StorageStrategy> {
    let mut storage = HashMap::new();
    let hidden_offset = ShmemOffset(0);
    let hidden_size = d.hidden_shmem;

    for (id, buf) in &dag.buffers {
        let strategy = match buf.kind {
            BufferKind::Weight | BufferKind::KvCache | BufferKind::Metadata => {
                StorageStrategy::Global {
                    accessor: format!("g.{}", id.0),
                }
            }
            BufferKind::Activation => {
                assign_activation_storage(id, buf, dag, d, hidden_offset, hidden_size)
            }
        };
        storage.insert(id.clone(), strategy);
    }
    storage
}

/// Polyalgorithmic activation storage assignment.
///
/// Rules (for cta_rows=16, HD=2048):
/// - `hidden_states`: Shmem(0) — persistent, 64KB, the primary slab
/// - Norm outputs (`normed`, `mlp_normed`): FusedIntoConsumer — no separate buffer,
///   inv_rms precomputed and applied during GEMM A-load. Keeps hidden in shmem.
/// - `qkv`: Global — 3072 × 16 × 2 = 96KB, too large for shmem alongside hidden
/// - `q_post_rope`: Shmem(0) — reuses hidden slab (hidden preserved via FusedIntoConsumer)
/// - `attn_out`: Shmem(0) — reuses same region, feeds o_proj
/// - `gate_out`, `up_out`, `silu_out` (MLP intermediates): Register — streaming
/// - `mlp_out`: Shmem(0) — final result overwrites hidden slab
#[allow(dead_code)]
fn assign_activation_storage(
    id: &BufferId,
    _buf: &crate::dag::Buffer,
    _dag: &ModelDag,
    _d: &FusedDecodeDerived,
    hidden_offset: ShmemOffset,
    hidden_size: ByteSize,
) -> StorageStrategy {
    match id.0.as_str() {
        // Primary hidden state slab — persistent in shmem
        "hidden_states" | "hidden_post_attn" | "hidden_post_mlp" => StorageStrategy::Shmem {
            offset: hidden_offset,
            size: hidden_size,
        },

        // Norm outputs — fused into the consuming GEMM's A-load
        "normed" | "mlp_normed" => StorageStrategy::FusedIntoConsumer,

        // QKV projection output — too large for shmem, goes to global
        "qkv" => StorageStrategy::Global {
            accessor: "g.silu_out".into(),
        },

        // Q after RoPE — lives in shmem, reuses hidden slab region
        // (hidden is preserved because norm is FusedIntoConsumer)
        "q_post_rope" => StorageStrategy::Shmem {
            offset: hidden_offset,
            size: hidden_size,
        },

        // Attention output — same shmem region as Q (Q consumed before attn writes)
        "attn_out" => StorageStrategy::Shmem {
            offset: hidden_offset,
            size: hidden_size,
        },

        // K/V outputs go to paged cache (global)
        "k_out" | "v_out" => StorageStrategy::Global {
            accessor: "g.kv_cache".into(),
        },

        // MLP intermediates — streaming in registers, never materialized
        "gate_out" | "up_out" | "silu_mul_out" => StorageStrategy::Register,

        // Fallback: anything else goes to global
        other => StorageStrategy::Global {
            accessor: format!("g.{other}"),
        },
    }
}

// ── Decode phase rendering helpers ──────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn render_decode_rmsnorm(
    _d: &FusedDecodeDerived,
    _cfg: &FusedDecodeConfig,
    weight_global: &str,
    input_offset: ShmemOffset,
    output_offset: ShmemOffset,
    wgt_shmem_offset: ShmemOffset,
    scratch_offset: ShmemOffset,
    rdpw: usize,
) -> String {
    DecodeRmsNormCtx {
        weight_global,
        input_offset: input_offset.0,
        output_offset: output_offset.0,
        wgt_shmem_offset: wgt_shmem_offset.0,
        scratch_offset: scratch_offset.0,
        rdpw,
    }
    .render()
    .expect("decode rmsnorm render")
}

enum DecodeEpilogueKind<'a> {
    /// Store to shmem slab (offset, stride in elements).
    StoreShmem(ShmemOffset, usize),
    /// Store to global.
    StoreGlobal(&'a str),
    /// Add residual from shmem, write to shmem.
    ResidualShmem {
        residual_offset: ShmemOffset,
        output_offset: ShmemOffset,
    },
}

#[allow(clippy::too_many_arguments)]
fn render_decode_gemm(
    d: &FusedDecodeDerived,
    cfg: &FusedDecodeConfig,
    phase_comment: &str,
    input_shmem_offset: ShmemOffset,
    weight_global: &str,
    num_k_iters: KIters,
    num_col_tiles: TileCount,
    epilogue: DecodeEpilogueKind<'_>,
) -> String {
    let epilogue_str = match epilogue {
        DecodeEpilogueKind::StoreShmem(offset, stride) => DecodeEpilogueStoreShmemCtx {
            output_shmem_offset: offset.0,
            output_stride: stride,
            col_var: "col",
            a_size: d.a_size.0,
            b_size: d.b_size.0,
        }
        .render()
        .expect("decode epilogue store shmem"),
        DecodeEpilogueKind::StoreGlobal(output) => DecodeEpilogueStoreGlobalCtx {
            output_global: output,
            col_var: "col",
            a_size: d.a_size.0,
            b_size: d.b_size.0,
        }
        .render()
        .expect("decode epilogue store global"),
        DecodeEpilogueKind::ResidualShmem {
            residual_offset,
            output_offset,
        } => DecodeEpilogueResidualShmemCtx {
            residual_shmem_offset: residual_offset.0,
            output_shmem_offset: output_offset.0,
            col_var: "col",
            a_size: d.a_size.0,
            b_size: d.b_size.0,
        }
        .render()
        .expect("decode epilogue residual shmem"),
    };

    DecodeGemmCtx {
        phase_comment,
        input_shmem_offset: input_shmem_offset.0,
        weight_global,
        num_k_iters: num_k_iters.0,
        num_col_tiles: num_col_tiles.0,
        a_size: d.a_size.0,
        b_size: d.b_size.0,
        stage_size: d.stage_size.0,
        b_offset: d.b_offset.0,
        num_stages: cfg.num_stages,
        epilogue: epilogue_str,
    }
    .render()
    .expect("decode gemm render")
}

fn render_decode_rope_kv_append(
    d: &FusedDecodeDerived,
    q_output_shmem_offset: ShmemOffset,
    q_output_stride: usize,
) -> String {
    let q_end = d.nah * d.hdm;
    let k_start = q_end;
    let k_end = q_end + d.nkh * d.hdm;
    let v_start = k_end;
    let kv_elems = d.nkh * d.hdm;
    DecodeRopeKvAppendCtx {
        hdm: d.hdm,
        q_end,
        k_start,
        v_start,
        kv_elems,
        q_output_shmem_offset: q_output_shmem_offset.0,
        q_output_stride,
    }
    .render()
    .expect("decode rope_kv_append render")
}

fn render_decode_attention(
    d: &FusedDecodeDerived,
    q_shmem_offset: ShmemOffset,
    q_stride: usize,
    output_shmem_offset: ShmemOffset,
    output_stride: usize,
) -> String {
    DecodeAttentionCtx {
        nkh: d.nkh,
        nah: d.nah,
        q_shmem_offset: q_shmem_offset.0,
        q_stride,
        output_shmem_offset: output_shmem_offset.0,
        output_stride,
    }
    .render()
    .expect("decode attention render")
}

fn render_decode_fused_mlp(
    d: &FusedDecodeDerived,
    cfg: &FusedDecodeConfig,
    input_shmem_offset: ShmemOffset,
    hidden_shmem_offset: ShmemOffset,
) -> String {
    DecodeFusedMlpCtx {
        input_shmem_offset: input_shmem_offset.0,
        hidden_shmem_offset: hidden_shmem_offset.0,
        gate_weight_global: "g.gate_weights",
        up_weight_global: "g.up_weights",
        down_weight_global: "g.down_weights",
        hd_k_iters: d.hd_k_iters.0,
        id_col_tiles: d.id_col_tiles.0,
        a_size: d.a_size.0,
        b_size: d.b_size.0,
        num_stages: cfg.num_stages,
        hd: d.hd,
        id: d.id,
    }
    .render()
    .expect("decode fused_mlp render")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn build_1b_dag() -> ModelDag {
        let tokens: proc_macro2::TokenStream = quote::quote! {
            kernel llama_sm89<NL=16, HD=2048, ID=8192, HDM=64, NAH=32, NKH=8, VS=128256> {
                for layer in 0..NL {
                    let normed = rmsnorm(hidden_states, attn_norm[layer]);
                    let qkv = gemm(normed, qkv_weights[layer]);
                    let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
                    let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
                    hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

                    let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
                    let gate = silu(gemm(normed2, gate_weights[layer]));
                    let up = gemm(normed2, up_weights[layer]);
                    hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
                }
                let normed = rmsnorm(hidden_states, lm_head_norm);
                logits = gemm(normed, lm_head);
            }
        };
        let def: parse::MegakernelDef = syn::parse2(tokens).expect("parse");
        parse::build_dag(&def).expect("build dag")
    }

    #[test]
    fn renders_16row_kernel() {
        let dag = build_1b_dag();
        let v2 = generate_fused_prefill_v2(&dag, &FusedPrefillConfig::rows16());

        // Check key markers
        assert!(
            v2.contains("fused_prefill_layer("),
            "missing kernel function"
        );
        assert!(
            v2.contains("fused_prefill_layer_launch("),
            "missing launch wrapper"
        );
        assert!(v2.contains("PFL_CTA_ROWS = 16"), "wrong CTA_ROWS value");

        // All phases present
        assert!(v2.contains("g.attn_norm_weights"), "missing attn_norm");
        assert!(v2.contains("g.qkv_weights"), "missing QKV");
        assert!(v2.contains("g.o_weights"), "missing o_proj");
        assert!(v2.contains("g.mlp_norm_weights"), "missing mlp_norm");
        assert!(v2.contains("g.gate_weights"), "missing gate");
        assert!(v2.contains("g.up_weights"), "missing up");
        assert!(v2.contains("g.down_weights"), "missing down");

        // Key ops
        assert!(v2.contains("warp::mma_ABt"), "missing GEMM MMA");
        assert!(v2.contains("rms_norm_eps"), "missing RMSNorm");
        assert!(v2.contains("1.f + expf(-v.x)"), "missing SiLU");
        assert!(v2.contains("RoPE"), "missing RoPE");
    }

    #[test]
    fn renders_128row_cooperative_kernel() {
        let dag = build_1b_dag();
        let v2 = generate_fused_prefill_v2(&dag, &FusedPrefillConfig::rows128());

        assert!(v2.contains("PFL_CTA_ROWS = 128"));
        assert!(v2.contains("cooperative"));
        assert!(v2.contains("my_row_bid"), "missing per-warp row index");
        // Each warp loads its own A tile
        assert!(v2.contains("warp::load_async(*my_a_stages"));
        // Attention 8-pass loop
        assert!(v2.contains("for (int attn_pass = 0; attn_pass < 8; attn_pass++)"));
    }

    #[test]
    fn renders_mcta_kernel() {
        let dag = build_1b_dag();
        let v2 = generate_fused_prefill_mcta(&dag, &FusedPrefillConfig::rows16_col4(), 128);

        assert!(v2.contains("fused_prefill_layer(const globals g, int batch_size, int num_layers, int *mcta_bar)"), "missing mcta kernel function");
        assert!(
            v2.contains("fused_prefill_layer_launch("),
            "missing mcta launch wrapper"
        );
        assert!(v2.contains("MCTA_MAX_GRID = 128"), "missing grid size");
        assert!(v2.contains("mcta_barrier"), "missing cross-CTA barrier");
        // Multi-CTA GEMM: tiles distributed via wu loop
        assert!(
            v2.contains("for (int wu = bid; wu < total_work; wu += num_ctas)"),
            "missing multi-CTA tile distribution"
        );
        // Multi-CTA RMSNorm: row distribution
        assert!(
            v2.contains("for (int rb = bid; rb < row_tiles; rb += num_ctas)"),
            "missing multi-CTA row distribution"
        );
        // Multi-CTA Attention: (q_block, kv_head) distribution
        assert!(
            v2.contains("const int attn_work = row_tiles *"),
            "missing attention work distribution"
        );
        // Col-batch
        assert!(v2.contains("COL_BATCH"), "missing col_batch");
        // Barrier allocation
        assert!(v2.contains("cudaMallocAsync"), "missing barrier allocation");
        // All weight phases
        assert!(v2.contains("g.qkv_weights"), "missing QKV");
        assert!(v2.contains("g.o_weights"), "missing o_proj");
        assert!(v2.contains("g.gate_weights"), "missing gate");
        assert!(v2.contains("g.up_weights"), "missing up");
        assert!(v2.contains("g.down_weights"), "missing down");
    }

    #[test]
    fn renders_mcta_fused_gateup_kernel() {
        let dag = build_1b_dag();
        let v2 =
            generate_fused_prefill_mcta_fused_gateup(&dag, &FusedPrefillConfig::rows128(), 128);

        // 8 phases instead of 9
        assert!(v2.contains("MCTA_NUM_PHASES = 8"), "should have 8 phases");
        // Fused gate+up marker
        assert!(
            v2.contains("fused gate+up"),
            "missing fused gate+up comment"
        );
        // Both weight references in one phase
        assert!(v2.contains("g.gate_weights"), "missing gate weights");
        assert!(v2.contains("g.up_weights"), "missing up weights");
        // SiLU epilogue inline
        assert!(
            v2.contains("1.f + expf(-v.x)"),
            "missing inline SiLU in fused gate+up"
        );
        // Mulgate epilogue inline
        assert!(v2.contains("mulgate"), "missing mulgate in fused gate+up");
        // All other phases still present
        assert!(v2.contains("g.qkv_weights"), "missing QKV");
        assert!(v2.contains("g.down_weights"), "missing down");
    }

    #[test]
    fn dump_for_diff() {
        let dag = build_1b_dag();
        let v1 = crate::cuda_codegen::generate_fused_prefill_layer_kernel(&dag);
        let v2_16 = generate_fused_prefill_v2(&dag, &FusedPrefillConfig::rows16());
        let v2_128 = generate_fused_prefill_v2(&dag, &FusedPrefillConfig::rows128());

        let v2_mcta = generate_fused_prefill_mcta(&dag, &FusedPrefillConfig::rows16_col4(), 128);
        let v2_mcta_fused =
            generate_fused_prefill_mcta_fused_gateup(&dag, &FusedPrefillConfig::rows128(), 128);

        std::fs::write("/tmp/fused_v1.cu", &v1).ok();
        std::fs::write("/tmp/fused_v2_16.cu", &v2_16).ok();
        std::fs::write("/tmp/fused_v2_128.cu", &v2_128).ok();
        std::fs::write("/tmp/fused_v2_mcta.cu", &v2_mcta).ok();
        std::fs::write("/tmp/fused_v2_mcta_fused.cu", &v2_mcta_fused).ok();

        let dec = generate_fused_decode(&dag, &config::FusedDecodeConfig::decode_rows16());
        std::fs::write("/tmp/fused_decode_rows16.cu", &dec).ok();

        eprintln!("v1: {} lines", v1.lines().count());
        eprintln!("v2_16: {} lines", v2_16.lines().count());
        eprintln!("v2_128: {} lines", v2_128.lines().count());
        eprintln!("v2_mcta: {} lines", v2_mcta.lines().count());
        eprintln!("v2_mcta_fused: {} lines", v2_mcta_fused.lines().count());
        eprintln!("decode_rows16: {} lines", dec.lines().count());
    }

    // Smoke test from earlier — kept to verify askama setup
    #[derive(Template)]
    #[template(path = "fused/smoke.txt")]
    struct SmokeContext {
        hd: usize,
        nl: usize,
        phases: Vec<&'static str>,
    }

    #[test]
    fn askama_smoke() {
        let ctx = SmokeContext {
            hd: 2048,
            nl: 16,
            phases: vec!["rmsnorm", "qkv_gemm", "attention"],
        };
        let rendered = ctx.render().unwrap();
        assert!(rendered.contains("hd = 2048"));
        assert!(rendered.contains("phase: rmsnorm"));
    }

    // ── Decode config / derived tests ─────────────────────────────────

    #[test]
    fn renders_decode_rmsnorm_kernel() {
        let dag = build_1b_dag();
        let cfg = config::FusedDecodeConfig::decode_rows16();
        let cuda = generate_fused_decode(&dag, &cfg);

        // Kernel function
        assert!(
            cuda.contains("fused_decode_layer("),
            "missing decode kernel function"
        );
        assert!(
            cuda.contains("fused_decode_layer_launch("),
            "missing decode launch wrapper"
        );

        // Decode-specific constants
        assert!(cuda.contains("DEC_CTA_ROWS      = 16"), "wrong CTA_ROWS");
        assert!(cuda.contains("DEC_PADDED_ROWS   = 16"), "wrong PADDED_ROWS");
        assert!(cuda.contains("DEC_NUM_WARPS     = 8"), "wrong NUM_WARPS");

        // Row-fused: no cross-CTA barriers
        assert!(
            !cuda.contains("mcta_barrier"),
            "decode should NOT have cross-CTA barriers"
        );

        // Per-row metadata
        assert!(
            cuda.contains("DecRowMeta"),
            "missing per-row metadata struct"
        );
        assert!(
            cuda.contains("row_meta[r].position_id"),
            "missing metadata load"
        );

        // Hidden states in shmem
        assert!(
            cuda.contains("DEC_HIDDEN_SHMEM"),
            "missing hidden shmem constant"
        );

        // RMSNorm from shmem
        assert!(
            cuda.contains("g.attn_norm_weights"),
            "missing attn_norm weight"
        );
        assert!(cuda.contains("rms_norm_eps"), "missing RMSNorm epsilon");
        assert!(cuda.contains("rsqrtf"), "missing rsqrtf in RMSNorm");

        // QKV GEMM
        assert!(cuda.contains("g.qkv_weights"), "missing QKV weight");
        assert!(cuda.contains("mma_ABt_base"), "missing GEMM MMA in decode");
        assert!(cuda.contains("g.silu_out"), "missing QKV output");

        // RoPE + KV append
        assert!(cuda.contains("rope_cos"), "missing RoPE cos");
        assert!(cuda.contains("k_cache"), "missing K cache write");
        assert!(cuda.contains("v_cache"), "missing V cache write");

        // Attention
        assert!(cuda.contains("Decode attention"), "missing attention phase");
        assert!(cuda.contains("Online softmax"), "missing online softmax");
        assert!(cuda.contains("DEC_GQA_RATIO"), "missing GQA");

        // o_proj
        assert!(cuda.contains("g.o_weights"), "missing o_proj weights");

        // MLP norm
        assert!(cuda.contains("g.mlp_norm_weights"), "missing mlp_norm");

        // Fused MLP
        assert!(cuda.contains("g.gate_weights"), "missing gate weights");
        assert!(cuda.contains("g.up_weights"), "missing up weights");
        assert!(cuda.contains("g.down_weights"), "missing down weights");
        assert!(cuda.contains("SiLU"), "missing SiLU in fused MLP");

        // 7 phases
        assert!(cuda.contains("NUM_PHASES = 7"), "expected 7 phases");

        // Layer loop
        assert!(
            cuda.contains("for (int layer = 0; layer < num_layers; layer++)"),
            "missing layer loop"
        );

        // Final writeback
        assert!(
            cuda.contains("Write final hidden_states back to global"),
            "missing final writeback"
        );
    }

    #[test]
    fn decode_derived_1b_rows16() {
        let dag = build_1b_dag();
        let cfg = config::FusedDecodeConfig::decode_rows16();
        let d = derived::FusedDecodeDerived::new(&dag, &cfg);

        // Model dimensions
        assert_eq!(d.hd, 2048);
        assert_eq!(d.id, 8192);
        assert_eq!(d.nl, 16);
        assert_eq!(d.nah, 32);
        assert_eq!(d.nkh, 8);
        assert_eq!(d.hdm, 64);
        assert_eq!(d.gqa_ratio, 4);
        assert_eq!(d.qkv_dim, (32 + 16) * 64); // 3072

        // CTA geometry
        assert_eq!(d.cta_rows, 16);
        assert_eq!(d.padded_cta_rows, 16);
        assert_eq!(d.num_warps, 8);
        assert_eq!(d.num_threads, 256);

        // Shmem
        let hidden = 16 * 2048 * 2; // 65536
        assert_eq!(d.hidden_shmem.0, hidden);

        // Peak must fit in sm89
        assert!(
            d.peak_shmem.0 <= config::SM89_MAX_SHMEM,
            "peak_shmem {} exceeds sm89 limit {}",
            d.peak_shmem.0,
            config::SM89_MAX_SHMEM,
        );
    }

    #[test]
    fn decode_derived_rows4() {
        let dag = build_1b_dag();
        let cfg = config::FusedDecodeConfig::decode_rows4();
        let d = derived::FusedDecodeDerived::new(&dag, &cfg);
        assert_eq!(d.cta_rows, 4);
        assert_eq!(d.padded_cta_rows, 16); // rounds up to 16
        assert!(d.peak_shmem.0 <= config::SM89_MAX_SHMEM);
    }

    #[test]
    fn decode_derived_rows1() {
        let dag = build_1b_dag();
        let cfg = config::FusedDecodeConfig::decode_rows1();
        let d = derived::FusedDecodeDerived::new(&dag, &cfg);
        assert_eq!(d.cta_rows, 1);
        assert_eq!(d.padded_cta_rows, 16);
        assert!(d.peak_shmem.0 <= config::SM89_MAX_SHMEM);
    }

    #[test]
    fn assign_storage_hidden_in_shmem() {
        let dag = build_1b_dag();
        let cfg = config::FusedDecodeConfig::decode_rows16();
        let d = derived::FusedDecodeDerived::new(&dag, &cfg);
        let storage = assign_decode_storage(&dag, &d, &cfg);

        // hidden_states must be in shmem at offset 0
        let hs = &storage[&BufferId("hidden_states".into())];
        assert!(
            matches!(hs, StorageStrategy::Shmem { offset, .. } if offset.0 == 0),
            "hidden_states must be Shmem(0), got {hs:?}"
        );

        // Norm outputs must be FusedIntoConsumer (shmem budget too tight for second slab)
        if let Some(normed) = storage.get(&BufferId("normed".into())) {
            assert_eq!(*normed, StorageStrategy::FusedIntoConsumer);
        }

        // QKV must be global (too large for shmem)
        if let Some(qkv) = storage.get(&BufferId("qkv".into())) {
            assert!(
                matches!(qkv, StorageStrategy::Global { .. }),
                "qkv must be Global"
            );
        }
    }
}
