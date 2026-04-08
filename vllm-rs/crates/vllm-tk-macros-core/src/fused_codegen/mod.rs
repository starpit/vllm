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
pub mod units;

use askama::Template;

use crate::cuda_codegen::{
    LAUNCH_PARAMS, emit_globals_construction, emit_tensor_arg_and_globals_helper,
};
use crate::dag::ModelDag;
use crate::fused_codegen::config::{FusedPrefillConfig, GemmMode};
use crate::fused_codegen::derived::FusedDerived;
use crate::fused_codegen::templates::*;
use crate::fused_codegen::units::{Bytes, Count, Dim, Iters, Tiles};

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
        gemm_warp_m: d.gemm_warp_m,
        gemm_m_subs: d.gemm_m_subs,
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
        wgt_offset: Bytes(d.hd.0 * 2),
        scratch_offset: Bytes(d.hd.0 * 4),
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
    num_k_iters: Iters,
    num_col_tiles: Tiles,
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
    let q_end = d.nah.0 * d.hdm.0;
    let k_start = q_end;
    let k_end = q_end + d.nkh.0 * d.hdm.0;
    let v_start = k_end;
    let kv_elems = d.nkh.0 * d.hdm.0;
    RopeKvAppendCtx {
        hdm: d.hdm,
        q_end: Dim(q_end),
        k_start: Dim(k_start),
        v_start: Dim(v_start),
        kv_elems: Dim(kv_elems),
    }
    .render()
    .expect("rope_kv_append template render")
}

fn render_attention(d: &FusedDerived, cfg: &FusedPrefillConfig) -> String {
    AttentionCtx {
        attn_passes: cfg.attn_passes(),
        nkh: d.nkh,
        nah: d.nah,
        stage_sz: Bytes(d.kv_tile_bytes.0 * 2),
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
    grid_size: Count,
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
        gemm_warp_m: d.gemm_warp_m,
        gemm_m_subs: d.gemm_m_subs,
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
        kernel_suffix: "",
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
        kernel_suffix: "",
    }
    .render()
    .expect("kernel_mcta template render")
}

/// Generate a multi-CTA fused prefill kernel with fused gate+up (8 phases instead of 9).
pub fn generate_fused_prefill_mcta_fused_gateup(
    dag: &ModelDag,
    cfg: &FusedPrefillConfig,
    grid_size: Count,
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
        gemm_warp_m: d.gemm_warp_m,
        gemm_m_subs: d.gemm_m_subs,
    }
    .render()
    .expect("preamble template render");

    // Build the 8-phase fused-gateup pipeline. Honors cfg.phase_opt to apply
    // per-phase tile overrides for the small-N GEMMs (QKV, o_proj, down_proj).
    let phases: Vec<String> = build_fused_gateup_phases(dag, &d, cfg);

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
        kernel_suffix: "",
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
        kernel_suffix: "",
    }
    .render()
    .expect("kernel_mcta template render")
}

/// Generate a multi-CTA fused prefill kernel using warp-specialized producer/consumer GEMM.
pub fn generate_fused_prefill_mcta_warpspec(
    dag: &ModelDag,
    cfg: &FusedPrefillConfig,
    grid_size: Count,
) -> String {
    let d = FusedDerived::new(dag, cfg);

    let preamble = PreambleCtx {
        mode_label: "multi-CTA-warpspec",
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
        gemm_warp_m: d.gemm_warp_m,
        gemm_m_subs: d.gemm_m_subs,
    }
    .render()
    .expect("preamble template render");

    let phases = build_warpspec_phases(&d, cfg);

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
        kernel_suffix: "",
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
        "gate_silu",
        "up_mulgate",
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
        kernel_suffix: "",
    }
    .render()
    .expect("kernel_mcta template render")
}

/// Generate a polyalgorithm kernel that includes 3 variants and dispatches at runtime.
///
/// Variants (determined by the L4 sweep on LLaMA 1B, rounds 1–5):
/// - `_small`  = `rows64_k128` for seq ≤ 64 (parity with old baseline at ~10.9 ms @ seq=48).
///   Classic 64-row CTA with `k_dim=128` (half the K-loop iterations); wins for tiny
///   sequences where total work is dominated by setup overhead.
/// - `_medium` = `rows128_gemm16_dual_1stage` for 64 < seq < 256 (1.40× @ seq=128).
///   Dual-accumulator fused gate+up (single K-loop, A reused across gate/up MMAs)
///   with a 1-stage pipeline that fits 2 CTAs/SM on L4. 8 warps × 16 rows.
/// - `_large`  = `rows256_gemm32_dual_1stage` for seq ≥ 256 (1.34–1.43× @ seq=256–1024).
///   8 warps × 32 rows cooperative, dual-accumulator, 1-stage; 1 CTA/SM with half
///   the cross-CTA `mcta_barrier` trips of `_medium` plus A reuse in gate+up.
///
/// Each variant is wrapped in its own C++ namespace to isolate `PFL_*` constants
/// and type aliases. A single `fused_prefill_layer_launch` dispatches based on
/// `num_prefill_tokens`.
pub fn generate_fused_prefill_polyalgorithm(dag: &ModelDag) -> String {
    let grid_size = Count(128);

    // Three winning configs from the L4 sweep (rounds 1-5):
    //   Baseline (old polyalgo) at seq=1024: 74.47 ms
    //     seq=48  : rows64_k128               @ 10.86 ms   (parity with old small)
    //     seq=128 : rows128_gemm16_dual_1stage @ 12.97 ms  (1.40×)
    //     seq=256 : rows256_gemm32_dual_1stage @ 16.94 ms  (1.42×)
    //     seq=512 : rows256_gemm32_dual_1stage @ 30.56 ms  (1.38×)
    //     seq=1024: rows256_gemm32_dual_1stage @ 55.63 ms  (1.34×)
    //
    // _small is the classic rows64_k128 which has k_dim=128 (half the K-loop
    // iterations, less per-iter barrier overhead — wins for tiny sequences
    // where total work is dominated by setup). _medium and _large are the
    // 1-stage dual-accumulator variants (single K-loop over gate+up with A
    // reuse, 1-stage pipeline letting 2 CTAs/SM on rows128 or hosting a big
    // 256-row CTA on a single SM slot with half the mcta_barrier trips).
    // _large is now cutlass4: every GEMM phase runs through CUTLASS
    // device-side ThreadblockMma. gate_up uses two sequential CUTLASS calls
    // (up + gate-with-LinearCombinationSiluMul). 1.76× over baseline at
    // seq=1024 vs the old polyalgo's 1.34×.
    let variants: Vec<(&str, FusedPrefillConfig)> = vec![
        ("_small", FusedPrefillConfig::rows64_k128()), // seq ≤ 64
        ("_medium", FusedPrefillConfig::rows128_gemm16_dual_1stage()), // 64 < seq < 256
        (
            "_large",
            FusedPrefillConfig::rows256_gemm32_dual_1stage_cutlass4(),
        ), // seq ≥ 256
    ];

    let mut out = String::new();

    // Emit #define SM89_* and #include ONCE at file scope (before any namespace).
    // This avoids pulling C++ standard headers inside a namespace.
    let d0 = FusedDerived::new(dag, &variants[0].1);
    let header = PreambleHeaderCtx {
        mode_label: "polyalgorithm",
        nl: d0.nl,
        hd: d0.hd,
        id: d0.id,
        hdm: d0.hdm,
        nah: d0.nah,
        nkh: d0.nkh,
    }
    .render()
    .expect("preamble_header template render");
    out.push_str(&header);
    out.push('\n');

    // Each variant in its own namespace with PFL_* constants + type aliases.
    for (suffix, cfg) in &variants {
        let d = FusedDerived::new(dag, cfg);

        let constants = PreambleConstantsCtx {
            cta_rows: cfg.cta_rows,
            num_warps: cfg.num_warps,
            gqa_ratio: d.gqa_ratio,
            kv_page_size: cfg.kv_page_size,
            iters_per_page: d.iters_per_page,
            total_shmem: d.total_shmem,
            kv_tile_bytes: d.kv_tile_bytes,
            k_dim: cfg.k_dim,
            out_block: cfg.out_block,
            rdpw: d.rdpw,
            hdm: d.hdm,
            gemm_warp_m: d.gemm_warp_m,
            gemm_m_subs: d.gemm_m_subs,
        }
        .render()
        .expect("preamble_constants template render");

        let phases = build_fused_gateup_phases(dag, &d, cfg);

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

        let inner_launch = LaunchInnerMctaCtx {
            num_threads: d.num_threads,
            id_col_tiles: d.id_col_tiles,
            kernel_suffix: suffix,
        }
        .render()
        .expect("launch_inner_mcta template render");

        // KernelMctaCtx expects a preamble string — use the constants-only preamble
        let kernel = KernelMctaCtx {
            preamble: &constants,
            phases: &phases,
            launch_wrapper: &inner_launch,
            num_threads: d.num_threads,
            phase_names_str: &phase_names_str,
            num_phases: phases.len(),
            grid_size,
            kernel_suffix: suffix,
        }
        .render()
        .expect("kernel_mcta template render");

        out.push_str(&format!("namespace pfl{} {{\n", suffix));
        out.push_str(&kernel);
        out.push_str(&format!("}}  // namespace pfl{}\n\n", suffix));
    }

    // Dispatch launch wrapper at file scope
    let mut tah = String::new();
    emit_tensor_arg_and_globals_helper(&mut tah);
    let mut gc = String::new();
    emit_globals_construction(&mut gc, "    ");

    let dispatch = PolyalgorithmDispatchCtx {
        tensor_arg_helper: &tah,
        launch_params: LAUNCH_PARAMS,
        globals_construction: &gc,
    }
    .render()
    .expect("polyalgorithm dispatch render");

    out.push_str(&dispatch);
    out
}

/// Build 8-phase list using warp-specialized GEMM (producer/consumer pipeline).
fn build_warpspec_phases(d: &FusedDerived, cfg: &FusedPrefillConfig) -> Vec<String> {
    vec![
        render_rmsnorm_mcta(
            d,
            "g.hidden_states",
            "g.attn_norm_weights",
            "g.rms_rope_intermediates",
        ),
        render_gemm_warpspec_mcta(
            d,
            cfg,
            "QKV GEMM (warpspec)",
            "g.rms_rope_intermediates",
            "g.qkv_weights",
            d.hd_k_iters,
            d.qkv_col_tiles,
            EpilogueKind::Store("g.silu_out"),
        ),
        render_rope_kv_append_mcta(d),
        render_attention_mcta(d),
        render_gemm_warpspec_mcta(
            d,
            cfg,
            "o_proj + residual (warpspec)",
            "g.attn_out",
            "g.o_weights",
            d.hd_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
        render_rmsnorm_mcta(
            d,
            "g.hidden_states",
            "g.mlp_norm_weights",
            "g.rms_gate_intermediates",
        ),
        // Gate SiLU + up mulgate as separate phases (no fused gate+up warpspec template yet)
        render_gemm_warpspec_mcta(
            d,
            cfg,
            "gate GEMM + SiLU (warpspec)",
            "g.rms_gate_intermediates",
            "g.gate_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::SiLU("g.silu_out"),
        ),
        render_gemm_warpspec_mcta(
            d,
            cfg,
            "up GEMM × gate (warpspec)",
            "g.rms_gate_intermediates",
            "g.up_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::MulGate("g.silu_out"),
        ),
        render_gemm_warpspec_mcta(
            d,
            cfg,
            "down_proj + residual (warpspec)",
            "g.silu_out",
            "g.down_weights",
            d.id_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
    ]
}

/// Build the 8-phase (fused gate+up) phase list for a variant.
/// When `cfg.phase_opt` is true, the small-N GEMMs (QKV, o_proj, down_proj)
/// get a per-phase tile override (gemm_warp_m=16, out_block=32) wrapped in a
/// C++ scope-shadow block. gate_up keeps the kernel-wide shape.
fn build_fused_gateup_phases(
    dag: &ModelDag,
    d: &FusedDerived,
    cfg: &FusedPrefillConfig,
) -> Vec<String> {
    let small_n_ovr = PhaseTileOverride {
        gemm_warp_m: 16,
        out_block: 32,
    };
    let qkv_phase = if cfg.cutlass_qkv_o {
        // CUTLASS path. QKV writes to silu_out (used as scratch). No residual,
        // but the LinearCombination(α=1, β=0) form gives plain "store".
        // For now we use the same residual-add template path but with the
        // "C" iterator pointing at silu_out and beta=0 — this lets us reuse
        // gemm_cutlass_mcta.cu without forking for store-only.
        render_gemm_cutlass_mcta(
            "QKV GEMM (CUTLASS)",
            "g.rms_rope_intermediates.raw_ptr",
            "(g.qkv_weights.raw_ptr + (size_t)layer * (size_t)g.qkv_weights.cols() * (size_t)g.hidden_dim)",
            "g.silu_out.raw_ptr",
            "q_size",
            d.hd.0,      // K = HD = 2048
            d.qkv_dim.0, // N = QKV_DIM = 2304
            "0.0f",      // beta=0: pure store, no residual
        )
    } else if cfg.phase_opt {
        render_gemm_mcta_override(
            dag,
            cfg,
            small_n_ovr,
            "QKV GEMM (phase-opt)",
            "g.rms_rope_intermediates",
            "g.qkv_weights",
            EpilogueKind::Store("g.silu_out"),
        )
    } else {
        render_gemm_mcta(
            d,
            cfg,
            "QKV GEMM",
            "g.rms_rope_intermediates",
            "g.qkv_weights",
            d.hd_k_iters,
            d.qkv_col_tiles,
            EpilogueKind::Store("g.silu_out"),
        )
    };
    let o_phase = if cfg.cutlass_qkv_o {
        render_gemm_cutlass_mcta(
            "o_proj + residual (CUTLASS)",
            "g.attn_out.raw_ptr",
            "(g.o_weights.raw_ptr + (size_t)layer * (size_t)g.hidden_dim * (size_t)g.hidden_dim)",
            "g.hidden_states.raw_ptr",
            "q_size",
            d.hd.0, // K = HD
            d.hd.0, // N = HD
            "1.0f", // beta=1: residual add
        )
    } else if cfg.phase_opt {
        render_gemm_mcta_override(
            dag,
            cfg,
            small_n_ovr,
            "o_proj + residual (phase-opt)",
            "g.attn_out",
            "g.o_weights",
            EpilogueKind::ResidualAdd("g.hidden_states"),
        )
    } else {
        render_gemm_mcta(
            d,
            cfg,
            "o_proj + residual",
            "g.attn_out",
            "g.o_weights",
            d.hd_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        )
    };
    let down_phase = if cfg.cutlass_down_proj {
        // CUTLASS device-side ThreadblockMma. The hand-rolled cooperative
        // GEMM is bypassed for this phase. Output is written via a manual
        // residual-add epilogue inside the template body.
        render_gemm_cutlass_mcta(
            "down_proj + residual (CUTLASS)",
            // A = silu_out [seq, ID]
            "g.silu_out.raw_ptr",
            // B = down_weights[layer] [HD, ID]. Per-layer offset added at runtime.
            "(g.down_weights.raw_ptr + (size_t)layer * (size_t)g.hidden_dim * (size_t)g.intermediate_dim)",
            "g.hidden_states.raw_ptr",
            "q_size",
            d.id.0, // K = intermediate_dim
            d.hd.0, // N = hidden_dim
            "1.0f", // beta=1: residual add
        )
    } else if cfg.phase_opt {
        render_gemm_mcta_override(
            dag,
            cfg,
            small_n_ovr,
            "down_proj + residual (phase-opt)",
            "g.silu_out",
            "g.down_weights",
            EpilogueKind::ResidualAdd("g.hidden_states"),
        )
    } else {
        render_gemm_mcta(
            d,
            cfg,
            "down_proj + residual",
            "g.silu_out",
            "g.down_weights",
            d.id_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        )
    };
    let gate_up_phase = if cfg.cutlass_gate_up {
        // Two CUTLASS calls: up first (β=0 plain store to silu_out), then
        // gate with the SiluMul epilogue that reads silu_out (= up output)
        // and writes silu_out = silu(gate_acc) * up_loaded.
        let up_call = render_gemm_cutlass_mcta(
            "fused gate+up :: UP (CUTLASS, plain store)",
            "g.rms_gate_intermediates.raw_ptr",
            "(g.up_weights.raw_ptr + (size_t)layer * (size_t)g.intermediate_dim * (size_t)g.hidden_dim)",
            "g.silu_out.raw_ptr",
            "q_size",
            d.hd.0, // K = HD
            d.id.0, // N = ID
            "0.0f", // β=0
        );
        let gate_call = render_gemm_cutlass_silumul_mcta(
            "fused gate+up :: GATE (CUTLASS, silu*source)",
            "g.rms_gate_intermediates.raw_ptr",
            "(g.gate_weights.raw_ptr + (size_t)layer * (size_t)g.intermediate_dim * (size_t)g.hidden_dim)",
            "g.silu_out.raw_ptr",
            "q_size",
            d.hd.0,
            d.id.0,
        );
        format!("{up_call}\n{gate_call}")
    } else {
        render_gemm_gate_up_mcta(
            d,
            cfg,
            "fused gate+up",
            "g.rms_gate_intermediates",
            "g.gate_weights",
            "g.up_weights",
            "g.silu_out",
            d.hd_k_iters,
            d.id_col_tiles,
        )
    };
    vec![
        render_rmsnorm_mcta(
            d,
            "g.hidden_states",
            "g.attn_norm_weights",
            "g.rms_rope_intermediates",
        ),
        qkv_phase,
        render_rope_kv_append_mcta(d),
        render_attention_mcta(d),
        o_phase,
        render_rmsnorm_mcta(
            d,
            "g.hidden_states",
            "g.mlp_norm_weights",
            "g.rms_gate_intermediates",
        ),
        gate_up_phase,
        down_phase,
    ]
}

/// Build the 9-phase (separate gate/up) phase list for a variant.
fn build_separate_gateup_phases(d: &FusedDerived, cfg: &FusedPrefillConfig) -> Vec<String> {
    vec![
        render_rmsnorm_mcta(
            d,
            "g.hidden_states",
            "g.attn_norm_weights",
            "g.rms_rope_intermediates",
        ),
        render_gemm_mcta(
            d,
            cfg,
            "QKV GEMM",
            "g.rms_rope_intermediates",
            "g.qkv_weights",
            d.hd_k_iters,
            d.qkv_col_tiles,
            EpilogueKind::Store("g.silu_out"),
        ),
        render_rope_kv_append_mcta(d),
        render_attention_mcta(d),
        render_gemm_mcta(
            d,
            cfg,
            "o_proj + residual",
            "g.attn_out",
            "g.o_weights",
            d.hd_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
        render_rmsnorm_mcta(
            d,
            "g.hidden_states",
            "g.mlp_norm_weights",
            "g.rms_gate_intermediates",
        ),
        render_gemm_mcta(
            d,
            cfg,
            "gate GEMM + SiLU",
            "g.rms_gate_intermediates",
            "g.gate_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::SiLU("g.silu_out"),
        ),
        render_gemm_mcta(
            d,
            cfg,
            "up GEMM × gate",
            "g.rms_gate_intermediates",
            "g.up_weights",
            d.hd_k_iters,
            d.id_col_tiles,
            EpilogueKind::MulGate("g.silu_out"),
        ),
        render_gemm_mcta(
            d,
            cfg,
            "down_proj + residual",
            "g.silu_out",
            "g.down_weights",
            d.id_k_iters,
            d.hd_col_tiles,
            EpilogueKind::ResidualAdd("g.hidden_states"),
        ),
    ]
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
        wgt_offset: Bytes(d.hd.0 * 2),
        scratch_offset: Bytes(d.hd.0 * 4),
    }
    .render()
    .expect("rmsnorm_mcta template render")
}

/// Per-phase tile override. When supplied, the GEMM phase is emitted with
/// its own gemm_warp_m / out_block (and matching shmem layout), wrapped in a
/// C++ block that shadows the kernel-wide PFL_* constants and tile typedefs.
/// num_warps stays at the kernel-wide value (the launch geometry can't change
/// per-phase). cta_rows is therefore num_warps × override.gemm_warp_m.
#[derive(Clone, Copy, Debug)]
pub struct PhaseTileOverride {
    pub gemm_warp_m: u32,
    pub out_block: u32,
}

/// Build a per-phase config + derived by overriding the small-N tile dims.
/// Used by render_gemm_mcta to emit a phase block with shadowed PFL_* values.
fn phase_override_cfg(base: &FusedPrefillConfig, ovr: PhaseTileOverride) -> FusedPrefillConfig {
    FusedPrefillConfig {
        cta_rows: Dim(base.num_warps.0 * ovr.gemm_warp_m as usize),
        out_block: Dim(ovr.out_block as usize),
        gemm_warp_m: Dim(ovr.gemm_warp_m as usize),
        // Per-phase override is only meaningful for non-dual_accum GEMMs
        // (QKV, o_proj, down_proj). gate_up uses its own dual_accum path.
        dual_accum_gate_up: false,
        ..base.clone()
    }
}

/// Emit a C++ scope block that shadows the kernel-wide PFL_* constants and
/// tile typedefs with phase-local versions, so a phase body using PFL_GEMM_M
/// etc. resolves to the per-phase value. The block is closed by `phase_override_close`.
fn phase_override_open(ovr: PhaseTileOverride) -> String {
    let m = ovr.gemm_warp_m;
    let m_subs = m / 16;
    let out = ovr.out_block;
    let n_tiles = out / 16;
    format!(
        "    // ── Phase tile override: gemm_warp_m={m} out_block={out} ──\n\
         {{\n\
         constexpr int PFL_GEMM_M = {m};\n\
         constexpr int PFL_GEMM_M_SUBS = {m_subs};\n\
         constexpr int PFL_OUT_BLOCK = {out};\n\
         constexpr int PFL_N_TILES = {n_tiles};\n\
         constexpr int PFL_CTA_ROWS = PFL_NUM_WARPS * PFL_GEMM_M;\n\
         using pfl_a_st = st_bf<PFL_GEMM_M, PFL_K_DIM>;\n\
         using pfl_b_st = st_bf<PFL_OUT_BLOCK, PFL_K_DIM>;\n\
         using pfl_acc_rt = rt_fl<PFL_GEMM_M, PFL_OUT_BLOCK>;\n\
         using pfl_a_rt = rt_bf<PFL_GEMM_M, PFL_K_DIM>;\n\
         using pfl_b_slice_st = st_bf<16, PFL_K_DIM>;\n"
    )
}

fn phase_override_close() -> String {
    "    }\n".to_string()
}

#[allow(clippy::too_many_arguments)]
fn render_gemm_mcta(
    d: &FusedDerived,
    cfg: &FusedPrefillConfig,
    phase_comment: &str,
    input_global: &str,
    weight_global: &str,
    num_k_iters: Iters,
    num_col_tiles: Tiles,
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
        kstripe: cfg.kstripe_inner,
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
        per_warp_b: cfg.per_warp_b,
    }
    .render()
    .expect("gemm_mcta template render")
}

/// Render a CUTLASS-backed GEMM phase. Replaces the hand-rolled cooperative
/// GEMM with `cutlass::gemm::threadblock::ThreadblockMma` instantiated against
/// our shmem region. Currently used for `down_proj` only as a proof-of-concept
/// to close the cuBLAS gap on the worst-performing single GEMM phase.
///
/// `m_dim_expr`: C++ expression evaluated at runtime for the M extent
/// (e.g. "q_size" for our prefill kernel).
/// `k_dim_value` / `n_dim_value`: compile-time K and N values from the model dim.
#[allow(clippy::too_many_arguments)]
fn render_gemm_cutlass_mcta(
    phase_comment: &str,
    a_ptr_expr: &str,
    b_ptr_expr: &str,
    out_ptr_expr: &str,
    m_dim_expr: &str,
    k_dim_value: usize,
    n_dim_value: usize,
    beta_literal: &str,
) -> String {
    GemmCutlassMctaCtx {
        phase_comment,
        a_ptr_expr,
        b_ptr_expr,
        out_ptr_expr,
        m_dim: m_dim_expr,
        k_dim_value,
        n_dim_value,
        beta_literal,
        silu_mul: false,
    }
    .render()
    .expect("gemm_cutlass_mcta template render")
}

#[allow(clippy::too_many_arguments)]
fn render_gemm_cutlass_silumul_mcta(
    phase_comment: &str,
    a_ptr_expr: &str,
    b_ptr_expr: &str,
    out_ptr_expr: &str,
    m_dim_expr: &str,
    k_dim_value: usize,
    n_dim_value: usize,
) -> String {
    GemmCutlassMctaCtx {
        phase_comment,
        a_ptr_expr,
        b_ptr_expr,
        out_ptr_expr,
        m_dim: m_dim_expr,
        k_dim_value,
        n_dim_value,
        beta_literal: "1.0f", // unused in silu_mul branch but kept for ctx parity
        silu_mul: true,
    }
    .render()
    .expect("gemm_cutlass_mcta silumul template render")
}

/// Render a GEMM phase with a per-phase tile override applied. The kernel-wide
/// `cfg` provides the launch geometry (num_warps, num_stages); the override
/// supplies gemm_warp_m and out_block. Phase shmem comes from a fresh
/// FusedDerived computed against the override-augmented cfg.
#[allow(clippy::too_many_arguments)]
fn render_gemm_mcta_override(
    dag: &ModelDag,
    base_cfg: &FusedPrefillConfig,
    ovr: PhaseTileOverride,
    phase_comment: &str,
    input_global: &str,
    weight_global: &str,
    epilogue: EpilogueKind<'_>,
) -> String {
    let phase_cfg = phase_override_cfg(base_cfg, ovr);
    let phase_d = FusedDerived::new(dag, &phase_cfg);
    // Pick num_k_iters / num_col_tiles based on what this phase actually
    // operates on. Caller will encode that via the input/weight global names,
    // but we still need to compute them from the phase_cfg.
    // We map them by inspecting the weight_global string (cheap heuristic).
    let (num_k_iters, num_col_tiles) = phase_dims(&phase_d, weight_global);
    let body = render_gemm_mcta(
        &phase_d,
        &phase_cfg,
        phase_comment,
        input_global,
        weight_global,
        num_k_iters,
        num_col_tiles,
        epilogue,
    );
    let mut out = String::new();
    out.push_str(&phase_override_open(ovr));
    out.push_str(&body);
    out.push_str(&phase_override_close());
    out
}

/// Map the weight global name to (num_k_iters, num_col_tiles) for the
/// phase's GEMM. Used by render_gemm_mcta_override to compute per-phase
/// loop bounds against the per-phase derived (which has the override's
/// k_dim and out_block).
fn phase_dims(d: &FusedDerived, weight_global: &str) -> (Iters, Tiles) {
    if weight_global.contains("qkv_weights") {
        (d.hd_k_iters, d.qkv_col_tiles)
    } else if weight_global.contains("o_weights") || weight_global.contains("o_proj") {
        (d.hd_k_iters, d.hd_col_tiles)
    } else if weight_global.contains("down_weights") {
        (d.id_k_iters, d.hd_col_tiles)
    } else if weight_global.contains("gate_weights") || weight_global.contains("up_weights") {
        (d.hd_k_iters, d.id_col_tiles)
    } else {
        // Fall back: use the same as a generic HD-input GEMM.
        (d.hd_k_iters, d.hd_col_tiles)
    }
}

#[allow(clippy::too_many_arguments)]
fn render_gemm_warpspec_mcta(
    d: &FusedDerived,
    cfg: &FusedPrefillConfig,
    phase_comment: &str,
    input_global: &str,
    weight_global: &str,
    num_k_iters: Iters,
    num_col_tiles: Tiles,
    epilogue: EpilogueKind<'_>,
) -> String {
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

    GemmWarpspecMctaCtx {
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
        num_stages: cfg.num_stages,
    }
    .render()
    .expect("gemm_warpspec_mcta template render")
}

fn render_rope_kv_append_mcta(d: &FusedDerived) -> String {
    let q_end = d.nah.0 * d.hdm.0;
    let k_start = q_end;
    let k_end = q_end + d.nkh.0 * d.hdm.0;
    let v_start = k_end;
    let kv_elems = d.nkh.0 * d.hdm.0;
    RopeKvAppendMctaCtx {
        hdm: d.hdm,
        q_end: Dim(q_end),
        k_start: Dim(k_start),
        v_start: Dim(v_start),
        kv_elems: Dim(kv_elems),
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
    num_k_iters: Iters,
    num_col_tiles: Tiles,
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
        per_warp_b: cfg.per_warp_b,
        dual_accum: cfg.dual_accum_gate_up,
        up_b_offset: Bytes(d.b_offset.0 + d.b_size.0),
        col_fixed: cfg.col_fixed_schedule,
        kstripe: cfg.kstripe_inner,
    }
    .render()
    .expect("gemm_gate_up_mcta template render")
}

fn render_attention_mcta(d: &FusedDerived) -> String {
    AttentionMctaCtx {
        nkh: d.nkh,
        nah: d.nah,
        stage_sz: Bytes(d.kv_tile_bytes.0 * 2),
    }
    .render()
    .expect("attention_mcta template render")
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
        let v2 = generate_fused_prefill_mcta(&dag, &FusedPrefillConfig::rows16_col4(), Count(128));

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
        let v2 = generate_fused_prefill_mcta_fused_gateup(
            &dag,
            &FusedPrefillConfig::rows128(),
            Count(128),
        );

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

        let v2_mcta =
            generate_fused_prefill_mcta(&dag, &FusedPrefillConfig::rows16_col4(), Count(128));
        let v2_mcta_fused = generate_fused_prefill_mcta_fused_gateup(
            &dag,
            &FusedPrefillConfig::rows128(),
            Count(128),
        );

        std::fs::write("/tmp/fused_v1.cu", &v1).ok();
        std::fs::write("/tmp/fused_v2_16.cu", &v2_16).ok();
        std::fs::write("/tmp/fused_v2_128.cu", &v2_128).ok();
        std::fs::write("/tmp/fused_v2_mcta.cu", &v2_mcta).ok();
        std::fs::write("/tmp/fused_v2_mcta_fused.cu", &v2_mcta_fused).ok();

        eprintln!("v1: {} lines", v1.lines().count());
        eprintln!("v2_16: {} lines", v2_16.lines().count());
        eprintln!("v2_128: {} lines", v2_128.lines().count());
        eprintln!("v2_mcta: {} lines", v2_mcta.lines().count());
        eprintln!("v2_mcta_fused: {} lines", v2_mcta_fused.lines().count());
    }

    #[test]
    fn renders_polyalgorithm_kernel() {
        let dag = build_1b_dag();
        let poly = generate_fused_prefill_polyalgorithm(&dag);

        // Three namespaces
        assert!(
            poly.contains("namespace pfl_small {"),
            "missing small namespace"
        );
        assert!(
            poly.contains("namespace pfl_medium {"),
            "missing medium namespace"
        );
        assert!(
            poly.contains("namespace pfl_large {"),
            "missing large namespace"
        );

        // Three kernel functions
        assert!(
            poly.contains("fused_prefill_layer_small(const globals g"),
            "missing small kernel"
        );
        assert!(
            poly.contains("fused_prefill_layer_medium(const globals g"),
            "missing medium kernel"
        );
        assert!(
            poly.contains("fused_prefill_layer_large(const globals g"),
            "missing large kernel"
        );

        // Three inner launch functions
        assert!(
            poly.contains("fused_prefill_layer_small_launch_inner"),
            "missing small inner launch"
        );
        assert!(
            poly.contains("fused_prefill_layer_medium_launch_inner"),
            "missing medium inner launch"
        );
        assert!(
            poly.contains("fused_prefill_layer_large_launch_inner"),
            "missing large inner launch"
        );

        // Dispatch wrapper
        assert!(
            poly.contains("fused_prefill_layer_launch("),
            "missing dispatch launch"
        );
        assert!(
            poly.contains("num_prefill_tokens <= 128"),
            "missing small threshold"
        );
        assert!(
            poly.contains("num_prefill_tokens < 1024"),
            "missing medium threshold"
        );

        // Different PFL_CTA_ROWS in different namespaces
        assert!(poly.contains("PFL_CTA_ROWS = 64"), "missing 64-row variant");
        assert!(
            poly.contains("PFL_CTA_ROWS = 128"),
            "missing 128-row variant"
        );

        // Different PFL_K_DIM values
        assert!(poly.contains("PFL_K_DIM = 128"), "missing k128 variant");
        assert!(poly.contains("PFL_K_DIM = 64"), "missing k64 variant");

        // Wide variant has different out_block
        assert!(
            poly.contains("PFL_OUT_BLOCK = 128"),
            "missing wide out_block"
        );
        assert!(
            poly.contains("PFL_OUT_BLOCK = 64"),
            "missing standard out_block"
        );

        std::fs::write("/tmp/fused_poly.cu", &poly).ok();
        eprintln!("polyalgorithm: {} lines", poly.lines().count());
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
}
