// SPDX-License-Identifier: Apache-2.0
//! Derived constants for the fused prefill kernel — model dims + config → numbers.

use crate::dag::ModelDag;
use crate::fused_codegen::config::{FusedPrefillConfig, GemmMode};
use crate::fused_codegen::units::{Bytes, Count, Dim, Iters, Tiles};

/// All the numeric constants the templates need.
pub struct FusedDerived {
    pub num_stages: Count,
    pub hd: Dim,
    pub id: Dim,
    pub nl: Count,
    pub nah: Count,
    pub nkh: Count,
    pub hdm: Dim,
    pub gqa_ratio: Count,
    pub qkv_dim: Dim,
    pub rdpw: Count,
    pub hd_k_iters: Iters,
    pub id_k_iters: Iters,
    pub qkv_col_tiles: Tiles,
    pub hd_col_tiles: Tiles,
    pub id_col_tiles: Tiles,
    pub iters_per_page: Iters,
    pub a_size: Bytes,
    pub b_size: Bytes,
    pub stage_size: Bytes,
    pub gemm_shmem: Bytes,
    pub rmsnorm_shmem: Bytes,
    pub attn_shmem: Bytes,
    pub total_shmem: Bytes,
    pub kv_tile_bytes: Bytes,
    pub num_threads: Count,
    /// B tile offset within a stage (cooperative mode only). 0 for redundant.
    pub b_offset: Bytes,
    /// Number of col tiles computed in parallel per K-loop pass.
    pub col_batch: Tiles,
    /// Per-warp GEMM accumulator M dimension (warp tile height in rows).
    pub gemm_warp_m: Dim,
    /// Number of 16-row MMA sub-tiles stacked in the M dimension (`gemm_warp_m / 16`).
    pub gemm_m_subs: Count,
}

impl FusedDerived {
    pub fn new(dag: &ModelDag, cfg: &FusedPrefillConfig) -> Self {
        let hd = dag.params.get("HD").copied().unwrap_or(2048);
        let id = dag.params.get("ID").copied().unwrap_or(8192);
        let nl = dag.params.get("NL").copied().unwrap_or(16);
        let nah = dag.params.get("NAH").copied().unwrap_or(32);
        let nkh = dag.params.get("NKH").copied().unwrap_or(8);
        let hdm = dag.params.get("HDM").copied().unwrap_or(64);

        let gqa_ratio = nah / nkh;
        let qkv_dim = (nah + 2 * nkh) * hdm;
        let rdpw = hd / cfg.num_warps.0;

        let hd_k_iters = hd / cfg.k_dim.0;
        let id_k_iters = id / cfg.k_dim.0;
        let qkv_col_tiles = qkv_dim / cfg.out_block.0;
        let hd_col_tiles = hd / cfg.out_block.0;
        let id_col_tiles = id / cfg.out_block.0;
        let iters_per_page = cfg.kv_page_size.0 / 16;

        // ── GEMM warp tile validation ──
        // Cooperative invariant: cta_rows == num_warps * gemm_warp_m.
        // Each warp owns `gemm_warp_m` rows; B shared across warps.
        let gemm_warp_m = cfg.gemm_warp_m.0;
        assert!(
            gemm_warp_m > 0 && gemm_warp_m.is_multiple_of(16),
            "gemm_warp_m must be a positive multiple of 16, got {gemm_warp_m}"
        );
        if matches!(cfg.gemm_mode, GemmMode::Cooperative) {
            assert_eq!(
                cfg.cta_rows.0,
                cfg.num_warps.0 * gemm_warp_m,
                "cooperative GEMM requires cta_rows ({}) == num_warps ({}) * gemm_warp_m ({})",
                cfg.cta_rows.0,
                cfg.num_warps.0,
                gemm_warp_m
            );
        }
        // Register-pressure envelope: accumulator is rt_fl<gemm_warp_m, out_block>,
        // storing 2 floats per thread per 16x16 sub-tile. With 255-reg cap on sm89
        // and a_reg + loop state, a safe budget is acc_elements/32 <= 128 → product
        // gemm_warp_m * out_block <= 4096. Dual-accumulator mode doubles the
        // accumulator footprint because gate_acc and up_acc live simultaneously.
        // K-stripe inner loop relaxes the constraint: live A footprint drops
        // from `gemm_warp_m * k_dim` to `gemm_warp_m * 16`, freeing roughly
        // `gemm_warp_m * (k_dim - 16) / 32` registers per thread.
        let acc_elements = gemm_warp_m * cfg.out_block.0;
        let live_acc_elements = if cfg.dual_accum_gate_up {
            2 * acc_elements
        } else {
            acc_elements
        };
        let acc_budget = if cfg.kstripe_inner { 8192 } else { 4096 };
        assert!(
            live_acc_elements <= acc_budget,
            "live accumulator elements = {} (gemm_warp_m={} * out_block={} * dual={}, kstripe={}) exceeds sm89 register budget (max {})",
            live_acc_elements,
            gemm_warp_m,
            cfg.out_block.0,
            cfg.dual_accum_gate_up,
            cfg.kstripe_inner,
            acc_budget
        );

        // A tile padded to max(gemm_warp_m, 32) rows to avoid OOB cp.async writes.
        // Per-warp A tile is st_bf<gemm_warp_m, k_dim>; legacy 32-row padding is
        // preserved when gemm_warp_m <= 32 for backward compat with existing variants.
        let a_rows_padded = if gemm_warp_m >= 32 { gemm_warp_m } else { 32 };
        let a_size = a_rows_padded * cfg.k_dim.0 * 2;
        let b_size = cfg.out_block.0 * cfg.k_dim.0 * 2;

        let col_batch = cfg.col_batch.0;
        let num_stages = cfg.num_stages.0;
        let num_warps = cfg.num_warps.0;

        // Per-stage B-tile multiplier: 2 if the gate+up phase holds both B_gate
        // and B_up live simultaneously (dual-accum mode), 1 otherwise. The
        // non-gate-up GEMM phases only ever hold 1 B tile, but they share the
        // kernel-wide shmem budget, so we size the stage for the worst case.
        let b_mul = if cfg.dual_accum_gate_up { 2 } else { 1 };

        let (stage_size, gemm_shmem, b_offset) = match cfg.gemm_mode {
            GemmMode::Redundant => {
                // col_batch B tiles per stage (each warp computes a different col)
                let ss = a_size + col_batch * b_size;
                (ss, num_stages * ss, 0)
            }
            GemmMode::Cooperative if cfg.per_warp_b => {
                // Each warp owns BOTH its A tile AND its own B tile copy.
                let per_warp = a_size + b_mul * b_size;
                let ss = num_warps * per_warp;
                (ss, num_stages * ss, a_size)
            }
            GemmMode::Cooperative => {
                // Each warp owns its own A tile; B is shared across warps.
                // In dual-accum mode, 2 B tiles (gate + up) are held per stage.
                let ss = num_warps * a_size + b_mul * b_size;
                let bo = num_warps * a_size;
                (ss, num_stages * ss, bo)
            }
            GemmMode::WarpSpecialized => {
                // (NUM_WARPS-1) consumer A tiles + 1 shared B tile per stage
                // Plus flags: STAGES * 2 ints at end
                let num_consumers = num_warps - 1;
                let ss = num_consumers * a_size + b_size;
                let bo = num_consumers * a_size;
                let flags_bytes = num_stages * 2 * 4; // 2 ints per stage
                (ss, num_stages * ss + flags_bytes, bo)
            }
        };

        let rmsnorm_shmem = hd * 4 + num_warps * 4;
        let kv_tile_bytes = cfg.kv_page_size.0 * hdm * 2;
        let attn_shmem = kv_tile_bytes * 2 * 2;

        let total_shmem = *[gemm_shmem, rmsnorm_shmem, attn_shmem]
            .iter()
            .max()
            .unwrap();

        Self {
            num_stages: cfg.num_stages,
            hd: Dim(hd),
            id: Dim(id),
            nl: Count(nl),
            nah: Count(nah),
            nkh: Count(nkh),
            hdm: Dim(hdm),
            gqa_ratio: Count(gqa_ratio),
            qkv_dim: Dim(qkv_dim),
            rdpw: Count(rdpw),
            hd_k_iters: Iters(hd_k_iters),
            id_k_iters: Iters(id_k_iters),
            qkv_col_tiles: Tiles(qkv_col_tiles),
            hd_col_tiles: Tiles(hd_col_tiles),
            id_col_tiles: Tiles(id_col_tiles),
            iters_per_page: Iters(iters_per_page),
            a_size: Bytes(a_size),
            b_size: Bytes(b_size),
            stage_size: Bytes(stage_size),
            gemm_shmem: Bytes(gemm_shmem),
            rmsnorm_shmem: Bytes(rmsnorm_shmem),
            attn_shmem: Bytes(attn_shmem),
            total_shmem: Bytes(total_shmem),
            kv_tile_bytes: Bytes(kv_tile_bytes),
            num_threads: Count(num_warps * 32),
            b_offset: Bytes(b_offset),
            col_batch: Tiles(col_batch),
            gemm_warp_m: Dim(gemm_warp_m),
            gemm_m_subs: Count(gemm_warp_m / 16),
        }
    }
}
