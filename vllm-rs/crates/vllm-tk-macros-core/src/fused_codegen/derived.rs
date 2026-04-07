// SPDX-License-Identifier: Apache-2.0
//! Derived constants for fused prefill and decode kernels — model dims + config → numbers.

use crate::dag::ModelDag;
use crate::fused_codegen::config::{FusedDecodeConfig, FusedPrefillConfig, GemmMode, SM89_MAX_SHMEM};

/// All the numeric constants the templates need.
pub struct FusedDerived {
    pub num_stages: usize,
    pub hd: usize,
    pub id: usize,
    pub nl: usize,
    pub nah: usize,
    pub nkh: usize,
    pub hdm: usize,
    pub gqa_ratio: usize,
    pub qkv_dim: usize,
    pub rdpw: usize,
    pub hd_k_iters: usize,
    pub id_k_iters: usize,
    pub qkv_col_tiles: usize,
    pub hd_col_tiles: usize,
    pub id_col_tiles: usize,
    pub iters_per_page: usize,
    pub a_size: usize,
    pub b_size: usize,
    pub stage_size: usize,
    pub gemm_shmem: usize,
    pub rmsnorm_shmem: usize,
    pub attn_shmem: usize,
    pub total_shmem: usize,
    pub kv_tile_bytes: usize,
    pub num_threads: usize,
    /// B tile offset within a stage (cooperative mode only). 0 for redundant.
    pub b_offset: usize,
    /// Number of col tiles computed in parallel per K-loop pass.
    pub col_batch: usize,
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
        let rdpw = hd / cfg.num_warps;

        let hd_k_iters = hd / cfg.k_dim;
        let id_k_iters = id / cfg.k_dim;
        let qkv_col_tiles = qkv_dim / cfg.out_block;
        let hd_col_tiles = hd / cfg.out_block;
        let id_col_tiles = id / cfg.out_block;
        let iters_per_page = cfg.kv_page_size / 16;

        // A tile padded to 32 rows to avoid OOB cp.async writes
        let a_size = 32 * cfg.k_dim * 2;
        let b_size = cfg.out_block * cfg.k_dim * 2;

        let col_batch = cfg.col_batch;
        let num_stages = cfg.num_stages;
        let (stage_size, gemm_shmem, b_offset) = match cfg.gemm_mode {
            GemmMode::Redundant => {
                // col_batch B tiles per stage (each warp computes a different col)
                let ss = a_size + col_batch * b_size;
                (ss, num_stages * ss, 0)
            }
            GemmMode::Cooperative => {
                // Each warp owns its own A tile; B is shared
                let ss = cfg.num_warps * a_size + b_size;
                let bo = cfg.num_warps * a_size;
                (ss, num_stages * ss, bo)
            }
        };

        let rmsnorm_shmem = hd * 4 + cfg.num_warps * 4;
        let kv_tile_bytes = cfg.kv_page_size * hdm * 2;
        let attn_shmem = kv_tile_bytes * 2 * 2;

        let total_shmem = *[gemm_shmem, rmsnorm_shmem, attn_shmem]
            .iter()
            .max()
            .unwrap();

        Self {
            num_stages,
            hd,
            id,
            nl,
            nah,
            nkh,
            hdm,
            gqa_ratio,
            qkv_dim,
            rdpw,
            hd_k_iters,
            id_k_iters,
            qkv_col_tiles,
            hd_col_tiles,
            id_col_tiles,
            iters_per_page,
            a_size,
            b_size,
            stage_size,
            gemm_shmem,
            rmsnorm_shmem,
            attn_shmem,
            total_shmem,
            kv_tile_bytes,
            num_threads: cfg.num_warps * 32,
            b_offset,
            col_batch,
        }
    }
}

// ── Decode derived constants ──────────────────────────────────────────

/// All numeric constants for the fused decode kernel templates.
///
/// Row-fused architecture: each CTA owns `cta_rows` sequences.
/// Shmem is time-shared across phases (union, not sum).
pub struct FusedDecodeDerived {
    // Model dimensions (from DAG)
    pub hd: usize,
    pub id: usize,
    pub nl: usize,
    pub nah: usize,
    pub nkh: usize,
    pub hdm: usize,
    pub vs: usize,
    pub gqa_ratio: usize,
    pub qkv_dim: usize,

    // Config echo
    pub cta_rows: usize,
    pub padded_cta_rows: usize,
    pub k_dim: usize,
    pub out_block: usize,
    pub num_warps: usize,
    pub num_threads: usize,
    pub kv_page_size: usize,
    pub num_stages: usize,

    // GEMM tile sizes
    /// A tile: [16, k_dim] BF16, padded to 32 rows for cp.async safety.
    pub a_size: usize,
    /// B tile: [out_block, k_dim] BF16.
    pub b_size: usize,
    /// Cooperative stage: num_warps × a_size + b_size.
    pub stage_size: usize,
    /// B offset within a stage (cooperative: num_warps × a_size).
    pub b_offset: usize,
    /// Total GEMM shmem: num_stages × stage_size.
    pub gemm_shmem: usize,

    // K-loop iteration counts
    pub hd_k_iters: usize,
    pub id_k_iters: usize,
    pub qkv_col_tiles: usize,
    pub hd_col_tiles: usize,
    pub id_col_tiles: usize,

    // Shmem for hidden-state slab: padded_cta_rows × hd × 2 bytes (BF16).
    pub hidden_shmem: usize,

    // RMSNorm shmem: hidden_shmem + weight vector + warp scratch.
    pub rmsnorm_shmem: usize,

    // Attention shmem: Q[padded_cta_rows, hd] + K_page[kv_page_size, hdm] + V_page.
    pub attn_shmem: usize,
    pub kv_tile_bytes: usize,
    pub iters_per_page: usize,

    // Per-row metadata shmem: 5 × cta_rows × 4 bytes (i32).
    pub meta_shmem: usize,

    /// Peak shmem across all phases (must fit in SM89_MAX_SHMEM).
    pub peak_shmem: usize,
}

impl FusedDecodeDerived {
    pub fn new(dag: &ModelDag, cfg: &FusedDecodeConfig) -> Self {
        let hd = dag.params.get("HD").copied().unwrap_or(2048);
        let id = dag.params.get("ID").copied().unwrap_or(8192);
        let nl = dag.params.get("NL").copied().unwrap_or(16);
        let nah = dag.params.get("NAH").copied().unwrap_or(32);
        let nkh = dag.params.get("NKH").copied().unwrap_or(8);
        let hdm = dag.params.get("HDM").copied().unwrap_or(64);
        let vs = dag.params.get("VS").copied().unwrap_or(128256);

        let gqa_ratio = nah / nkh;
        let qkv_dim = (nah + 2 * nkh) * hdm;

        let padded_cta_rows = cfg.padded_cta_rows();

        // GEMM tile sizes — always cooperative for decode (each warp owns A slice)
        let a_size = 32 * cfg.k_dim * 2; // padded to 32 rows
        let b_size = cfg.out_block * cfg.k_dim * 2;
        let stage_size = cfg.num_warps * a_size + b_size;
        let b_offset = cfg.num_warps * a_size;
        let gemm_shmem = cfg.num_stages * stage_size;

        // K-loop iterations
        let hd_k_iters = hd / cfg.k_dim;
        let id_k_iters = id / cfg.k_dim;
        let qkv_col_tiles = qkv_dim / cfg.out_block;
        let hd_col_tiles = hd / cfg.out_block;
        let id_col_tiles = id / cfg.out_block;

        // Hidden-state slab in shmem
        let hidden_shmem = padded_cta_rows * hd * 2;

        // RMSNorm: act[padded_cta_rows, hd] BF16 + weight[hd] BF16 + scratch[num_warps] f32
        let rmsnorm_shmem = hidden_shmem + hd * 2 + cfg.num_warps * 4;

        // Attention: Q[padded_cta_rows, hd] BF16 + K_page[kv_page_size, hdm] BF16 + V_page
        let kv_tile_bytes = cfg.kv_page_size * hdm * 2;
        let attn_shmem = padded_cta_rows * hd * 2 + kv_tile_bytes * 2;
        let iters_per_page = cfg.kv_page_size / 16;

        // Per-row metadata: 5 ints per row
        let meta_shmem = 5 * cfg.cta_rows * 4;

        // Peak shmem is the maximum across all phases.
        // Note: meta_shmem is persistent (loaded once, used by all layers),
        // so it adds to every phase.  But at 320 bytes for 16 rows, it's trivial.
        let peak_shmem = [rmsnorm_shmem, gemm_shmem, attn_shmem]
            .into_iter()
            .max()
            .unwrap()
            + meta_shmem;

        assert!(
            peak_shmem <= SM89_MAX_SHMEM,
            "Decode shmem {peak_shmem} exceeds sm89 limit {SM89_MAX_SHMEM} \
             (cta_rows={}, hd={hd}, hdm={hdm}, nkh={nkh})",
            cfg.cta_rows,
        );

        Self {
            hd,
            id,
            nl,
            nah,
            nkh,
            hdm,
            vs,
            gqa_ratio,
            qkv_dim,
            cta_rows: cfg.cta_rows,
            padded_cta_rows,
            k_dim: cfg.k_dim,
            out_block: cfg.out_block,
            num_warps: cfg.num_warps,
            num_threads: cfg.num_warps * 32,
            kv_page_size: cfg.kv_page_size,
            num_stages: cfg.num_stages,
            a_size,
            b_size,
            stage_size,
            b_offset,
            gemm_shmem,
            hd_k_iters,
            id_k_iters,
            qkv_col_tiles,
            hd_col_tiles,
            id_col_tiles,
            hidden_shmem,
            rmsnorm_shmem,
            attn_shmem,
            kv_tile_bytes,
            iters_per_page,
            meta_shmem,
            peak_shmem,
        }
    }
}
