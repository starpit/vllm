// SPDX-License-Identifier: Apache-2.0
//! Derived constants for the fused prefill kernel — model dims + config → numbers.

use crate::dag::ModelDag;
use crate::fused_codegen::config::{FusedPrefillConfig, GemmMode};

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
